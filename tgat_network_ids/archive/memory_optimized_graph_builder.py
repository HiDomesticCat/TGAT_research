#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
記憶體優化版動態圖結構建立模組

此模組是 graph_builder.py 的記憶體優化版本，提供以下功能：
1. 子圖採樣
2. 稀疏表示
3. 批次處理邊
4. 定期清理不活躍節點
5. 記憶體使用監控
"""

import numpy as np
import pandas as pd
import torch
import dgl
import logging
from collections import defaultdict
from tqdm import tqdm
import time
import gc
import random
from datetime import datetime
import os
import pickle

# 導入記憶體優化工具
from ..utils.memory_utils import (
    clean_memory, memory_usage_decorator, print_memory_usage,
    get_memory_usage, print_optimization_suggestions
)

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryOptimizedDynamicNetworkGraph:
    """記憶體優化版動態網路圖結構類別"""
    
    def __init__(self, config, device='cuda'):
        """
        初始化記憶體優化版動態網路圖結構
        
        參數:
            config (dict): 配置字典，包含以下鍵：
                - graph.temporal_window: 時間窗口大小（秒）
                - graph.use_subgraph_sampling: 是否使用子圖採樣
                - graph.max_nodes_per_subgraph: 子圖節點數量上限
                - graph.max_edges_per_subgraph: 子圖邊數量上限
                - graph.use_sparse_representation: 是否使用稀疏表示
                - graph.edge_batch_size: 邊批次處理大小
                - graph.prune_inactive_nodes: 是否定期清理不活躍節點
                - graph.inactive_threshold: 節點不活躍閾值（秒）
            device (str): 計算裝置 ('cpu' 或 'cuda')
        """
        # 從配置中提取圖相關設置
        graph_config = config.get('graph', {})
        self.temporal_window = graph_config.get('temporal_window', 300)  # 5分鐘
        self.use_subgraph_sampling = graph_config.get('use_subgraph_sampling', True)
        self.max_nodes_per_subgraph = graph_config.get('max_nodes_per_subgraph', 5000)
        self.max_edges_per_subgraph = graph_config.get('max_edges_per_subgraph', 10000)
        self.use_sparse_representation = graph_config.get('use_sparse_representation', True)
        self.edge_batch_size = graph_config.get('edge_batch_size', 5000)
        self.prune_inactive_nodes = graph_config.get('prune_inactive_nodes', True)
        self.inactive_threshold = graph_config.get('inactive_threshold', 1800)  # 30分鐘
        
        # 設置裝置
        # 檢查 CUDA 是否可用
        if device == 'cuda' and torch.cuda.is_available():
            try:
                # 嘗試初始化 CUDA (使用一個簡單的 CUDA 操作來觸發初始化)
                test_tensor = torch.zeros(1, device='cuda')
                del test_tensor  # 釋放測試張量
                self.device = device
                logger.info(f"成功初始化 CUDA，使用 GPU 設備")
            except Exception as e:
                logger.warning(f"CUDA 初始化失敗，使用 CPU 代替: {str(e)}")
                self.device = 'cpu'
        else:
            logger.info(f"CUDA 不可用，使用 CPU 設備")
            self.device = 'cpu'
        
        logger.info(f"最終使用裝置: {self.device}")
        
        # 初始化圖結構
        self.g = None  # 主圖
        self.node_features = {}  # 節點特徵
        self.edge_timestamps = {}  # 邊的時間戳記
        self.edge_features = {}  # 邊特徵
        self.node_timestamps = {}  # 節點時間戳記
        self.node_labels = {}  # 節點標籤
        self.current_time = 0  # 當前時間
        self.temporal_g = None  # 時間子圖
        
        # 記錄已存在的節點和邊，用於快速查詢
        self.existing_nodes = set()
        self.existing_edges = set()
        
        # 用於動態跟踪每個源IP到目標IP的連接
        self.src_to_dst = defaultdict(set)
        self.dst_to_src = defaultdict(set)
        
        # 特徵維度
        self.node_feat_dim = None
        self.edge_feat_dim = None
        
        # 記錄節點最後活躍時間，用於清理不活躍節點
        self.node_last_active = {}
        
        # 記錄上次清理時間
        self.last_pruning_time = time.time()
        
        # 初始化一個空圖
        self._init_graph()
        
        logger.info(f"初始化記憶體優化版動態網路圖結構: 時間窗口={self.temporal_window}秒")
        logger.info(f"子圖採樣設置: 啟用={self.use_subgraph_sampling}, 最大節點數={self.max_nodes_per_subgraph}, 最大邊數={self.max_edges_per_subgraph}")
        logger.info(f"稀疏表示設置: 啟用={self.use_sparse_representation}, 邊批次大小={self.edge_batch_size}")
        logger.info(f"節點清理設置: 啟用={self.prune_inactive_nodes}, 不活躍閾值={self.inactive_threshold}秒")
    
    def _init_graph(self):
        """初始化一個空圖"""
        self.g = dgl.graph(([],  # 源節點
                           []),  # 目標節點
                          num_nodes=0,
                          idtype=torch.int64,
                          device='cpu')  # 初始在 CPU 上創建，需要時再移至 GPU
        
        logger.info(f"初始化空圖")
    
    @memory_usage_decorator
    def add_nodes(self, node_ids, features, timestamps, labels=None):
        """
        添加節點到圖
        
        參數:
            node_ids (list): 節點ID列表
            features (np.ndarray): 節點特徵矩陣 [n_nodes, feat_dim]
            timestamps (list): 節點時間戳記列表
            labels (list, optional): 節點標籤列表
        """
        # 設置節點特徵維度 (首次添加時)
        if self.node_feat_dim is None and features is not None:
            self.node_feat_dim = features.shape[1]
        
        # 轉換為張量
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features)
        
        # 獲取新節點 (排除已存在的)
        new_nodes = [nid for nid in node_ids if nid not in self.existing_nodes]
        
        if not new_nodes:
            return
        
        # 更新圖的節點數
        current_num_nodes = self.g.num_nodes()
        new_num_nodes = current_num_nodes + len(new_nodes)
        self.g.add_nodes(len(new_nodes))
        
        # 更新節點特徵字典
        node_idx_map = {nid: i + current_num_nodes for i, nid in enumerate(new_nodes)}
        
        for i, nid in enumerate(new_nodes):
            idx = node_idx_map[nid]
            self.node_features[nid] = features[i]
            self.node_timestamps[nid] = timestamps[i]
            self.node_last_active[nid] = time.time()  # 記錄當前時間作為最後活躍時間
            
            if labels is not None:
                if isinstance(labels, pd.Series):
                    # 如果是 Series，通過位置訪問而非索引
                    self.node_labels[nid] = labels.iloc[i] if i < len(labels) else -1
                else:
                    # 如果是數組或列表，直接索引
                    self.node_labels[nid] = labels[i] if i < len(labels) else -1
            self.existing_nodes.add(nid)
            
        # 更新當前時間為最新的時間戳記
        if timestamps:
            self.current_time = max(self.current_time, max(timestamps))
        
        logger.info(f"添加 {len(new_nodes)} 個新節點，當前共 {new_num_nodes} 個節點")
        
        # 定期清理不活躍節點
        if self.prune_inactive_nodes and time.time() - self.last_pruning_time > 300:  # 每5分鐘檢查一次
            self._prune_inactive_nodes()
    
    def _prune_inactive_nodes(self):
        """高效清理不活躍節點"""
        if not self.prune_inactive_nodes:
            return
        
        # 檢查上次清理時間，避免頻繁清理
        current_time = time.time()
        if current_time - self.last_pruning_time < 600:  # 至少10分鐘才清理一次
            return
            
        # 找出不活躍節點
        inactive_nodes = []
        for nid, last_active in self.node_last_active.items():
            if current_time - last_active > self.inactive_threshold:
                inactive_nodes.append(nid)
        
        # 如果沒有不活躍節點或不活躍節點比例過低，不清理
        inactive_ratio = len(inactive_nodes) / max(1, len(self.existing_nodes))
        if not inactive_nodes or inactive_ratio < 0.1:  # 不活躍節點少於10%時不清理
            self.last_pruning_time = current_time  # 仍更新清理時間
            return
        
        logger.info(f"清理 {len(inactive_nodes)} 個不活躍節點 (佔比 {inactive_ratio:.1%})")
        
        # 獲取活躍節點集合
        active_nodes = list(self.existing_nodes - set(inactive_nodes))
        active_nodes_set = set(active_nodes)
        
        # 檢查活躍邊 - 只保留連接兩個活躍節點的邊
        active_edges = []
        for src, dst in self.existing_edges:
            if src in active_nodes_set and dst in active_nodes_set:
                active_edges.append((src, dst))
        
        # 建立高效的節點ID映射 - 使用連續的整數索引
        node_map = {old_id: new_id for new_id, old_id in enumerate(active_nodes)}
        
        # 創建新圖但不立即添加所有邊
        new_g = dgl.graph(([],  # 源節點
                          []),  # 目標節點
                         num_nodes=len(active_nodes),
                         idtype=torch.int64,
                         device='cpu')
        
        # 批量添加邊以提高效率
        if active_edges:
            # 將原始節點ID映射到新的連續節點ID
            new_src = [node_map[e[0]] for e in active_edges]
            new_dst = [node_map[e[1]] for e in active_edges]
            
            # 批量添加邊
            new_g.add_edges(new_src, new_dst)
        
        # 高效更新節點特徵和相關數據 - 使用批處理而非逐個處理
        # 只遍歷一次活躍節點列表
        new_node_features = {}
        new_node_timestamps = {}
        new_node_labels = {}
        new_node_last_active = {}
        new_edge_features = {}
        new_edge_timestamps = {}
        new_existing_edges = set()
        
        # 節點數據轉換
        for old_id in active_nodes:
            new_id = node_map[old_id]
            if old_id in self.node_features:
                new_node_features[new_id] = self.node_features[old_id]
            if old_id in self.node_timestamps:
                new_node_timestamps[new_id] = self.node_timestamps[old_id]
            if old_id in self.node_labels:
                new_node_labels[new_id] = self.node_labels[old_id]
            if old_id in self.node_last_active:
                new_node_last_active[new_id] = self.node_last_active[old_id]
        
        # 邊數據轉換 - 直接使用映射後的邊
        for i, (old_src, old_dst) in enumerate(active_edges):
            new_src = node_map[old_src]
            new_dst = node_map[old_dst]
            new_edge = (new_src, new_dst)
            old_edge = (old_src, old_dst)
            
            if old_edge in self.edge_features:
                new_edge_features[new_edge] = self.edge_features[old_edge]
            if old_edge in self.edge_timestamps:
                new_edge_timestamps[new_edge] = self.edge_timestamps[old_edge]
                
            new_existing_edges.add(new_edge)
        
        # 批量更新所有數據結構
        self.g = new_g
        self.node_features = new_node_features
        self.node_timestamps = new_node_timestamps
        self.node_labels = new_node_labels
        self.node_last_active = new_node_last_active
        self.edge_features = new_edge_features
        self.edge_timestamps = new_edge_timestamps
        self.existing_edges = new_existing_edges
        self.existing_nodes = set(active_nodes)
        
        # 高效重建源到目標的映射
        self.src_to_dst = defaultdict(set)
        self.dst_to_src = defaultdict(set)
        
        # 一次遍歷處理所有新邊
        for (src, dst) in self.existing_edges:
            self.src_to_dst[src].add(dst)
            self.dst_to_src[dst].add(src)
        
        # 更新最後清理時間
        self.last_pruning_time = current_time
        
        # 清理記憶體
        clean_memory()
        
        logger.info(f"清理完成，當前共 {len(self.existing_nodes)} 個節點，{len(self.existing_edges)} 條邊")
    
    @memory_usage_decorator
    def add_edges(self, src_nodes, dst_nodes, timestamps, edge_feats=None):
        """
        添加邊到圖
        
        參數:
            src_nodes (list): 源節點ID列表
            dst_nodes (list): 目標節點ID列表
            timestamps (list): 邊的時間戳記列表
            edge_feats (list, optional): 邊特徵列表 [n_edges, feat_dim]
        """
        if len(src_nodes) != len(dst_nodes) or len(src_nodes) != len(timestamps):
            raise ValueError("源節點、目標節點和時間戳記列表長度必須相同")
        
        if not src_nodes:
            return
        
        # 設置邊特徵維度 (首次添加時)
        if self.edge_feat_dim is None and edge_feats is not None:
            self.edge_feat_dim = len(edge_feats[0])
        
        # 獲取新邊 (排除已存在的)
        new_edges = []
        new_src = []
        new_dst = []
        new_timestamps = []
        new_edge_feats = []
        
        for i, (src, dst) in enumerate(zip(src_nodes, dst_nodes)):
            edge_key = (src, dst)
            if edge_key not in self.existing_edges and src in self.existing_nodes and dst in self.existing_nodes:
                new_edges.append(edge_key)
                new_src.append(src)
                new_dst.append(dst)
                new_timestamps.append(timestamps[i])
                if edge_feats is not None:
                    new_edge_feats.append(edge_feats[i])
                
                # 更新節點最後活躍時間
                self.node_last_active[src] = time.time()
                self.node_last_active[dst] = time.time()
        
        if not new_edges:
            return
        
        # 獲取節點的圖索引
        src_indices = []
        dst_indices = []
        
        for src, dst in zip(new_src, new_dst):
            src_idx = list(self.existing_nodes).index(src)
            dst_idx = list(self.existing_nodes).index(dst)
            src_indices.append(src_idx)
            dst_indices.append(dst_idx)
        
        # 添加新邊到圖
        self.g.add_edges(src_indices, dst_indices)
        
        # 更新邊特徵和時間戳記
        for i, edge_key in enumerate(new_edges):
            self.edge_timestamps[edge_key] = new_timestamps[i]
            if edge_feats is not None:
                self.edge_features[edge_key] = new_edge_feats[i]
            self.existing_edges.add(edge_key)
            
            # 更新來源到目標的映射
            src, dst = edge_key
            self.src_to_dst[src].add(dst)
            self.dst_to_src[dst].add(src)
        
        # 更新當前時間為最新的時間戳記
        if new_timestamps:
            self.current_time = max(self.current_time, max(new_timestamps))
        
        logger.info(f"添加 {len(new_edges)} 條新邊，當前共 {self.g.num_edges()} 條邊")
    
    @memory_usage_decorator
    def add_edges_in_batches(self, src_nodes, dst_nodes, timestamps, edge_feats=None, batch_size=None):
        """
        批量添加邊到圖
        
        參數:
            src_nodes (list): 源節點ID列表
            dst_nodes (list): 目標節點ID列表
            timestamps (list): 邊的時間戳記列表
            edge_feats (list, optional): 邊特徵列表
            batch_size (int, optional): 每批添加的邊數量，如果為 None 則使用配置中的批次大小
        """
        if len(src_nodes) != len(dst_nodes) or len(src_nodes) != len(timestamps):
            raise ValueError("源節點、目標節點和時間戳記列表長度必須相同")
        
        if not src_nodes:
            return
        
        if batch_size is None:
            batch_size = self.edge_batch_size
        
        total_edges = len(src_nodes)
        logger.info(f"批量添加 {total_edges} 條邊，每批 {batch_size} 條")
        
        for i in range(0, total_edges, batch_size):
            end_idx = min(i + batch_size, total_edges)
            logger.info(f"處理邊批次 {i//batch_size + 1}/{(total_edges+batch_size-1)//batch_size}: {i} 到 {end_idx-1}")
            
            batch_src = src_nodes[i:end_idx]
            batch_dst = dst_nodes[i:end_idx]
            batch_timestamps = timestamps[i:end_idx]
            
            if edge_feats is not None:
                batch_edge_feats = edge_feats[i:end_idx]
            else:
                batch_edge_feats = None
                
            self.add_edges(batch_src, batch_dst, batch_timestamps, batch_edge_feats)
            
            # 每批次處理後顯示進度
            logger.info(f"批次 {i//batch_size + 1} 完成，當前共 {self.g.num_edges()} 條邊")
            
            # 定期清理記憶體
            if (i//batch_size + 1) % 10 == 0:
                clean_memory()
    
    def get_node_features(self):
        """獲取節點特徵張量"""
        features = []
        for nid in sorted(self.existing_nodes):
            if nid in self.node_features:
                features.append(self.node_features[nid])
            else:
                # 對於沒有特徵的節點，使用零向量
                features.append(torch.zeros(self.node_feat_dim))
        
        return torch.stack(features) if features else torch.zeros((0, self.node_feat_dim))
    
    def get_edge_features(self):
        """獲取邊特徵張量"""
        features = []
        for eid in sorted(self.existing_edges):
            if eid in self.edge_features:
                features.append(torch.tensor(self.edge_features[eid]))
            else:
                # 對於沒有特徵的邊，使用零向量
                features.append(torch.zeros(self.edge_feat_dim))
        
        return torch.stack(features) if features else torch.zeros((0, self.edge_feat_dim))
    
    def get_node_labels(self):
        """獲取節點標籤張量"""
        labels = []
        for nid in sorted(self.existing_nodes):
            if nid in self.node_labels:
                labels.append(self.node_labels[nid])
            else:
                # 對於沒有標籤的節點，使用 -1
                labels.append(-1)
        
        return torch.tensor(labels) if labels else torch.zeros(0, dtype=torch.long)
    
    @memory_usage_decorator
    def update_temporal_graph(self, current_time=None):
        """
        更新時間圖 (僅保留時間窗口內的邊)
        
        參數:
            current_time (float, optional): 當前時間戳記，預設使用最新時間
        """
        if current_time is not None:
            self.current_time = current_time
        
        # 計算時間窗口的起始時間
        start_time = self.current_time - self.temporal_window
        
        # 過濾時間窗口內的邊
        temporal_src = []
        temporal_dst = []
        temporal_edge_feats = []
        temporal_edge_times = []
        
        # 獲取節點列表
        node_list = sorted(list(self.existing_nodes))
        
        # 如果啟用子圖採樣且節點數量超過限制，進行採樣
        if self.use_subgraph_sampling and len(node_list) > self.max_nodes_per_subgraph:
            logger.info(f"節點數量 {len(node_list)} 超過限制 {self.max_nodes_per_subgraph}，進行子圖採樣")
            node_list = random.sample(node_list, self.max_nodes_per_subgraph)
        
        # 創建節點索引映射
        node_idx_map = {nid: i for i, nid in enumerate(node_list)}
        
        # 過濾時間窗口內的邊
        edge_count = 0
        for (src, dst), timestamp in self.edge_timestamps.items():
            if timestamp >= start_time and src in node_idx_map and dst in node_idx_map:
                # 獲取節點索引
                src_idx = node_idx_map[src]
                dst_idx = node_idx_map[dst]
                
                temporal_src.append(src_idx)
                temporal_dst.append(dst_idx)
                
                if (src, dst) in self.edge_features:
                    temporal_edge_feats.append(self.edge_features[(src, dst)])
                else:
                    temporal_edge_feats.append([0.0] * self.edge_feat_dim)
                
                temporal_edge_times.append(timestamp)
                
                edge_count += 1
                
                # 如果啟用子圖採樣且邊數量超過限制，提前結束
                if self.use_subgraph_sampling and edge_count >= self.max_edges_per_subgraph:
                    logger.info(f"邊數量達到限制 {self.max_edges_per_subgraph}，提前結束邊過濾")
                    break
        
        # 建立時間子圖
        if self.use_sparse_representation and len(temporal_src) > 0:
            # 使用稀疏表示
            self.temporal_g = dgl.graph((temporal_src, temporal_dst), 
                                       num_nodes=len(node_list),
                                       idtype=torch.int64,
                                       device='cpu')
        else:
            # 使用密集表示
            self.temporal_g = dgl.graph((temporal_src, temporal_dst), 
                                       num_nodes=len(node_list),
                                       idtype=torch.int64,
                                       device='cpu')
        
        # 獲取節點特徵
        node_features = []
        for nid in node_list:
            if nid in self.node_features:
                node_features.append(self.node_features[nid])
            else:
                node_features.append(torch.zeros(self.node_feat_dim))
        
        # 設置節點特徵
        if node_features:
            self.temporal_g.ndata['h'] = torch.stack(node_features)
        
        # 設置邊特徵和時間戳記
        if temporal_edge_feats:
            self.temporal_g.edata['h'] = torch.tensor(temporal_edge_feats)
            self.temporal_g.edata['time'] = torch.tensor(temporal_edge_times)
        
        # 設置節點標籤
        node_labels = []
        for nid in node_list:
            if nid in self.node_labels:
                node_labels.append(self.node_labels[nid])
            else:
                node_labels.append(-1)
        
        if node_labels:
            self.temporal_g.ndata['label'] = torch.tensor(node_labels)
        
        # 將圖移至指定裝置
        if self.device != 'cpu':
            self.temporal_g = self.temporal_g.to(self.device)
        
        logger.info(f"更新時間圖: {len(temporal_src)} 條邊在時間窗口 {start_time} 到 {self.current_time}")
        
        return self.temporal_g
    
    @memory_usage_decorator
    def add_batch(self, node_ids, features, timestamps, src_nodes=None, dst_nodes=None,
                 edge_timestamps=None, edge_feats=None, labels=None):
        """
        批次添加節點和邊
        
        參數:
            node_ids (list): 節點ID列表
            features (np.ndarray): 節點特徵矩陣
            timestamps (list): 節點時間戳記列表
            src_nodes (list, optional): 源節點ID列表
            dst_nodes (list, optional): 目標節點ID列表
            edge_timestamps (list, optional): 邊時間戳記列表
            edge_feats (list, optional): 邊特徵列表
            labels (list, optional): 節點標籤列表
        """
        # 添加節點
        self.add_nodes(node_ids, features, timestamps, labels)
        
        # 添加邊 (如果提供)
        if src_nodes is not None and dst_nodes is not None and edge_timestamps is not None:
            self.add_edges_in_batches(src_nodes, dst_nodes, edge_timestamps, edge_feats)
        
        # 更新時間圖
        self.update_temporal_graph()
        
        return self.temporal_g
    
    @memory_usage_decorator
    def simulate_stream(self, node_ids, features, timestamps, labels=None):
        """
        模擬流式資料，自動建立時間性邊
        
        參數:
            node_ids (list): 節點ID列表
            features (np.ndarray): 節點特徵矩陣
            timestamps (list): 節點時間戳記列表
            labels (list, optional): 節點標籤列表
            
        此函數會：
        1. 添加新節點
        2. 基於時間接近和連接模式自動建立邊
        3. 更新時間圖
        """
        # 添加節點
        self.add_nodes(node_ids, features, timestamps, labels)
        
        # 自動建立時間性邊
        src_nodes = []
        dst_nodes = []
        edge_timestamps = []
        edge_feats = []
        
        # 時間閾值 (秒)，用於決定兩個封包是否"時間接近"
        time_threshold = 1.0
        
        # 先按時間排序節點
        sorted_nodes = sorted([(nid, self.node_timestamps[nid]) for nid in node_ids], 
                             key=lambda x: x[1])
        
        # 對於每個新節點
        for i, (nid, timestamp) in enumerate(sorted_nodes):
            # 尋找時間接近的先前節點
            for prev_nid in self.existing_nodes:
                if prev_nid == nid:
                    continue
                
                prev_timestamp = self.node_timestamps[prev_nid]
                time_diff = abs(timestamp - prev_timestamp)
                
                # 如果時間接近，建立邊
                if time_diff <= time_threshold:
                    # 確定邊的方向 (較早 -> 較晚)
                    if timestamp > prev_timestamp:
                        src, dst = prev_nid, nid
                    else:
                        src, dst = nid, prev_nid
                    
                    edge_timestamp = max(timestamp, prev_timestamp)
                    
                    # 邊特徵: 時間差、特徵相似度
                    if src in self.node_features and dst in self.node_features:
                        feat1 = self.node_features[src]
                        feat2 = self.node_features[dst]
                        # 計算簡單的特徵相似度 (餘弦相似度的簡化版)
                        sim = torch.dot(feat1, feat2) / (feat1.norm() * feat2.norm() + 1e-8)
                        edge_feat = [time_diff, sim.item()]
                    else:
                        edge_feat = [time_diff, 0.0]
                    
                    src_nodes.append(src)
                    dst_nodes.append(dst)
                    edge_timestamps.append(edge_timestamp)
                    edge_feats.append(edge_feat)
                    
                    # 如果啟用子圖採樣且邊數量超過限制，提前結束
                    if self.use_subgraph_sampling and len(src_nodes) >= self.max_edges_per_subgraph:
                        logger.info(f"邊數量達到限制 {self.max_edges_per_subgraph}，提前結束邊生成")
                        break
            
            # 如果達到邊數量限制，提前結束外層循環
            if self.use_subgraph_sampling and len(src_nodes) >= self.max_edges_per_subgraph:
                break
        
        # 添加生成的邊
        if src_nodes:
            self.add_edges_in_batches(src_nodes, dst_nodes, edge_timestamps, edge_feats)
        
        # 更新時間圖
        self.update_temporal_graph()
        
        return self.temporal_g
