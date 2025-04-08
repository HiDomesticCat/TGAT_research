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

# 導入記憶體優化工具
from memory_utils import (
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
        self.device = device
        
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
        """清理不活躍節點"""
        if not self.prune_inactive_nodes:
            return
        
        current_time = time.time()
        inactive_nodes = []
        
        # 找出不活躍節點
        for nid, last_active in self.node_last_active.items():
            if current_time - last_active > self.inactive_threshold:
                inactive_nodes.append(nid)
        
        if not inactive_nodes:
            return
        
        logger.info(f"清理 {len(inactive_nodes)} 個不活躍節點")
        
        # 從圖中移除不活躍節點
        # 注意：DGL 不支持直接移除節點，需要創建一個新圖
        active_nodes = list(self.existing_nodes - set(inactive_nodes))
        
        # 創建一個新圖，只包含活躍節點
        new_g = dgl.graph(([],  # 源節點
                           []),  # 目標節點
                          num_nodes=len(active_nodes),
                          idtype=torch.int64,
                          device='cpu')
        
        # 更新節點映射
        node_map = {nid: i for i, nid in enumerate(active_nodes)}
        
        # 更新邊
        new_edges = []
        for (src, dst) in self.existing_edges:
            if src in active_nodes and dst in active_nodes:
                new_edges.append((node_map[src], node_map[dst]))
        
        if new_edges:
            src_nodes, dst_nodes = zip(*new_edges)
            new_g.add_edges(src_nodes, dst_nodes)
        
        # 更新節點特徵、時間戳記和標籤
        new_node_features = {}
        new_node_timestamps = {}
        new_node_labels = {}
        new_node_last_active = {}
        
        for nid in active_nodes:
            new_nid = node_map[nid]
            new_node_features[new_nid] = self.node_features[nid]
            new_node_timestamps[new_nid] = self.node_timestamps[nid]
            if nid in self.node_labels:
                new_node_labels[new_nid] = self.node_labels[nid]
            new_node_last_active[new_nid] = self.node_last_active[nid]
        
        # 更新邊特徵和時間戳記
        new_edge_features = {}
        new_edge_timestamps = {}
        new_existing_edges = set()
        
        for i, (src, dst) in enumerate(new_edges):
            old_src = active_nodes[src]
            old_dst = active_nodes[dst]
            old_edge = (old_src, old_dst)
            
            if old_edge in self.edge_features:
                new_edge = (src, dst)
                new_edge_features[new_edge] = self.edge_features[old_edge]
                new_edge_timestamps[new_edge] = self.edge_timestamps[old_edge]
                new_existing_edges.add(new_edge)
        
        # 更新圖和相關數據結構
        self.g = new_g
        self.node_features = new_node_features
        self.node_timestamps = new_node_timestamps
        self.node_labels = new_node_labels
        self.node_last_active = new_node_last_active
        self.edge_features = new_edge_features
        self.edge_timestamps = new_edge_timestamps
        self.existing_edges = new_existing_edges
        self.existing_nodes = set(active_nodes)
        
        # 更新源到目標的映射
        self.src_to_dst = defaultdict(set)
        self.dst_to_src = defaultdict(set)
        
        for (src, dst) in self.existing_edges:
            self.src_to_dst[src].add(dst)
            self.dst_to_src[dst].add(src)
        
        # 更新最後清理時間
        self.last_pruning_time = current_time
        
        # 強制執行垃圾回收
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
                        logger.info(f"邊數量達到限制 {self.max_edges_per_subgraph}，提前結束邊創建")
                        break
            
            # 定期清理記憶體
            if i % 100 == 0:
                clean_memory()
        
        # 添加自動生成的邊
        if src_nodes:
            self.add_edges_in_batches(src_nodes, dst_nodes, edge_timestamps, edge_feats)
        
        # 更新時間圖
        self.update_temporal_graph()
        
        return self.temporal_g
    
    def to_device(self, device=None):
        """
        將圖移至指定裝置
        
        參數:
            device (str, optional): 裝置名稱，如果為 None 則使用初始化時指定的裝置
        """
        if device is None:
            device = self.device
        
        if device == 'cpu':
            return
        
        if self.temporal_g is not None:
            self.temporal_g = self.temporal_g.to(device)
            logger.info(f"將時間圖移至裝置: {device}")
    
    def save_graph(self, save_dir, prefix='graph'):
        """
        保存圖結構
        
        參數:
            save_dir (str): 保存目錄
            prefix (str): 文件名前綴
        """
        # 確保目錄存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成時間戳記
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存主圖
        graph_path = os.path.join(save_dir, f"{prefix}_main_{timestamp}.bin")
        dgl.save_graphs(graph_path, [self.g])
        
        # 保存時間圖
        if self.temporal_g is not None:
            temporal_graph_path = os.path.join(save_dir, f"{prefix}_temporal_{timestamp}.bin")
            dgl.save_graphs(temporal_graph_path, [self.temporal_g])
        
        # 保存節點和邊的元數據
        metadata = {
            'node_features': self.node_features,
            'edge_timestamps': self.edge_timestamps,
            'edge_features': self.edge_features,
            'node_timestamps': self.node_timestamps,
            'node_labels': self.node_labels,
            'current_time': self.current_time,
            'existing_nodes': list(self.existing_nodes),
            'existing_edges': list(self.existing_edges),
            'node_feat_dim': self.node_feat_dim,
            'edge_feat_dim': self.edge_feat_dim,
            'node_last_active': self.node_last_active
        }
        
        metadata_path = os.path.join(save_dir, f"{prefix}_metadata_{timestamp}.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"圖結構已保存至: {save_dir}")
        logger.info(f"  主圖: {graph_path}")
        if self.temporal_g is not None:
            logger.info(f"  時間圖: {temporal_graph_path}")
        logger.info(f"  元數據: {metadata_path}")
    
    def load_graph(self, graph_path, metadata_path):
        """
        載入圖結構
        
        參數:
            graph_path (str): 主圖文件路徑
            metadata_path (str): 元數據文件路徑
        """
        # 載入主圖
        graphs, _ = dgl.load_graphs(graph_path)
        self.g = graphs[0]
        
        # 載入元數據
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # 恢復節點和邊的元數據
        self.node_features = metadata['node_features']
        self.edge_timestamps = metadata['edge_timestamps']
        self.edge_features = metadata['edge_features']
        self.node_timestamps = metadata['node_timestamps']
        self.node_labels = metadata['node_labels']
        self.current_time = metadata['current_time']
        self.existing_nodes = set(metadata['existing_nodes'])
        self.existing_edges = set(metadata['existing_edges'])
        self.node_feat_dim = metadata['node_feat_dim']
        self.edge_feat_dim = metadata['edge_feat_dim']
        self.node_last_active = metadata['node_last_active']
        
        # 恢復源到目標的映射
        self.src_to_dst = defaultdict(set)
        self.dst_to_src = defaultdict(set)
        
        for (src, dst) in self.existing_edges:
            self.src_to_dst[src].add(dst)
            self.dst_to_src[dst].add(src)
        
        # 更新時間圖
        self.update_temporal_graph()
        
        logger.info(f"已載入圖結構: {graph_path}")
        logger.info(f"  節點數量: {self.g.num_nodes()}")
        logger.info(f"  邊數量: {self.g.num_edges()}")
    
    def get_memory_usage(self):
        """
        獲取圖結構的記憶體使用情況
        
        返回:
            dict: 記憶體使用信息
        """
        # 計算節點特徵的記憶體使用
        node_features_size = sum(f.element_size() * f.nelement() for f in self.node_features.values() if isinstance(f, torch.Tensor))
        
        # 計算邊特徵的記憶體使用
        edge_features_size = sum(len(f) * 8 for f in self.edge_features.values())  # 假設每個特徵是 float64 (8 bytes)
        
        # 計算時間戳記的記憶體使用
        timestamps_size = (len(self.node_timestamps) + len(self.edge_timestamps)) * 8  # 假設每個時間戳記是 float64 (8 bytes)
        
        # 計算標籤的記憶體使用
        labels_size = len(self.node_labels) * 4  # 假設每個標籤是 int32 (4 bytes)
        
        # 計算圖結構的記憶體使用
        graph_size = 0
        if self.g is not None:
            graph_size += self.g.num_nodes() * 8  # 節點 ID
            graph_size += self.g.num_edges() * 16  # 邊 (源節點 ID + 目標節點 ID)
        
        if self.temporal_g is not None:
            graph_size += self.temporal_g.num_nodes() * 8  # 節點 ID
            graph_size += self.temporal_g.num_edges() * 16  # 邊 (源節點 ID + 目標節點 ID)
            
            # 節點特徵
            if 'h' in self.temporal_g.ndata:
                graph_size += self.temporal_g.ndata['h'].element_size() * self.temporal_g.ndata['h'].nelement()
            
            # 邊特徵
            if 'h' in self.temporal_g.edata:
                graph_size += self.temporal_g.edata['h'].element_size() * self.temporal_g.edata['h'].nelement()
            
            # 邊時間戳記
            if 'time' in self.temporal_g.edata:
                graph_size += self.temporal_g.edata['time'].element_size() * self.temporal_g.edata['time'].nelement()
            
            # 節點標籤
            if 'label' in self.temporal_g.ndata:
                graph_size += self.temporal_g.ndata['label'].element_size() * self.temporal_g.ndata['label'].nelement()
        
        # 轉換為 MB
        total_size = (node_features_size + edge_features_size + timestamps_size + labels_size + graph_size) / (1024 * 1024)
        
        return {
            'node_features_mb': node_features_size / (1024 * 1024),
            'edge_features_mb': edge_features_size / (1024 * 1024),
            'timestamps_mb': timestamps_size / (1024 * 1024),
            'labels_mb': labels_size / (1024 * 1024),
            'graph_structure_mb': graph_size / (1024 * 1024),
            'total_mb': total_size
        }
    
    def print_memory_usage(self):
        """打印圖結構的記憶體使用情況"""
        mem_info = self.get_memory_usage()
        
        logger.info("圖結構記憶體使用情況:")
        logger.info(f"  節點特徵: {mem_info['node_features_mb']:.2f} MB")
        logger.info(f"  邊特徵: {mem_info['edge_features_mb']:.2f} MB")
        logger.info(f"  時間戳記: {mem_info['timestamps_mb']:.2f} MB")
        logger.info(f"  標籤: {mem_info['labels_mb']:.2f} MB")
        logger.info(f"  圖結構: {mem_info['graph_structure_mb']:.2f} MB")
        logger.info(f"  總計: {mem_info['total_mb']:.2f} MB")

# 測試圖構建器
if __name__ == "__main__":
    import yaml
    
    # 載入配置
    with open("memory_optimized_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # 初始化圖構建器
    graph_builder = MemoryOptimizedDynamicNetworkGraph(config, device='cpu')
    
    # 建立測試資料
    n_nodes = 10
    node_ids = list(range(n_nodes))
    features = np.random.randn(n_nodes, 5)  # 5維特徵
    timestamps = [float(i) for i in range(n_nodes)]
    labels = [i % 2 for i in range(n_nodes)]  # 二元標籤
    
    # 添加節點
    graph_builder.add_nodes(node_ids, features, timestamps, labels)
    
    # 建立一些邊
    src_nodes = [0, 1, 2, 3]
    dst_nodes = [1, 2, 3, 4]
    edge_timestamps = [0.5, 1.5, 2.5, 3.5]
    edge_feats = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
    
    graph_builder.add_edges(src_nodes, dst_nodes, edge_timestamps, edge_feats)
    
    # 更新時間圖
    temporal_g = graph_builder.update_temporal_graph()
    
    print(f"圖統計資訊:")
    print(f"  節點數: {temporal_g.num_nodes()}")
    print(f"  邊數: {temporal_g.num_edges()}")
    print(f"  節點特徵形狀: {temporal_g.ndata['h'].shape}")
    print(f"  邊特徵形狀: {temporal_g.edata['h'].shape}")
    print(f"  邊時間戳記形狀: {temporal_g.edata['time'].shape}")
    
    # 測試批次添加
    new_node_ids = list(range(n_nodes, n_nodes + 5))
    new_features = np.random.randn(5, 5)
    new_timestamps = [float(i + n_nodes) for i in range(5)]
    new_labels = [i % 2 for i in range(5)]
    
    graph_builder.add_batch(new_node_ids, new_features, new_timestamps, labels=new_labels)
    
    print(f"\n批次添加後圖統計資訊:")
    print(f"  節點數: {graph_builder.temporal_g.num_nodes()}")
    print(f"  邊數: {graph_builder.temporal_g.num_edges()}")
    
    # 測試模擬流式資料
    stream_node_ids = list(range(n_nodes + 5, n_nodes + 10))
    stream_features = np.random.randn(5, 5)
    stream_timestamps = [float(i + n_nodes + 5) for i in range(5)]
    stream_labels = [i % 2 for i in range(5)]
    
    graph_builder.simulate_stream(stream_node_ids, stream_features, stream_timestamps, stream_labels)
    
    print(f"\n模擬流式資料後圖統計資訊:")
    print(f"  節點數: {graph_builder.temporal_g.num_nodes()}")
    print(f"  邊數: {graph_builder.temporal_g.num_edges()}")
    
    # 測試記憶體使用情況
    graph_builder.print_memory_usage()
    
    # 測試保存和載入
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    graph_builder.save_graph(temp_dir)
    
    # 創建一個新的圖構建器
    new_graph_builder = MemoryOptimizedDynamicNetworkGraph(config, device='cpu')
    
    # 獲取保存的文件路徑
    import glob
    graph_files = glob.glob(os.path.join(temp_dir, "graph_main_*.bin"))
    metadata_files = glob.glob(os.path.join(temp_dir, "graph_metadata_*.pkl"))
    
    if graph_files and metadata_files:
        # 載入圖
        new_graph_builder.load_graph(graph_files[0], metadata_files[0])
        
        print(f"\n載入後圖統計資訊:")
        print(f"  節點數: {new_graph_builder.temporal_g.num_nodes()}")
        print(f"  邊數: {new_graph_builder.temporal_g.num_edges()}")
