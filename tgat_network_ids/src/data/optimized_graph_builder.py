#!/usr/bin/env python
# coding: utf-8 -*-

"""
優化版動態圖結構建立模組

根據建議進行高度優化，專注於：
1. 高度優化的稀疏圖表示
2. 增強的子圖採樣
3. 更高效的邊批次處理
4. 智能型節點清理策略
5. 深度整合DGL優化API
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
import scipy.sparse as sp

# 導入記憶體優化工具
from ..utils.memory_utils import (
    clean_memory, memory_usage_decorator, print_memory_usage,
    get_memory_usage, print_optimization_suggestions
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedDynamicNetworkGraph:
    """優化版動態網路圖結構類別"""

    def __init__(self, config, device='cuda'):
        """
        初始化優化版動態網路圖結構

        參數:
            config (dict): 配置字典
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
        
        # 新增優化設定
        self.use_block_sparse = graph_config.get('use_block_sparse', True)  # 使用分塊稀疏表示
        self.block_size = graph_config.get('block_size', 128)  # 分塊大小
        self.use_csr_format = graph_config.get('use_csr_format', True)  # 使用CSR格式
        self.use_heterograph = graph_config.get('use_heterograph', False)  # 使用異構圖(適用於有多種節點類型)
        self.use_dgl_transform = graph_config.get('use_dgl_transform', True)  # 使用DGL的transform API
        self.cache_neighbor_sampling = graph_config.get('cache_neighbor_sampling', True)  # 緩存鄰居採樣結果
        self.adaptive_pruning = graph_config.get('adaptive_pruning', True)  # 自適應清理策略
        self.smart_memory_allocation = graph_config.get('smart_memory_allocation', True)  # 智能記憶體分配
        self.edge_index_format = graph_config.get('edge_index_format', True)  # 使用邊索引格式(PyG風格)

        # 設置裝置
        # 檢查 CUDA 是否可用
        if device == 'cuda' and torch.cuda.is_available():
            try:
                # 嘗試初始化 CUDA
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
        self.edge_index = None  # 邊索引(PyG格式)
        self.node_features = {}  # 節點特徵
        self.edge_timestamps = {}  # 邊的時間戳記
        self.edge_features = {}  # 邊特徵
        self.node_timestamps = {}  # 節點時間戳記
        self.node_labels = {}  # 節點標籤
        self.current_time = 0  # 當前時間
        self.temporal_g = None  # 時間子圖
        
        # 分塊稀疏表示
        self.sparse_blocks = {}  # 分塊稀疏表示
        
        # 鄰居採樣緩存
        self.neighbor_cache = {}  # 鄰居採樣結果緩存
        self.cache_hits = 0
        self.cache_misses = 0

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
        
        # 記錄節點重要性分數，用於自適應清理
        self.node_importance = {}

        # 記錄上次清理時間
        self.last_pruning_time = time.time()

        # 初始化一個空圖
        self._init_graph()

        logger.info(f"初始化優化版動態網路圖結構: 時間窗口={self.temporal_window}秒")
        logger.info(f"子圖採樣設置: 啟用={self.use_subgraph_sampling} 最大節點數={self.max_nodes_per_subgraph}")
        logger.info(f"稀疏表示設置: 啟用={self.use_sparse_representation} 分塊稀疏={self.use_block_sparse}") 
        logger.info(f"記憶體優化: 邊索引格式={self.edge_index_format} 智能記憶體分配={self.smart_memory_allocation}")

    def _init_graph(self):
        """初始化一個空圖 - 根據不同設定選擇最佳表示方法"""
        if self.edge_index_format:
            # 使用邊索引格式 (PyG風格)
            self.edge_index = ([], [])  # (src_list, dst_list)
            
            # 創建一個空的DGL圖，使用稀疏表示
            self.g = dgl.graph(([], []),
                              num_nodes=0,
                              idtype=torch.int64,
                              device='cpu')  # 初始在 CPU 上創建
        elif self.use_heterograph:
            # 使用異構圖 - 適用於有多種節點/邊類型
            # 定義關係類型
            graph_data = {
                ('node', 'connects', 'node'): ([], [])
            }
            self.g = dgl.heterograph(graph_data)
        else:
            # 使用標準DGL圖
            self.g = dgl.graph(([], []),  # 源節點、目標節點
                              num_nodes=0,
                              idtype=torch.int64,
                              device='cpu')  # 初始在 CPU 上創建

        # 初始化稀疏塊
        if self.use_block_sparse:
            self.sparse_blocks = {
                'rows': [],
                'cols': [],
                'blocks': {}
            }

        logger.info("初始化空圖 - 使用優化格式")

    @memory_usage_decorator
    def add_nodes(self, node_ids, features, timestamps, labels=None):
        """
        添加節點到圖 - 使用智能記憶體管理

        參數:
            node_ids (list): 節點ID列表
            features (np.ndarray): 節點特徵矩陣 [n_nodes, feat_dim]
            timestamps (list): 節點時間戳記列表
            labels (list, optional): 節點標籤列表
        """
        # 設置節點特徵維度 (首次添加時)
        if self.node_feat_dim is None and features is not None:
            self.node_feat_dim = features.shape[1]

        # 轉換為張量 - 優化記憶體使用
        if isinstance(features, np.ndarray):
            if self.smart_memory_allocation and features.dtype == np.float64:
                # 降低精度以節省記憶體 - 對特徵使用float32足夠
                features = torch.tensor(features, dtype=torch.float32)
            else:
                features = torch.FloatTensor(features)

        # 獲取新節點 (排除已存在的)
        new_nodes = [nid for nid in node_ids if nid not in self.existing_nodes]

        if not new_nodes:
            return

        # 更新圖的節點數
        current_num_nodes = self.g.num_nodes()
        new_num_nodes = current_num_nodes + len(new_nodes)
        self.g.add_nodes(len(new_nodes))

        # 更新節點特徵字典 - 高效率版本
        node_idx_map = {nid: i + current_num_nodes for i, nid in enumerate(new_nodes)}

        # 直接批量更新而非逐個處理
        for i, nid in enumerate(new_nodes):
            idx = node_idx_map[nid]
            self.node_features[nid] = features[i]
            self.node_timestamps[nid] = timestamps[i]
            self.node_last_active[nid] = time.time()  # 記錄當前時間作為最後活躍時間
            
            # 初始化節點重要性分數 - 用於自適應清理
            if self.adaptive_pruning:
                self.node_importance[nid] = 1.0  # 初始重要性分數

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

        # 智能定期清理不活躍節點
        if self.prune_inactive_nodes:
            current_time = time.time()
            # 根據圖大小和最後清理時間自適應調整清理頻率
            pruning_interval = self._get_adaptive_pruning_interval()
            if current_time - self.last_pruning_time > pruning_interval:
                self._prune_inactive_nodes()

    def _get_adaptive_pruning_interval(self):
        """智能調整清理間隔 - 根據圖大小和記憶體使用情況"""
        # 基礎間隔
        base_interval = 300  # 5分鐘
        
        if not self.adaptive_pruning:
            return base_interval
        
        # 根據節點數量調整
        node_factor = min(3.0, max(0.5, len(self.existing_nodes) / 10000))
        
        # 根據記憶體使用調整
        mem_info = get_memory_usage()
        mem_percent = mem_info['system_memory_percent']
        mem_factor = 1.0
        
        # 記憶體使用率高時，更頻繁地清理
        if mem_percent > 80:
            mem_factor = 0.5
        elif mem_percent < 50:
            mem_factor = 2.0
            
        # 計算最終間隔
        interval = base_interval * node_factor * mem_factor
        
        return interval

    def _prune_inactive_nodes(self):
        """高效清理不活躍節點 - 使用重要性評分"""
        if not self.prune_inactive_nodes:
            return

        # 檢查上次清理時間，避免頻繁清理
        current_time = time.time()
        if current_time - self.last_pruning_time < 600:  # 至少10分鐘才清理一次
            return

        # 找出不活躍節點
        inactive_nodes = []
        for nid, last_active in self.node_last_active.items():
            inactive_time = current_time - last_active
            
            # 使用自適應重要性閾值
            if self.adaptive_pruning:
                # 重要性高的節點可以存活更長時間
                importance = self.node_importance.get(nid, 1.0)
                threshold = self.inactive_threshold * importance
                if inactive_time > threshold:
                    inactive_nodes.append(nid)
            else:
                # 標準清理方式
                if inactive_time > self.inactive_threshold:
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

        # 使用DGL的高效API建立新圖
        if self.edge_index_format:
            # 使用邊索引格式
            if active_edges:
                new_src = [node_map[e[0]] for e in active_edges]
                new_dst = [node_map[e[1]] for e in active_edges]
                self.edge_index = (new_src, new_dst)
            else:
                self.edge_index = ([], [])
                
            # 創建新圖
            new_g = dgl.graph((self.edge_index[0], self.edge_index[1]),
                            num_nodes=len(active_nodes),
                            idtype=torch.int64,
                            device='cpu')
        else:
            # 使用標準DGL API
            if active_edges:
                # 將原始節點ID映射到新的連續節點ID
                new_src = [node_map[e[0]] for e in active_edges]
                new_dst = [node_map[e[1]] for e in active_edges]

                # 批量添加邊 - 使用DGL高效API
                if len(new_src) > 0:
                    # 創建新圖並一次性添加所有邊
                    new_g = dgl.graph((new_src, new_dst),
                                    num_nodes=len(active_nodes),
                                    idtype=torch.int64,
                                    device='cpu')
                else:
                    # 如果沒有邊，創建空圖
                    new_g = dgl.graph(([], []),
                                    num_nodes=len(active_nodes),
                                    idtype=torch.int64,
                                    device='cpu')
            else:
                # 如果沒有邊，創建空圖
                new_g = dgl.graph(([], []),
                                num_nodes=len(active_nodes),
                                idtype=torch.int64,
                                device='cpu')

        # 高效更新節點特徵和相關數據 - 使用向量化操作而非逐個處理
        new_node_features = {}
        new_node_timestamps = {}
        new_node_labels = {}
        new_node_last_active = {}
        new_node_importance = {}
        new_edge_features = {}
        new_edge_timestamps = {}
        new_existing_edges = set()

        # 節點數據轉換 - 批量處理
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
            if self.adaptive_pruning and old_id in self.node_importance:
                # 提升保留下來的節點的重要性
                new_node_importance[new_id] = self.node_importance[old_id] * 1.2

        # 邊數據轉換 - 使用映射表加速
        edge_map = {}
        for i, (old_src, old_dst) in enumerate(active_edges):
            new_src = node_map[old_src]
            new_dst = node_map[old_dst]
            new_edge = (new_src, new_dst)
            old_edge = (old_src, old_dst)
            edge_map[old_edge] = new_edge
            
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
        self.existing_nodes = set(range(len(active_nodes)))  # 使用連續索引
        if self.adaptive_pruning:
            self.node_importance = new_node_importance

        # 高效重建源到目標的映射 - 使用defaultdict減少檢查
        self.src_to_dst = defaultdict(set)
        self.dst_to_src = defaultdict(set)

        # 一次遍歷處理所有新邊
        for (src, dst) in self.existing_edges:
            self.src_to_dst[src].add(dst)
            self.dst_to_src[dst].add(src)

        # 清空鄰居緩存
        if self.cache_neighbor_sampling:
            self.neighbor_cache = {}

        # 更新最後清理時間
        self.last_pruning_time = current_time

        # 清理記憶體
        clean_memory()

        logger.info(f"清理完成，當前共 {len(self.existing_nodes)} 個節點，{len(self.existing_edges)} 條邊")

    @memory_usage_decorator
    def add_edges(self, src_nodes, dst_nodes, timestamps, edge_feats=None):
        """
        添加邊到圖 - 使用高效批量操作

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

        # 批量獲取新邊 - 避免逐個處理
        new_edges_info = [(i, src, dst) for i, (src, dst) in enumerate(zip(src_nodes, dst_nodes)) 
                         if (src, dst) not in self.existing_edges and src in self.existing_nodes and dst in self.existing_nodes]
        
        if not new_edges_info:
            return
        
        # 拆分信息
        indices, new_src, new_dst = zip(*new_edges_info)
        new_timestamps = [timestamps[i] for i in indices]
        
        if edge_feats is not None:
            new_edge_feats = [edge_feats[i] for i in indices]
        else:
            new_edge_feats = None

        # 更新節點最後活躍時間 - 批量更新
        current_time = time.time()
        for src, dst in zip(new_src, new_dst):
            self.node_last_active[src] = current_time
            self.node_last_active[dst] = current_time
            
            # 更新節點重要性 - 連接度高的節點更重要
            if self.adaptive_pruning:
                self.node_importance[src] = min(10.0, self.node_importance.get(src, 1.0) + 0.1)
                self.node_importance[dst] = min(10.0, self.node_importance.get(dst, 1.0) + 0.1)

        # 獲取節點的圖索引 - 使用連續索引加速
        if self.edge_index_format:
            # 使用邊索引格式時，直接添加到邊索引
            curr_src, curr_dst = self.edge_index
            curr_src = list(curr_src) + list(new_src)
            curr_dst = list(curr_dst) + list(new_dst)
            self.edge_index = (curr_src, curr_dst)
            
            # 更新DGL圖
            self.g = dgl.graph((curr_src, curr_dst), 
                             num_nodes=self.g.num_nodes(),
                             idtype=torch.int64,
                             device='cpu')
        else:
            # 標準DGL格式
            # 將節點ID轉換為連續索引
            src_indices = []
            dst_indices = []
            
            # 如果使用了連續索引，直接使用節點ID
            if isinstance(next(iter(self.existing_nodes), 0), int) and max(self.existing_nodes, default=0) < self.g.num_nodes():
                src_indices = list(new_src)
                dst_indices = list(new_dst)
            else:
                # 需要查找索引
                node_idx_map = {nid: i for i, nid in enumerate(self.existing_nodes)}
                for src, dst in zip(new_src, new_dst):
                    src_indices.append(node_idx_map[src])
                    dst_indices.append(node_idx_map[dst])

            # 批量添加新邊到圖 - 使用DGL高效API
            self.g.add_edges(src_indices, dst_indices)

        # 更新邊特徵和時間戳記 - 使用字典批量更新
        edge_updates = {}
        timestamp_updates = {}
        
        for i, edge_key in enumerate(zip(new_src, new_dst)):
            edge_updates[edge_key] = new_edge_feats[i] if new_edge_feats else None
            timestamp_updates[edge_key] = new_timestamps[i]
            self.existing_edges.add(edge_key)

            # 更新來源到目標的映射
            src, dst = edge_key
            self.src_to_dst[src].add(dst)
            self.dst_to_src[dst].add(src)
            
        # 批量更新特徵和時間戳
        if new_edge_feats:
            self.edge_features.update(edge_updates)
        self.edge_timestamps.update(timestamp_updates)

        # 更新當前時間為最新的時間戳記
        if new_timestamps:
            self.current_time = max(self.current_time, max(new_timestamps))

        # 清空鄰居緩存 - 因為新添加的邊可能會改變採樣結果
        if self.cache_neighbor_sampling:
            self.neighbor_cache = {}

        logger.info(f"添加 {len(new_src)} 條新邊，當前共 {self.g.num_edges()} 條邊")

    @memory_usage_decorator
    def add_edges_in_batches(self, src_nodes, dst_nodes, timestamps, edge_feats=None, batch_size=None):
        """
        批量添加邊到圖 - 智能批次處理

        參數:
            src_nodes (list): 源節點ID列表
            dst_nodes (list): 目標節點ID列表
            timestamps (list): 邊的時間戳記列表
            edge_feats (list, optional): 邊特徵列表
            batch_size (int, optional): 每批添加的邊數量
        """
        if len(src_nodes) != len(dst_nodes) or len(src_nodes) != len(timestamps):
            raise ValueError("源節點、目標節點和時間戳記列表長度必須相同")

        if not src_nodes:
            return

        # 智能批次大小選擇 - 根據記憶體使用情況調整
        if batch_size is None:
            mem_info = get_memory_usage()
            mem_percent = mem_info.get('system_memory_percent', 0)
            
            # 根據記憶體使用率動態調整批次大小
            if mem_percent > 85:
                # 記憶體使用率高，使用小批次
                batch_size = min(1000, self.edge_batch_size // 5)
            elif mem_percent > 70:
                # 記憶體使用率中等，適當減小批次
                batch_size = min(2000, self.edge_batch_size // 2)
            else:
                # 記憶體充足，使用標準批次大小
                batch_size = self.edge_batch_size
                
            logger.info(f"根據記憶體使用率({mem_percent:.1f}%)動態調整批次大小為: {batch_size}")
        
        # 預先過濾無效邊，避免在批次循環中重複檢查
        valid_edges = []
        for i, (src, dst) in enumerate(zip(src_nodes, dst_nodes)):
            if src in self.existing_nodes and dst in self.existing_nodes:
                valid_edges.append((i, src, dst))
        
        if not valid_edges:
            logger.info("沒有有效的邊需要添加")
            return
            
        # 從有效邊中提取信息
        valid_indices, filtered_src, filtered_dst = zip(*valid_edges)
        filtered_timestamps = [timestamps[i] for i in valid_indices]
        
        if edge_feats is not None:
            filtered_edge_feats = [edge_feats[i] for i in valid_indices]
        else:
            filtered_edge_feats = None
            
        # 計算批次數量
        total_edges = len(filtered_src)
        num_batches = (total_edges + batch_size - 1) // batch_size
        
        logger.info(f"批量添加 {total_edges} 條有效邊，分為 {num_batches} 批次處理")
        
        # 分批次處理邊
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_edges)
            
            # 獲取當前批次
            batch_src = filtered_src[start_idx:end_idx]
            batch_dst = filtered_dst[start_idx:end_idx]
            batch_timestamps = filtered_timestamps[start_idx:end_idx]
            
            if filtered_edge_feats is not None:
                batch_edge_feats = filtered_edge_feats[start_idx:end_idx]
            else:
                batch_edge_feats = None
                
            # 添加當前批次的邊
            self.add_edges(batch_src, batch_dst, batch_timestamps, batch_edge_feats)
            
            # 每隔幾個批次清理一次記憶體
            if (batch_idx + 1) % 5 == 0:
                clean_memory()
                logger.info(f"已處理 {min(end_idx, total_edges)}/{total_edges} 條邊")
                
        logger.info(f"批量添加邊完成")

    @memory_usage_decorator
    def update_temporal_graph(self, current_time=None):
        """
        更新時間圖 (優化版) - 使用高效稀疏表示和採樣技術

        參數:
            current_time (float, optional): 當前時間戳記，預設使用最新時間
        """
        if current_time is not None:
            self.current_time = current_time

        # 計算時間窗口的起始時間
        start_time = self.current_time - self.temporal_window

        # 使用高效的子圖構建技術
        if self.cache_neighbor_sampling and f"{start_time}_{self.current_time}" in self.neighbor_cache:
            # 使用緩存的結果
            self.cache_hits += 1
            logger.info(f"使用緩存的時間子圖 ({start_time} 到 {self.current_time}), 緩存命中率: {self.cache_hits/(self.cache_hits+self.cache_misses):.2f}")
            self.temporal_g = self.neighbor_cache[f"{start_time}_{self.current_time}"]
            return self.temporal_g
        else:
            self.cache_misses += 1

        # 過濾時間窗口內的邊 - 使用向量化操作加速
        temporal_src = []
        temporal_dst = []
        temporal_edge_feats = []
        temporal_edge_times = []

        # 獲取節點列表
        node_list = sorted(list(self.existing_nodes))
        node_set = set(node_list)  # 快速查找用的集合

        # 高效採樣 - 如果啟用子圖採樣且節點數量超過限制
        if self.use_subgraph_sampling and len(node_list) > self.max_nodes_per_subgraph:
            # 使用度數加權採樣而非簡單隨機採樣 - 保留高連接度節點
            if self.adaptive_pruning:
                # 計算節點權重 - 根據連接度和重要性
                node_weights = {}
                for nid in node_list:
                    in_degree = len(self.dst_to_src.get(nid, set()))
                    out_degree = len(self.src_to_dst.get(nid, set()))
                    total_degree = in_degree + out_degree
                    importance = self.node_importance.get(nid, 1.0)
                    node_weights[nid] = total_degree * importance
                
                # 根據權重進行採樣
                if sum(node_weights.values()) > 0:
                    sampled_nodes = random.choices(
                        node_list,
                        weights=[node_weights.get(nid, 1.0) for nid in node_list],
                        k=min(self.max_nodes_per_subgraph, len(node_list))
                    )
                    # 去除可能的重複
                    node_list = list(set(sampled_nodes))
                else:
                    node_list = random.sample(node_list, self.max_nodes_per_subgraph)
            else:
                # 簡單隨機採樣
                node_list = random.sample(node_list, self.max_nodes_per_subgraph)
            
            node_set = set(node_list)  # 更新節點集合
            logger.info(f"節點數量 {len(self.existing_nodes)} 超過限制 {self.max_nodes_per_subgraph}，採樣後剩餘 {len(node_list)} 個節點")

        # 高效映射建立 - 為了快速索引查找
        node_idx_map = {nid: i for i, nid in enumerate(node_list)}

        # 高效邊過濾 - 使用字典結構和集合操作加速
        if self.edge_index_format:
            # 使用邊索引格式時的高效過濾
            src_array, dst_array = self.edge_index
            time_filtered_edges = []
            
            # 批量檢查時間戳
            for edge_idx, (src, dst) in enumerate(zip(src_array, dst_array)):
                edge_key = (src, dst)
                if edge_key in self.edge_timestamps:
                    timestamp = self.edge_timestamps[edge_key]
                    if timestamp >= start_time and src in node_set and dst in node_set:
                        time_filtered_edges.append((edge_idx, src, dst, timestamp))
                        
                        # 限制邊數量
                        if self.use_subgraph_sampling and len(time_filtered_edges) >= self.max_edges_per_subgraph:
                            break
            
            # 從過濾後的邊中提取信息
            if time_filtered_edges:
                edge_indices, filtered_src, filtered_dst, filtered_times = zip(*time_filtered_edges)
                
                # 獲取新的連續索引
                temporal_src = [node_idx_map[src] for src in filtered_src]
                temporal_dst = [node_idx_map[dst] for dst in filtered_dst]
                temporal_edge_times = list(filtered_times)
                
                # 獲取邊特徵
                if self.edge_feat_dim is not None:
                    temporal_edge_feats = []
                    for s, d in zip(filtered_src, filtered_dst):
                        edge_key = (s, d)
                        if edge_key in self.edge_features:
                            temporal_edge_feats.append(self.edge_features[edge_key])
                        else:
                            # 對於沒有特徵的邊，使用零向量
                            temporal_edge_feats.append([0.0] * self.edge_feat_dim)
        else:
            # 標準格式的高效過濾
            edge_count = 0
            for (src, dst), timestamp in self.edge_timestamps.items():
                if timestamp >= start_time and src in node_set and dst in node_set:
                    # 獲取節點索引
                    src_idx = node_idx_map[src]
                    dst_idx = node_idx_map[dst]

                    temporal_src.append(src_idx)
                    temporal_dst.append(dst_idx)
                    temporal_edge_times.append(timestamp)

                    # 邊特徵
                    if (src, dst) in self.edge_features:
                        temporal_edge_feats.append(self.edge_features[(src, dst)])
                    else:
                        # 對於沒有特徵的邊，使用零向量
                        temporal_edge_feats.append([0.0] * self.edge_feat_dim if self.edge_feat_dim else [])

                    edge_count += 1

                    # 如果啟用子圖採樣且邊數量超過限制，提前結束
                    if self.use_subgraph_sampling and edge_count >= self.max_edges_per_subgraph:
                        logger.info(f"邊數量達到限制 {self.max_edges_per_subgraph}，提前結束邊過濾")
                        break

        # 建立時間子圖 - 使用最適合的表示方法
        if len(temporal_src) > 0:
            if self.use_sparse_representation:
                # 使用DGL的稀疏表示API
                if self.use_csr_format:
                    # 創建CSR格式的稀疏圖
                    indices = torch.tensor([temporal_src, temporal_dst], dtype=torch.int64)
                    # 創建稀疏鄰接矩陣
                    adj = torch.sparse_coo_tensor(
                        indices=indices,
                        values=torch.ones(len(temporal_src)),
                        size=(len(node_list), len(node_list))
                    )
                    # 轉換為CSR格式
                    csr = adj.to_sparse_csr()
                    
                    # 使用DGL的from_csr構造函數
                    if hasattr(dgl, 'from_csr'):
                        indptr = csr.crow_indices()
                        indices = csr.col_indices()
                        data = csr.values()
                        self.temporal_g = dgl.from_csr(indptr, indices, data, len(node_list))
                    else:
                        # 備用方法：仍使用標準構造
                        self.temporal_g = dgl.graph((temporal_src, temporal_dst),
                                                num_nodes=len(node_list),
                                                idtype=torch.int64,
                                                device='cpu')
                else:
                    # 使用稀疏COO表示
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
        else:
            # 如果沒有邊，創建空圖
            self.temporal_g = dgl.graph(([], []),
                                     num_nodes=len(node_list),
                                     idtype=torch.int64,
                                     device='cpu')

        # 獲取節點特徵 - 向量化操作
        node_features = []
        for nid in node_list:
            if nid in self.node_features:
                node_features.append(self.node_features[nid])
            else:
                # 對於沒有特徵的節點，使用零向量
                node_features.append(torch.zeros(self.node_feat_dim))

        # 設置節點特徵
        if node_features:
            self.temporal_g.ndata['h'] = torch.stack(node_features)

        # 設置邊特徵和時間戳記
        if temporal_edge_feats:
            self.temporal_g.edata['h'] = torch.tensor(temporal_edge_feats)
        if temporal_edge_times:
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

        # 使用DGL的transform API優化圖
        if self.use_dgl_transform:
            # 添加自環 - 確保圖有完整連接性
            self.temporal_g = dgl.add_self_loop(self.temporal_g)
            
            # 如果支持，轉換為最適合的圖格式
            try:
                if self.use_csr_format and hasattr(dgl.transforms, 'to_simple_graph'):
                    # 將圖轉換為簡單圖 - 合併多重邊
                    self.temporal_g = dgl.transforms.to_simple_graph(self.temporal_g)
            except Exception as e:
                logger.warning(f"圖轉換失敗: {str(e)}")

        # 將圖移至指定裝置
        if self.device != 'cpu':
            self.temporal_g = self.temporal_g.to(self.device)

        # 使用緩存優化
        if self.cache_neighbor_sampling:
            # 緩存當前時間窗口的子圖
            self.neighbor_cache[f"{start_time}_{self.current_time}"] = self.temporal_g
            
            # 限制緩存大小，清理過舊的緩存
            if len(self.neighbor_cache) > 10:
                # 移除最舊的緩存
                oldest_key = sorted(self.neighbor_cache.keys())[0]
                del self.neighbor_cache[oldest_key]

        logger.info(f"更新時間圖: {len(temporal_src)} 條邊在時間窗口 {start_time} 到 {self.current_time}")

        return self.temporal_g

    @memory_usage_decorator
    def add_batch(self, node_ids, features, timestamps, src_nodes=None, dst_nodes=None,
                 edge_timestamps=None, edge_feats=None, labels=None):
        """
        批次添加節點和邊 - 高效集成操作

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
        # 添加節點 - 使用向量化操作
        self.add_nodes(node_ids, features, timestamps, labels)

        # 添加邊 (如果提供) - 使用批次處理提高效率
        if src_nodes is not None and dst_nodes is not None and edge_timestamps is not None:
            self.add_edges_in_batches(src_nodes, dst_nodes, edge_timestamps, edge_feats)

        # 更新時間圖 - 使用優化的子圖構建
        self.update_temporal_graph()

        return self.temporal_g

    @memory_usage_decorator
    def simulate_stream(self, node_ids, features, timestamps, labels=None):
        """
        模擬流式資料，自動建立時間性邊 - 優化版本

        參數:
            node_ids (list): 節點ID列表
            features (np.ndarray): 節點特徵矩陣
            timestamps (list): 節點時間戳記列表
            labels (list, optional): 節點標籤列表
        """
        # 添加節點
        self.add_nodes(node_ids, features, timestamps, labels)

        # 自動建立時間性邊 - 使用高效數據結構
        src_nodes = []
        dst_nodes = []
        edge_timestamps = []
        edge_feats = []

        # 時間閾值 (秒)，用於決定兩個封包是否"時間接近"
        time_threshold = 1.0

        # 先按時間排序節點 - 使用NumPy/Pandas高效排序
        sorted_indices = np.argsort(timestamps)
        sorted_nodes = [(node_ids[i], timestamps[i]) for i in sorted_indices]

        # 使用滑動窗口優化邊生成 - 避免N^2複雜度
        window_size = min(100, len(sorted_nodes))  # 動態調整窗口大小
        
        # 對於每個新節點，連接到時間窗口內的相關節點
        for i, (nid, timestamp) in enumerate(sorted_nodes):
            # 初始化窗口範圍
            window_start = max(0, i - window_size)
            window_end = i  # 不包括當前節點
            
            # 遍歷窗口中的節點
            for j in range(window_start, window_end):
                prev_nid, prev_timestamp = sorted_nodes[j]
                
                # 檢查是否時間接近
                if abs(timestamp - prev_timestamp) <= time_threshold:
                    # 隨機決定連接方向 - 雙向連接增加50%
                    if random.random() < 0.7:  # 70%機率建立邊
                        # 創建 prev -> current 邊
                        src_nodes.append(prev_nid)
                        dst_nodes.append(nid)
                        edge_timestamps.append(timestamp)
                        
                        # 生成簡單的邊特徵 (時間差和隨機特性)
                        time_diff = timestamp - prev_timestamp
                        random_feat = [random.random(), time_diff, random.random()]
                        edge_feats.append(random_feat)
                        
                        # 隨機決定是否建立反向邊
                        if random.random() < 0.5:  # 50%機率建立反向邊
                            src_nodes.append(nid)
                            dst_nodes.append(prev_nid)
                            edge_timestamps.append(timestamp)
                            
                            # 反向邊使用相似但不同的特徵
                            rev_random_feat = [random.random(), -time_diff, random.random()]
                            edge_feats.append(rev_random_feat)
            
            # 優化：動態調整窗口大小以控制邊密度
            # 如果節點多，我們可以使用小窗口；如果節點少，使用大窗口
            if i % 50 == 0:
                edge_density = len(src_nodes) / max(1, len(self.existing_nodes))
                if edge_density > 10:  # 邊密度過高
                    window_size = max(10, window_size // 2)
                elif edge_density < 2:  # 邊密度過低
                    window_size = min(200, window_size * 2)

        # 批量添加所有邊
        if src_nodes:
            self.add_edges_in_batches(src_nodes, dst_nodes, edge_timestamps, edge_feats)

        # 更新時間圖
        self.update_temporal_graph()

        logger.info(f"模擬流式數據: 添加 {len(src_nodes)} 條時間性邊")
        
        return self.temporal_g
    
    def to_sparse_tensor(self):
        """
        將圖轉換為稀疏張量表示 - 用於記憶體高效的表示和計算
        
        返回:
            tuple: (indices, values, shape) - PyTorch稀疏張量的組件
        """
        if not self.g:
            logger.warning("圖為空，無法轉換為稀疏張量")
            return None
            
        # 獲取圖的邊
        if self.edge_index_format:
            src, dst = self.edge_index
        else:
            src, dst = self.g.edges()
            
        # 創建稀疏COO格式表示
        indices = torch.stack([torch.tensor(src), torch.tensor(dst)])
        values = torch.ones(len(src))
        shape = (self.g.num_nodes(), self.g.num_nodes())
        
        # 使用SciPy稀疏矩陣
        if sp is not None and self.use_sparse_representation:
            try:
                # 轉換為SciPy CSR矩陣
                scipy_sparse = sp.csr_matrix(
                    (values.numpy(), (indices[0].numpy(), indices[1].numpy())),
                    shape=shape
                )
                logger.info(f"成功轉換為SciPy CSR稀疏矩陣，大小: {shape}, 非零元素: {scipy_sparse.nnz}")
                return scipy_sparse
            except Exception as e:
                logger.warning(f"轉換為SciPy稀疏矩陣失敗: {str(e)}")
        
        # 使用PyTorch稀疏張量
        sparse_tensor = torch.sparse_coo_tensor(indices, values, shape)
        logger.info(f"成功轉換為PyTorch稀疏張量，大小: {shape}, 非零元素: {len(values)}")
        
        return sparse_tensor
