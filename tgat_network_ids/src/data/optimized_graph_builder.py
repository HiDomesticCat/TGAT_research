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

class OptimizedGraphBuilder:
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
        logger.info(f"子圖採樣設置: 啟用={self.use_subgraph_sampling}, 最大節點數={self.max_nodes_per_subgraph}")
        logger.info(f"稀疏表示設置: 啟用={self.use_sparse_representation}, 分塊稀疏={self.use_block_sparse}")
        logger.info(f"記憶體優化: 邊索引格式={self.edge_index_format}, 智能記憶體分配={self.smart_memory_allocation}")

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
            filtered_edge_feats = [edge_feats[i]
