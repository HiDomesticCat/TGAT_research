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
import scipy.sparse as sp # 僅在 use_sparse_representation 為 True 時實際使用
from typing import Dict, List, Tuple, Set, Union, Optional, Any

# 導入記憶體優化工具 (假設路徑正確)
# 注意：導入路徑可能需要根據您的實際專案結構調整
try:
    # 嘗試使用相對導入
    from ..utils.memory_utils import (
        clean_memory, memory_usage_decorator, print_memory_usage,
        get_memory_usage, print_optimization_suggestions
    )
except ImportError:
     # 如果相對導入失敗，嘗試直接導入（可能在直接運行此文件時需要）
     print("警告：圖建立器無法從相對路徑導入記憶體工具，嘗試直接導入。")
     try:
         # 假設 utils 目錄與 src 在同一層級或已添加到 PYTHONPATH
         from utils.memory_utils import (
             clean_memory, memory_usage_decorator, print_memory_usage,
             get_memory_usage, print_optimization_suggestions
         )
     except ImportError as ie:
          # 如果都失敗，定義虛設函數以允許程式碼運行
          print(f"直接導入 memory_utils 失敗: {ie}。將使用虛設函數。")
          def clean_memory(*args, **kwargs): pass
          # 裝飾器返回原始函數
          def memory_usage_decorator(func): return func
          def print_memory_usage(*args, **kwargs): pass
          def get_memory_usage(*args, **kwargs): return {}
          def print_optimization_suggestions(*args, **kwargs): pass


# 配置日誌記錄器
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedGraphBuilder:
    """優化版動態網路圖結構類別 (已修正)"""

    def __init__(self, config: Dict[str, Any], device: str = 'cpu'):
        """
        初始化優化版動態網路圖結構

        參數:
            config (dict): 配置字典，應包含 'graph' 子字典
            device (str): 計算裝置 ('cpu' 或 'cuda:x')。注意：DGL圖主要在CPU管理。
        """
        # --- 配置讀取 ---
        graph_config = config.get('graph', {})
        self.temporal_window = float(graph_config.get('temporal_window', 300.0))  # 時間窗口大小 (秒)
        self.use_sparse_representation = bool(graph_config.get('use_sparse_representation', False))
        self.inactive_threshold = float(graph_config.get('inactive_threshold', 600.0)) # 節點不活躍閾值 (秒)
        self.pruning_interval = float(graph_config.get('pruning_interval', 300.0)) # 節點清理間隔 (秒)
        # self.edge_index_format = bool(graph_config.get('edge_index_format', False)) # 這個選項似乎未使用，可以考慮移除
        self.use_subgraph_sampling = bool(graph_config.get('use_subgraph_sampling', False))
        self.max_nodes_per_subgraph = int(graph_config.get('max_nodes_per_subgraph', 10000))
        self.max_edges_per_subgraph = int(graph_config.get('max_edges_per_subgraph', 50000))
        self.use_dgl_transform = bool(graph_config.get('use_dgl_transform', False))
        self.cache_neighbor_sampling = bool(graph_config.get('cache_neighbor_sampling', False))

        # 設置設備（主要用於返回的子圖或 Tensor）
        try:
            self.device = torch.device(device)
            logger.info(f"圖建立器數據輸出將使用裝置: {self.device}")
        except Exception as e:
             logger.error(f"無法識別的裝置 '{device}'，將使用 CPU。錯誤: {e}")
             self.device = torch.device('cpu')

        # --- 內部狀態初始化 ---
        # DGL 圖物件 (始終在 CPU 上管理主要結構)
        self.g: Optional[dgl.DGLGraph] = None
        self.temporal_g: Optional[dgl.DGLGraph] = None # 時間窗口內的子圖

        # 節點數據 (使用 *原始* ID 作為鍵)
        self.node_features: Dict[Any, torch.Tensor] = {}
        self.node_timestamps: Dict[Any, float] = {}
        self.node_labels: Dict[Any, int] = {}
        self.node_last_active: Dict[Any, float] = {}
        self.node_importance: Dict[Any, float] = {} # (可選)

        # 邊數據 (使用 *原始* ID 對 (src, dst) 作為鍵)
        self.edge_features: Dict[Tuple[Any, Any], Any] = {} # 可以是 Tensor 或其他類型
        self.edge_timestamps: Dict[Tuple[Any, Any], float] = {}

        # 索引與映射
        self.existing_nodes: Set[Any] = set() # 儲存 *原始* 節點 ID
        self.existing_edges: Set[Tuple[Any, Any]] = set() # 儲存 *原始* ID 對 (src, dst)
        self.node_id_map: Dict[Any, int] = {} # 原始 ID -> DGL 內部連續 ID (0 to N-1)
        self.reverse_node_id_map: Dict[int, Any] = {} # DGL 內部連續 ID -> 原始 ID
        self.next_internal_id: int = 0 # 下一個可用的內部 ID

        # 輔助映射 (使用 *原始* ID 作為鍵)
        self.src_to_dst: Dict[Any, Set[Any]] = defaultdict(set)
        self.dst_to_src: Dict[Any, Set[Any]] = defaultdict(set)

        # 狀態變數
        self.node_feat_dim: Optional[int] = None
        self.edge_feat_dim: Optional[int] = None
        self.current_time: float = 0.0 # 圖的當前邏輯時間 (由輸入數據的時間戳更新)
        self.last_pruning_time: float = time.time() # 上次清理的系統時間戳

        # 鄰居採樣緩存
        self.neighbor_cache: Dict[Tuple[Any, int], List[Any]] = {}

        # 初始化空圖
        self._init_graph()

        logger.info("優化版圖建立器初始化完成")

    def _init_graph(self):
        """初始化或重置為空圖"""
        # 圖結構始終在 CPU 上創建和管理
        self.g = dgl.graph(([], []), num_nodes=0, idtype=torch.int64, device='cpu')
        self.temporal_g = None # 重置時間子圖
        # 清空所有數據字典和映射
        self.node_features.clear()
        self.node_timestamps.clear()
        self.node_labels.clear()
        self.node_last_active.clear()
        self.node_importance.clear()
        self.edge_features.clear()
        self.edge_timestamps.clear()
        self.existing_nodes.clear()
        self.existing_edges.clear()
        self.node_id_map.clear()
        self.reverse_node_id_map.clear()
        self.next_internal_id = 0
        self.src_to_dst.clear()
        self.dst_to_src.clear()
        self.neighbor_cache.clear()
        gc.collect() # 執行垃圾回收
        logger.info("已初始化空圖和所有相關數據結構")

    @memory_usage_decorator
    def add_nodes(self, node_ids: List[Any], features: Optional[Union[np.ndarray, torch.Tensor]] = None,
                  timestamps: Optional[List[float]] = None, labels: Optional[List[int]] = None):
        """
        添加節點到圖 - 使用節點 ID 映射 (修正版)

        參數:
            node_ids: 節點 ID 列表 (原始 ID, 可以是任何可哈希類型)
            features: 節點特徵 (Numpy array 或 Tensor, [num_nodes, feat_dim])
            timestamps: 節點時間戳列表 (對應 node_ids)
            labels: 節點標籤列表 (對應 node_ids, 可選)
        """
        if not node_ids:
            logger.debug("add_nodes: 收到空的 node_ids 列表，不執行任何操作。")
            return

        num_input_nodes = len(node_ids)

        # --- 處理特徵 ---
        if features is not None:
            # 設置或驗證特徵維度
            if self.node_feat_dim is None and len(features) > 0:
                 if hasattr(features, 'shape') and len(features.shape) > 1:
                     self.node_feat_dim = features.shape[1]
                     logger.info(f"檢測到節點特徵維度: {self.node_feat_dim}")
                 else:
                      self.node_feat_dim = 1 # 標量特徵
                      logger.info("檢測到標量節點特徵，維度設為 1。")

            # 確保特徵是 Tensor
            if isinstance(features, np.ndarray):
                features = torch.tensor(features, dtype=torch.float32)
            elif not isinstance(features, torch.Tensor):
                 logger.error(f"預期的特徵類型是 numpy.ndarray 或 torch.Tensor，但收到 {type(features)}")
                 features = None # 無法處理，設為 None
             # 維度檢查
            if features is not None:
                 if features.shape[0] != num_input_nodes:
                      logger.error(f"節點 ID 數量 ({num_input_nodes}) 與特徵數量 ({features.shape[0]}) 不匹配！將忽略特徵。")
                      features = None
                 elif self.node_feat_dim is not None and features.shape[1] != self.node_feat_dim:
                      logger.error(f"輸入特徵維度 ({features.shape[1]}) 與現有維度 ({self.node_feat_dim}) 不匹配！將忽略特徵。")
                      features = None
        else:
             logger.debug("add_nodes: 未提供節點特徵。")

        # --- 處理時間戳 ---
        if timestamps is None:
             current_sys_time_float = time.time()
             timestamps = [current_sys_time_float] * num_input_nodes
             logger.debug(f"add_nodes: 未提供時間戳，使用當前系統時間 {current_sys_time_float}")
        elif len(timestamps) != num_input_nodes:
             logger.error(f"節點 ID 數量 ({num_input_nodes}) 與時間戳數量 ({len(timestamps)}) 不匹配！將忽略時間戳。")
             timestamps = [time.time()] * num_input_nodes # 回退到系統時間

        # --- 處理標籤 ---
        if labels is not None and len(labels) != num_input_nodes:
             logger.error(f"節點 ID 數量 ({num_input_nodes}) 與標籤數量 ({len(labels)}) 不匹配！將忽略標籤。")
             labels = None

        # --- 找出需要添加到 DGL 圖的新節點並更新數據 ---
        new_dgl_nodes_count = 0
        current_sys_time = time.time() # 用於記錄活躍時間
        max_timestamp_in_batch = 0.0

        for i, nid in enumerate(node_ids):
            # 檢查節點是否已在映射中
            if nid not in self.node_id_map:
                internal_id = self.next_internal_id
                self.node_id_map[nid] = internal_id
                self.reverse_node_id_map[internal_id] = nid
                self.next_internal_id += 1
                new_dgl_nodes_count += 1
                self.existing_nodes.add(nid) # 添加到原始 ID 集合

            # 更新節點數據 (使用原始 ID 作為鍵)
            current_timestamp = float(timestamps[i]) # 確保是浮點數
            max_timestamp_in_batch = max(max_timestamp_in_batch, current_timestamp)
            if features is not None: self.node_features[nid] = features[i]
            self.node_timestamps[nid] = current_timestamp
            if labels is not None: self.node_labels[nid] = int(labels[i]) # 確保是整數
            self.node_last_active[nid] = current_sys_time # 更新活躍時間

        # --- 添加新節點到 DGL 圖結構 ---
        if new_dgl_nodes_count > 0:
            self.g = dgl.add_nodes(self.g, new_dgl_nodes_count) # 在 CPU 上添加
            logger.info(f"添加 {new_dgl_nodes_count} 個新節點到 DGL 圖，總內部節點數: {self.g.num_nodes()}")
            # 驗證計數器是否匹配
            if self.g.num_nodes() != self.next_internal_id:
                 logger.critical(f"內部 ID 計數器 ({self.next_internal_id}) 與 DGL 圖節點數 ({self.g.num_nodes()}) 不匹配！映射可能已損壞。")

        # 更新圖的當前邏輯時間
        self.current_time = max(self.current_time, max_timestamp_in_batch)

        # --- 執行節點清理（如果達到間隔）---
        if current_sys_time - self.last_pruning_time > self.pruning_interval:
            self._prune_inactive_nodes()

    @memory_usage_decorator
    def _prune_inactive_nodes(self):
        """高效清理不活躍節點 - 更新映射 (修正版)"""
        if not self.existing_nodes: return # 沒有節點

        current_sys_time = time.time()
        inactive_original_nodes = {
            nid for nid, last_active in self.node_last_active.items()
            if current_sys_time - last_active > self.inactive_threshold
        }

        num_inactive = len(inactive_original_nodes)
        if num_inactive == 0:
            self.last_pruning_time = current_sys_time # 即使沒有清理，也更新時間戳
            logger.debug("無需清理節點。")
            return

        logger.info(f"檢測到 {num_inactive} 個不活躍節點，開始清理...")
        start_prune_time = time.time()

        # --- 獲取活躍節點的 *原始* ID ---
        active_original_nodes = sorted(list(self.existing_nodes - inactive_original_nodes)) # 排序以保證映射穩定性
        num_active_nodes = len(active_original_nodes)
        logger.info(f"清理後預計剩餘 {num_active_nodes} 個活躍節點。")

        if num_active_nodes == 0:
             logger.info("所有節點均不活躍，重置圖。")
             self._init_graph()
             self.last_pruning_time = current_sys_time
             return

        # --- 建立新的映射 (只包含活躍節點) ---
        new_node_id_map = {old_id: new_internal_id for new_internal_id, old_id in enumerate(active_original_nodes)}
        new_reverse_node_id_map = {v: k for k, v in new_node_id_map.items()}
        new_next_internal_id = num_active_nodes

        # --- 遍歷現有邊，篩選活躍邊並獲取 *新* 內部索引 ---
        new_src_indices = []
        new_dst_indices = []
        active_edge_original_keys = set() # 保存活躍邊的原始鍵

        # 直接迭代集合可能更高效
        for src_orig, dst_orig in self.existing_edges:
            if src_orig in new_node_id_map and dst_orig in new_node_id_map:
                new_src_idx = new_node_id_map[src_orig]
                new_dst_idx = new_node_id_map[dst_orig]
                new_src_indices.append(new_src_idx)
                new_dst_indices.append(new_dst_idx)
                active_edge_original_keys.add((src_orig, dst_orig))

        num_active_edges = len(active_edge_original_keys)
        logger.info(f"清理後預計剩餘 {num_active_edges} 條邊。")

        # --- 創建新的 DGL 圖 (使用新的內部索引) ---
        logger.debug("創建新的 DGL 圖結構...")
        new_g = dgl.graph((new_src_indices, new_dst_indices),
                          num_nodes=new_next_internal_id,
                          idtype=torch.int64,
                          device='cpu') # 始終在 CPU 創建

        # --- 更新節點相關數據字典 (只保留活躍節點) ---
        logger.debug("更新節點數據字典...")
        self.node_features = {nid: feat for nid, feat in self.node_features.items() if nid in new_node_id_map}
        self.node_timestamps = {nid: ts for nid, ts in self.node_timestamps.items() if nid in new_node_id_map}
        self.node_labels = {nid: label for nid, label in self.node_labels.items() if nid in new_node_id_map}
        self.node_last_active = {nid: ts for nid, ts in self.node_last_active.items() if nid in new_node_id_map}
        if hasattr(self, 'node_importance'):
             self.node_importance = {nid: score for nid, score in self.node_importance.items() if nid in new_node_id_map}

        # --- 更新邊相關數據字典 (只保留活躍邊) ---
        logger.debug("更新邊數據字典...")
        self.edge_timestamps = {key: ts for key, ts in self.edge_timestamps.items() if key in active_edge_original_keys}
        self.edge_features = {key: feat for key, feat in self.edge_features.items() if key in active_edge_original_keys}

        # --- 更新內部狀態 ---
        logger.debug("更新圖建立器內部狀態...")
        self.g = new_g
        self.existing_nodes = set(active_original_nodes) # 更新活躍原始 ID 集合
        self.existing_edges = active_edge_original_keys # 更新活躍原始邊集合
        self.node_id_map = new_node_id_map # 更新映射
        self.reverse_node_id_map = new_reverse_node_id_map # 更新反向映射
        self.next_internal_id = new_next_internal_id # 更新下一個內部 ID

        # --- 重建輔助映射 (src_to_dst, dst_to_src) ---
        logger.debug("重建輔助映射...")
        self.src_to_dst = defaultdict(set)
        self.dst_to_src = defaultdict(set)
        for src, dst in self.existing_edges:
            self.src_to_dst[src].add(dst)
            self.dst_to_src[dst].add(src)

        # --- 清理緩存和記憶體 ---
        self.neighbor_cache.clear()
        self.last_pruning_time = current_sys_time
        clean_memory(aggressive=True)
        end_prune_time = time.time()
        logger.info(f"節點清理完成，耗時 {end_prune_time - start_prune_time:.2f} 秒。")
        logger.info(f"清理後 DGL 圖: {self.g.num_nodes()} 個內部節點, {self.g.num_edges()} 條邊")
        print_memory_usage()


    @memory_usage_decorator
    def add_edges(self, src_nodes: List[Any], dst_nodes: List[Any],
                  timestamps: List[float], edge_feats: Optional[Union[np.ndarray, torch.Tensor, List]] = None):
        """
        添加邊到圖 - 使用節點 ID 映射 (修正版)

        參數:
            src_nodes: 源節點 ID 列表 (原始 ID)
            dst_nodes: 目標節點 ID 列表 (原始 ID)
            timestamps: 邊的時間戳列表 (對應邊)
            edge_feats: 邊的特徵 (Numpy array, Tensor 或列表, 可選)
        """
        if not src_nodes:
            logger.debug("add_edges: 收到空的 src_nodes 列表，不執行任何操作。")
            return

        num_input_edges = len(src_nodes)
        # --- 驗證輸入長度 ---
        if not (num_input_edges == len(dst_nodes) == len(timestamps)):
            raise ValueError(f"輸入列表長度必須相同: src={len(src_nodes)}, dst={len(dst_nodes)}, time={len(timestamps)}")
        if edge_feats is not None and len(edge_feats) != num_input_edges:
            raise ValueError(f"邊特徵數量 ({len(edge_feats)}) 與邊數量 ({num_input_edges}) 不匹配")

        # --- 處理邊特徵 ---
        edge_feats_tensor: Optional[torch.Tensor] = None
        if edge_feats is not None:
            # 設置或驗證邊特徵維度
            if self.edge_feat_dim is None and len(edge_feats) > 0:
                feat_example = edge_feats[0]
                # 處理 Tensor 或 Numpy 數組的情況
                if hasattr(feat_example, 'shape') and len(getattr(feat_example, 'shape', ())) > 0:
                     self.edge_feat_dim = feat_example.shape[0]
                # 處理列表或純量的情況
                elif hasattr(feat_example, '__len__'):
                     self.edge_feat_dim = len(feat_example)
                else:
                     self.edge_feat_dim = 1 # 標量
                logger.info(f"檢測到邊特徵維度: {self.edge_feat_dim}")

            # 確保是 Tensor
            if isinstance(edge_feats, np.ndarray):
                edge_feats_tensor = torch.tensor(edge_feats, dtype=torch.float32)
            elif isinstance(edge_feats, list):
                 try:
                     edge_feats_tensor = torch.tensor(edge_feats, dtype=torch.float32)
                 except Exception as e:
                      logger.error(f"無法將列表轉換為邊特徵 Tensor: {e}。將忽略邊特徵。")
                      edge_feats = None # 重置為 None
            elif isinstance(edge_feats, torch.Tensor):
                 edge_feats_tensor = edge_feats.float() # 確保是 float32
            else:
                 logger.error(f"不支持的邊特徵類型: {type(edge_feats)}。將忽略邊特徵。")
                 edge_feats = None

            # 維度檢查
            if edge_feats_tensor is not None:
                 if edge_feats_tensor.shape[0] != num_input_edges:
                      logger.error("邊特徵 Tensor 的第一維與邊數量不匹配！忽略邊特徵。")
                      edge_feats_tensor = None
                      edge_feats = None
                 elif self.edge_feat_dim is not None and edge_feats_tensor.ndim > 1 and edge_feats_tensor.shape[1] != self.edge_feat_dim:
                      logger.error(f"輸入邊特徵維度 ({edge_feats_tensor.shape[1]}) 與現有 ({self.edge_feat_dim}) 不匹配！忽略邊特徵。")
                      edge_feats_tensor = None
                      edge_feats = None


        # --- 篩選有效的、新的邊 ---
        new_internal_src_indices = []
        new_internal_dst_indices = []
        valid_edge_original_indices = [] # 記錄有效邊在輸入列表中的原始索引

        current_sys_time = time.time()
        max_timestamp_in_batch = 0.0

        for i, (src_orig, dst_orig) in enumerate(zip(src_nodes, dst_nodes)):
            edge_key_orig = (src_orig, dst_orig)
            # 檢查節點是否存在於映射中，且邊是新的
            if src_orig in self.node_id_map and dst_orig in self.node_id_map and \
               edge_key_orig not in self.existing_edges:

                internal_src = self.node_id_map[src_orig]
                internal_dst = self.node_id_map[dst_orig]

                new_internal_src_indices.append(internal_src)
                new_internal_dst_indices.append(internal_dst)
                valid_edge_original_indices.append(i)

                # 更新節點活躍時間
                self.node_last_active[src_orig] = current_sys_time
                self.node_last_active[dst_orig] = current_sys_time

                # 更新最大時間戳
                current_timestamp = float(timestamps[i]) # 確保是 float
                max_timestamp_in_batch = max(max_timestamp_in_batch, current_timestamp)

        if not new_internal_src_indices:
            logger.debug("沒有新的有效邊需要添加到 DGL 圖。")
            return # 沒有需要添加的新邊

        # --- 批量添加到 DGL 圖 (使用內部索引) ---
        num_edges_before = self.g.num_edges()
        self.g = dgl.add_edges(self.g, new_internal_src_indices, new_internal_dst_indices)
        added_count = self.g.num_edges() - num_edges_before
        logger.info(f"嘗試添加 {len(new_internal_src_indices)} 條邊，實際添加到 DGL 圖 {added_count} 條。")
        if added_count != len(new_internal_src_indices):
             # 這通常發生在 DGL 內部去除了重複邊或自環（取決於 DGL 版本和設置）
             logger.warning(f"DGL 添加邊的數量與預期不符，可能 DGL 自動處理了重複邊或自環。")


        # --- 更新邊相關的字典 (使用原始 ID 對作為鍵) ---
        for i, original_idx in enumerate(valid_edge_original_indices):
            # 需要確保 i 對應到正確添加的邊，如果 added_count < len(...) 則需要更複雜的映射
            # 為了簡化，我們先假設 add_edges 返回的順序與輸入對應（在無重複和自環時通常如此）
            # 如果 DGL 行為不確定，這裡可能需要基於 internal indices 重新查找邊 ID
            src_orig = src_nodes[original_idx]
            dst_orig = dst_nodes[original_idx]
            edge_key_orig = (src_orig, dst_orig)

            # 再次檢查邊是否已在集合中（以防 DGL 自動去重導致問題）
            if edge_key_orig not in self.existing_edges:
                self.existing_edges.add(edge_key_orig) # 添加到已存在邊集合
                self.edge_timestamps[edge_key_orig] = float(timestamps[original_idx])
                if edge_feats is not None:
                     # 從 Tensor 或原始列表中獲取對應特徵
                     feat_to_store = edge_feats_tensor[original_idx] if edge_feats_tensor is not None else edge_feats[original_idx]
                     self.edge_features[edge_key_orig] = feat_to_store

                # 更新輔助映射 (原始 ID)
                self.src_to_dst[src_orig].add(dst_orig)
                self.dst_to_src[dst_orig].add(src_orig)
            else:
                 # 這通常不應發生在篩選邏輯正確的情況下
                 logger.warning(f"邊 {edge_key_orig} 在更新字典時已存在於 existing_edges 中，跳過字典更新。")


        # 更新圖的當前邏輯時間
        self.current_time = max(self.current_time, max_timestamp_in_batch)

        # 清空鄰居採樣緩存
        if self.cache_neighbor_sampling: self.neighbor_cache.clear()

        logger.info(f"添加邊完成，當前 DGL 圖總邊數: {self.g.num_edges()}, 記錄的唯一邊數: {len(self.existing_edges)}")


    @memory_usage_decorator
    def update_temporal_graph(self, current_time: Optional[float] = None):
        """
        更新時間窗口子圖 (修正版：使用節點映射)

        參數:
            current_time: 指定當前時間戳 (可選, 否則使用內部 current_time)

        返回:
            dgl.DGLGraph: 時間窗口內的子圖 (在指定設備上) 或 None (如果圖為空)
        """
        # 確定用於過濾的時間戳
        filter_time = current_time if current_time is not None else self.current_time
        start_time = filter_time - self.temporal_window # 計算窗口起始時間
        logger.info(f"更新時間圖，窗口: [{start_time:.2f}, {filter_time:.2f}]")

        # --- 獲取當前全局圖 (在 CPU 上操作) ---
        current_g = self.g
        if current_g is None or current_g.num_nodes() == 0:
             logger.warning("全局圖為空，無法創建時間子圖。")
             self.temporal_g = dgl.graph(([],[]), num_nodes=0, device=self.device if self.device.type != 'cpu' else 'cpu')
             return self.temporal_g

        # --- 篩選時間窗口內的邊 ---
        # 創建布爾掩碼來標識在時間窗口內的邊
        edge_mask = torch.zeros(current_g.num_edges(), dtype=torch.bool)
        active_edge_original_keys_in_window = [] # 保存窗口內邊的原始鍵

        # 遍歷邊數據字典 (鍵是原始 ID 對)
        num_edges_in_window = 0
        for i, edge_key_orig in enumerate(self.existing_edges): # 迭代原始邊集合可能更清晰
             timestamp = self.edge_timestamps.get(edge_key_orig)
             if timestamp is not None and timestamp >= start_time and timestamp <= filter_time:
                 src_orig, dst_orig = edge_key_orig
                 # 查找對應的邊 ID (如果需要)
                 # DGL 的 edge_subgraph 可以直接接收邊 ID 列表
                 # 我們需要找到這個原始邊 key 對應的 DGL 內部邊 ID
                 # 這一步比較耗時，更好的方法是直接使用 dgl.edge_subgraph 按邊數據過濾
                 # 假設 edge_timestamps 的迭代順序與圖的邊順序相關（不安全）
                 # 或者在 edge_timestamps 中也存儲邊 ID?
                 # **簡化方法**: 我們先收集內部索引，再創建子圖
                 if src_orig in self.node_id_map and dst_orig in self.node_id_map:
                      # 找到這條邊在 current_g 中的 ID (可能有多條)
                      src_internal = self.node_id_map[src_orig]
                      dst_internal = self.node_id_map[dst_orig]
                      eids = current_g.edge_ids(src_internal, dst_internal, return_uv=False)
                      if len(eids) > 0:
                          edge_mask[eids] = True # 標記這些邊 ID
                          active_edge_original_keys_in_window.append(edge_key_orig)
                          num_edges_in_window += len(eids) # 計算窗口內的邊數

        logger.info(f"在時間窗口內找到 {num_edges_in_window} 條邊 (對應 {len(active_edge_original_keys_in_window)} 個原始鍵)")

        # 如果窗口內沒有邊
        if num_edges_in_window == 0:
             logger.info("時間窗口內沒有活躍的邊。")
             self.temporal_g = dgl.graph(([],[]), num_nodes=0, device=self.device if self.device.type != 'cpu' else 'cpu')
             return self.temporal_g

        # --- 創建時間子圖 ---
        # 使用掩碼創建邊子圖，並重新標記節點
        self.temporal_g = dgl.edge_subgraph(current_g, edge_mask, relabel_nodes=True)

        logger.info(f"創建時間子圖: {self.temporal_g.num_nodes()} 節點, {self.temporal_g.num_edges()} 邊")

        # --- 附加節點特徵和標籤到子圖 ---
        # 子圖的 ndata[dgl.NID] 儲存了其節點對應的原圖內部 ID
        original_node_internal_ids = self.temporal_g.ndata[dgl.NID]
        node_features_list = []
        node_labels_list = []

        for internal_idx in original_node_internal_ids.tolist(): # 轉為列表迭代
             original_id = self.reverse_node_id_map.get(internal_idx)
             if original_id is not None:
                  # 使用原始 ID 從字典獲取數據
                  feat = self.node_features.get(original_id, torch.zeros(self.node_feat_dim or 1))
                  label = self.node_labels.get(original_id, -1)
                  node_features_list.append(feat.float()) # 確保是 float32
                  node_labels_list.append(label)
             else:
                  logger.warning(f"附加節點數據時，內部索引 {internal_idx} 找不到對應的原始 ID。")
                  node_features_list.append(torch.zeros(self.node_feat_dim or 1))
                  node_labels_list.append(-1)

        if node_features_list:
             # 檢查維度是否一致
             feat_shapes = [f.shape for f in node_features_list]
             if len(set(feat_shapes)) > 1:
                 logger.error(f"子圖節點特徵維度不一致: {set(feat_shapes)}。無法堆疊。")
                 # 可以嘗試填充或報錯
             else:
                 self.temporal_g.ndata['feat'] = torch.stack(node_features_list)
        if node_labels_list:
             self.temporal_g.ndata['label'] = torch.tensor(node_labels_list, dtype=torch.long)

        # --- 附加邊特徵和時間戳到子圖 ---
        # 子圖的 edata[dgl.EID] 儲存了其邊對應的原圖邊 ID
        # 我們需要用這些原圖邊 ID 找到對應的原始鍵 (src_orig, dst_orig)
        original_edge_internal_ids = self.temporal_g.edata[dgl.EID]
        edge_features_list = []
        edge_timestamps_list = []
        missing_edge_data_count = 0

        # 獲取原圖的邊用於查找
        orig_src_all, orig_dst_all = current_g.edges()

        for edge_idx_in_subgraph, orig_edge_id in enumerate(original_edge_internal_ids.tolist()):
             # 獲取這條邊在原圖中的端點（內部索引）
             src_internal = orig_src_all[orig_edge_id].item()
             dst_internal = orig_dst_all[orig_edge_id].item()
             # 轉換為原始 ID
             src_orig = self.reverse_node_id_map.get(src_internal)
             dst_orig = self.reverse_node_id_map.get(dst_internal)

             if src_orig is None or dst_orig is None:
                 missing_edge_data_count += 1
                 edge_features_list.append(torch.zeros(self.edge_feat_dim or 1))
                 edge_timestamps_list.append(0.0)
                 continue

             edge_key_orig = (src_orig, dst_orig)
             # 使用原始鍵查找數據
             edge_feat = self.edge_features.get(edge_key_orig)
             edge_time = self.edge_timestamps.get(edge_key_orig)

             # 處理特徵
             if edge_feat is None:
                  edge_features_list.append(torch.zeros(self.edge_feat_dim or 1))
             elif not isinstance(edge_feat, torch.Tensor):
                  try:
                       edge_features_list.append(torch.tensor(edge_feat, dtype=torch.float32))
                  except: # 如果轉換失敗
                       edge_features_list.append(torch.zeros(self.edge_feat_dim or 1))
             else:
                  edge_features_list.append(edge_feat.float())

             # 處理時間戳
             if edge_time is None: edge_timestamps_list.append(0.0)
             else: edge_timestamps_list.append(float(edge_time))

        if missing_edge_data_count > 0:
            logger.warning(f"在附加邊數據時，有 {missing_edge_data_count} 條邊未能找到其原始節點 ID。")

        if edge_features_list:
             # 檢查維度一致性
             feat_shapes = [f.shape for f in edge_features_list]
             if len(set(feat_shapes)) > 1:
                 logger.error(f"子圖邊特徵維度不一致: {set(feat_shapes)}。無法堆疊。")
             else:
                 self.temporal_g.edata['feat'] = torch.stack(edge_features_list)
        if edge_timestamps_list:
             self.temporal_g.edata['time'] = torch.tensor(edge_timestamps_list, dtype=torch.float32)


        # --- DGL 優化和設備轉移 ---
        if self.use_dgl_transform:
            try:
                 # 添加自環（如果需要）
                 self.temporal_g = dgl.add_self_loop(self.temporal_g)
                 logger.debug("已添加自環到時間子圖。")
                 # 可以考慮其他圖轉換, e.g., dgl.to_bidirected
            except dgl.DGLError as e:
                 logger.warning(f"DGL 圖轉換失敗（例如，圖已包含自環或不支持的操作）: {e}")

        # 將最終的時間子圖移到目標設備
        if self.device.type != 'cpu':
            self.temporal_g = self.temporal_g.to(self.device)

        logger.info(f"更新時間圖完成: {self.temporal_g.num_nodes()} 節點, {self.temporal_g.num_edges()} 邊 (在 {self.temporal_g.device})")

        return self.temporal_g


    def get_graph(self, temporal: bool = False) -> Optional[dgl.DGLGraph]:
        """獲取當前圖或時間窗口子圖"""
        if temporal:
            if self.temporal_g is None:
                 logger.info("時間子圖尚未創建，將立即更新。")
                 self.update_temporal_graph()
            # 返回副本以避免外部修改影響內部狀態？取決於使用場景
            return self.temporal_g #.clone() if self.temporal_g else None
        else:
            # 返回全局圖的副本？
            return self.g #.clone() if self.g else None

    def get_node_features(self, node_ids: Optional[List[Any]] = None) -> Optional[torch.Tensor]:
        """
        獲取指定節點或所有活躍節點的特徵

        參數:
            node_ids: 節點 ID 列表 (原始 ID)。如果為 None，返回所有活躍節點的特徵。

        返回:
            包含節點特徵的 Tensor 或 None
        """
        default_feat = torch.zeros(self.node_feat_dim or 1)
        features_list = []

        if node_ids is None: # 獲取所有活躍節點的特徵
            if not self.node_features or not self.g or self.g.num_nodes() == 0:
                 logger.warning("無法獲取所有節點特徵：特徵字典為空或全局圖不存在。")
                 return None
            # 按當前 DGL 圖的內部索引順序返回特徵
            num_nodes_in_g = self.g.num_nodes()
            for internal_idx in range(num_nodes_in_g):
                  original_id = self.reverse_node_id_map.get(internal_idx)
                  if original_id is not None:
                       features_list.append(self.node_features.get(original_id, default_feat))
                  else:
                       logger.warning(f"獲取所有節點特徵時，內部索引 {internal_idx} 找不到原始 ID。")
                       features_list.append(default_feat)
        else: # 獲取指定節點的特徵
             for nid in node_ids:
                  # 檢查節點是否存在於活躍節點中 (可選)
                  if nid in self.existing_nodes:
                       features_list.append(self.node_features.get(nid, default_feat))
                  else:
                       logger.debug(f"請求獲取不存在或不活躍的節點 {nid} 的特徵，返回零向量。")
                       features_list.append(default_feat)

        if not features_list:
             logger.warning("未能獲取任何有效的節點特徵。")
             return None

        # 檢查特徵維度是否一致
        feat_shapes = {f.shape for f in features_list}
        if len(feat_shapes) > 1:
            logger.error(f"獲取的節點特徵維度不一致: {feat_shapes}。無法堆疊。")
            return None # 或者嘗試填充/截斷

        try:
            return torch.stack(features_list).float() # 確保是 float32
        except Exception as e:
             logger.error(f"堆疊節點特徵時出錯: {e}", exc_info=True)
             return None


    def build_graph(self, features: Union[np.ndarray, torch.Tensor],
                  target: Union[np.ndarray, torch.Tensor],
                  edge_creation_method: str = 'knn', k: int = 5) -> Optional[dgl.DGLGraph]:
         """
         使用提供的特徵和目標數據建立一個靜態圖。主要用於訓練/評估前的準備。

         參數:
             features: 節點特徵 (Numpy array 或 Tensor, [num_nodes, feat_dim])
             target: 節點標籤 (Numpy array 或 Tensor, [num_nodes])
             edge_creation_method: 邊創建方法 ('knn')
             k: KNN 中的鄰居數量

         返回:
             dgl.DGLGraph: 創建的圖 (包含 'feat' 和 'label' 數據) 或 None
         """
         if features is None or target is None:
             logger.error("無法建立靜態圖，缺少特徵或目標數據。")
             return None

         n_nodes = features.shape[0]
         if n_nodes != len(target):
             logger.error(f"建立靜態圖錯誤：特徵行數 ({n_nodes}) 與目標數量 ({len(target)}) 不匹配！")
             return None

         logger.info(f"基於提供的數據建立靜態圖，方法: {edge_creation_method}，節點數: {n_nodes}")

         # 確保數據是 Tensor
         if isinstance(features, np.ndarray): features = torch.tensor(features, dtype=torch.float32)
         if isinstance(target, np.ndarray): target = torch.tensor(target, dtype=torch.long) # 標籤通常是 Long

         # 將特徵移到 CPU 進行 KNN 計算
         features_cpu = features.cpu()

         g = None
         if edge_creation_method == 'knn':
             from sklearn.neighbors import kneighbors_graph # 延遲導入
             try:
                 # 清理 NaN/Inf
                 if torch.any(torch.isnan(features_cpu)) or torch.any(torch.isinf(features_cpu)):
                     logger.warning("靜態圖建立：特徵包含 NaN/Inf，使用 0 填充。")
                     features_cpu = torch.nan_to_num(features_cpu, nan=0.0, posinf=0.0, neginf=0.0)

                 logger.info(f"計算 KNN 圖 (k={k})...")
                 # kneighbors_graph 需要 numpy
                 adj_matrix = kneighbors_graph(features_cpu.numpy(), n_neighbors=k, mode='connectivity', include_self=False)
                 src, dst = adj_matrix.nonzero()
                 logger.info(f"KNN 圖邊數: {len(src)}")

                 # 創建 DGL 圖 (在 CPU 上)
                 g = dgl.graph((src, dst), num_nodes=n_nodes, device='cpu')

             except ImportError:
                 logger.error("建立 KNN 圖需要安裝 scikit-learn。請運行 'pip install scikit-learn'。")
                 return None
             except Exception as e:
                 logger.error(f"建立 KNN 圖時出錯: {e}", exc_info=True)
                 return None
         else:
             logger.error(f"不支持的靜態圖邊創建方法: {edge_creation_method}")
             return None

         # 添加節點特徵和標籤
         g.ndata['feat'] = features # 使用原始的、可能在 GPU 上的 Tensor
         g.ndata['label'] = target.long() # 確保標籤是 LongTensor

         logger.info(f"靜態圖建立完成: {g.num_nodes()} 節點, {g.num_edges()} 邊")
         # 將圖移到目標設備
         g = g.to(self.device)
         logger.info(f"靜態圖已移至設備: {g.device}")
         return g

    # --- 其他方法 (simulate_stream, to_sparse_tensor) ---
    # 警告：這些方法可能需要根據新的 ID 映射機制進行審查和調整
    def simulate_stream(self, *args, **kwargs):
        logger.warning("simulate_stream 方法尚未針對新的 ID 映射機制進行驗證，可能需要調整。")
        # 保留原始邏輯或進行修改
        pass

    def to_sparse_tensor(self):
        logger.warning("to_sparse_tensor 方法尚未針對新的 ID 映射機制進行驗證，可能需要調整。")
        # 保留原始邏輯或進行修改
        if not self.g: return None
        # ... (原始邏輯) ...
        pass
