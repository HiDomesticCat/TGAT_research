#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
動態圖結構建立模組

此模組負責：
1. 將網路封包資料轉換為圖結構
2. 動態更新圖結構
3. 維護時間性特徵

基於 DGL (Deep Graph Library) 實作圖結構
"""

import numpy as np
import pandas as pd
import torch
import dgl
import logging
from collections import defaultdict
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DynamicNetworkGraph:
    """動態網路圖結構類別"""
    
    def __init__(self, device='cuda'):
        """
        初始化動態網路圖結構
        
        參數:
            device (str): 計算裝置 ('cpu' 或 'cuda')
        """
        self.device = device
        self.g = None  # 主圖
        self.node_features = {}  # 節點特徵
        self.edge_timestamps = {}  # 邊的時間戳記
        self.edge_features = {}  # 邊特徵
        self.node_timestamps = {}  # 節點時間戳記
        self.node_labels = {}  # 節點標籤
        self.current_time = 0  # 當前時間
        self.temporal_g = None  # 時間子圖
        self.temporal_window = 600  # 時間窗口 (秒)
        
        # 記錄已存在的節點和邊，用於快速查詢
        self.existing_nodes = set()
        self.existing_edges = set()
        
        # 用於動態跟踪每個源IP到目標IP的連接
        self.src_to_dst = defaultdict(set)
        self.dst_to_src = defaultdict(set)
        
        # 特徵維度
        self.node_feat_dim = None
        self.edge_feat_dim = None
        
        # 初始化一個空圖
        self._init_graph()
    
    def _init_graph(self):
        """初始化一個空圖"""
        self.g = dgl.graph(([],  # 源節點
                           []),  # 目標節點
                          num_nodes=0,
                          idtype=torch.int64,
                          device=self.device)
        
        logger.info(f"初始化空圖於裝置 {self.device}")
    
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
        
        if not new_edges:
            return
        
        # 獲取節點的圖索引
        src_indices = [self.g.nodes().tolist().index(nid) if nid in self.g.nodes().tolist() else None for nid in new_src]
        dst_indices = [self.g.nodes().tolist().index(nid) if nid in self.g.nodes().tolist() else None for nid in new_dst]
        
        # 過濾無效索引
        valid_edges = [(i, s, d) for i, (s, d) in enumerate(zip(src_indices, dst_indices)) if s is not None and d is not None]
        if not valid_edges:
            return
        
        valid_indices, valid_src, valid_dst = zip(*valid_edges)
        
        # 添加新邊到圖
        self.g.add_edges(valid_src, valid_dst)
        
        # 更新邊特徵和時間戳記
        for i, idx in enumerate(valid_indices):
            edge_key = new_edges[idx]
            self.edge_timestamps[edge_key] = new_timestamps[idx]
            if edge_feats is not None:
                self.edge_features[edge_key] = new_edge_feats[idx]
            self.existing_edges.add(edge_key)
            
            # 更新來源到目標的映射
            src, dst = edge_key
            self.src_to_dst[src].add(dst)
            self.dst_to_src[dst].add(src)
        
        # 更新當前時間為最新的時間戳記
        if new_timestamps:
            self.current_time = max(self.current_time, max(new_timestamps))
        
        logger.info(f"添加 {len(valid_edges)} 條新邊，當前共 {self.g.num_edges()} 條邊")
    
    def add_edges_in_batches(self, src_nodes, dst_nodes, timestamps, edge_feats=None, batch_size=10000):
        """
        批量添加邊到圖
        
        參數:
            src_nodes (list): 源節點ID列表
            dst_nodes (list): 目標節點ID列表
            timestamps (list): 邊的時間戳記列表
            edge_feats (list, optional): 邊特徵列表
            batch_size (int): 每批添加的邊數量
        """
        if len(src_nodes) != len(dst_nodes) or len(src_nodes) != len(timestamps):
            raise ValueError("源節點、目標節點和時間戳記列表長度必須相同")
        
        if not src_nodes:
            return
        
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
        
        for (src, dst), timestamp in self.edge_timestamps.items():
            if timestamp >= start_time:
                # 獲取節點索引
                src_idx = sorted(list(self.existing_nodes)).index(src)
                dst_idx = sorted(list(self.existing_nodes)).index(dst)
                
                temporal_src.append(src_idx)
                temporal_dst.append(dst_idx)
                
                if (src, dst) in self.edge_features:
                    temporal_edge_feats.append(self.edge_features[(src, dst)])
                else:
                    temporal_edge_feats.append([0.0] * self.edge_feat_dim)
                
                temporal_edge_times.append(timestamp)
        
        # 建立時間子圖
        self.temporal_g = dgl.graph((temporal_src, temporal_dst), 
                                   num_nodes=len(self.existing_nodes),
                                   idtype=torch.int64,
                                   device=self.device)
        
        # 設置節點特徵
        self.temporal_g.ndata['h'] = self.get_node_features().to('cuda')
        
        # 設置邊特徵和時間戳記
        if temporal_edge_feats:
            self.temporal_g.edata['h'] = torch.tensor(temporal_edge_feats).to('cuda')
            self.temporal_g.edata['time'] = torch.tensor(temporal_edge_times).to('cuda')
        
        # 設置節點標籤
        node_labels = self.get_node_labels().to('cuda')
        if len(node_labels) > 0:
            self.temporal_g.ndata['label'] = node_labels
        
        logger.info(f"更新時間圖: {len(temporal_src)} 條邊在時間窗口 {start_time} 到 {self.current_time}")
        
        return self.temporal_g
    
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
            self.add_edges_in_batches(src_nodes, dst_nodes, edge_timestamps, edge_feats, batch_size=10000)
        
        # 更新時間圖
        self.update_temporal_graph()
        
        return self.temporal_g
    
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
        
        # 添加自動生成的邊
        if src_nodes:
            self.add_edges_in_batches(src_nodes, dst_nodes, edge_timestamps, edge_feats, batch_size=10000)
        
        # 更新時間圖
        self.update_temporal_graph()
        
        return self.temporal_g

# 測試圖構建器
if __name__ == "__main__":
    import numpy as np
    
    # 建立測試資料
    n_nodes = 10
    node_ids = list(range(n_nodes))
    features = np.random.randn(n_nodes, 5)  # 5維特徵
    timestamps = [float(i) for i in range(n_nodes)]
    labels = [i % 2 for i in range(n_nodes)]  # 二元標籤
    
    # 建立圖
    graph_builder = DynamicNetworkGraph()
    
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