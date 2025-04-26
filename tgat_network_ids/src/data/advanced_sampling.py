#!/usr/bin/env python
# coding: utf-8 -*-

"""
進階圖採樣策略模組

實現多種先進的圖採樣算法，用於大規模圖上的高效訓練，同時減少採樣過程中的信息損失。
支持的採樣策略包括：
1. GraphSAINT - 隨機行走和節點採樣方法
2. Cluster-GCN - 基於圖聚類的採樣
3. Frontier採樣 - 基於層次的鄰居採樣
4. 歷史感知採樣 - 保留重要歷史信息的採樣
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import dgl
import torch
import logging
from sklearn.cluster import KMeans, SpectralClustering
from collections import defaultdict, deque
import random
from tqdm import tqdm
import time
from datetime import datetime
from typing import Dict, List, Tuple, Set, Union, Optional, Callable, Any

# 配置日誌
logger = logging.getLogger(__name__)

class AdvancedGraphSampler:
    """進階圖採樣器
    
    實現多種先進的採樣方法，專注於減少時間動態圖採樣過程中的信息損失。
    """
    
    def __init__(self, config=None):
        """初始化進階圖採樣器
        
        參數:
            config (dict): 採樣配置字典
        """
        self.config = config or {}
        
        # 採樣配置
        self.sampling_method = self.config.get('sampling_method', 'graphsaint')
        self.sample_size = self.config.get('sample_size', 5000)
        self.batch_size = self.config.get('batch_size', 1000)
        self.num_clusters = self.config.get('num_clusters', 50)
        self.walk_length = self.config.get('walk_length', 10)
        self.num_walks = self.config.get('num_walks', 30)
        self.reset_between_batches = self.config.get('reset_between_batches', True)
        
        # 記憶機制配置
        self.use_memory = self.config.get('use_memory', True)
        self.memory_size = self.config.get('memory_size', 1000)
        self.memory_update_strategy = self.config.get('memory_update_strategy', 'fifo')
        self.importance_sampling = self.config.get('importance_sampling', True)
        
        # 位置嵌入配置
        self.use_position_embedding = self.config.get('use_position_embedding', True)
        self.position_embedding_dim = self.config.get('position_embedding_dim', 32)
        
        # 初始化記憶緩衝區
        self._initialize_memory()
        
        logger.info(f"初始化進階圖採樣器: 方法={self.sampling_method}, 採樣大小={self.sample_size}")
        logger.info(f"記憶機制: 啟用={self.use_memory}, 大小={self.memory_size}")
    
    def _initialize_memory(self):
        """初始化記憶緩衝區"""
        self.memory_buffer = {
            'nodes': [],           # 記憶節點ID
            'importance': [],      # 節點重要性分數
            'last_access': [],     # 最後訪問時間戳
            'access_count': [],    # 訪問次數
            'features': [],        # 節點特徵 
            'timestamps': [],      # 節點時間戳
            'edge_indices': [],    # 記憶邊索引
            'edge_timestamps': [], # 邊時間戳
            'edge_features': []    # 邊特徵
        }
        
        # 節點索引映射
        self.memory_node_idx_map = {}
        
        # 記憶緩衝區使用統計
        self.memory_stats = {
            'hit_count': 0,
            'miss_count': 0,
            'update_count': 0,
            'eviction_count': 0
        }
    
    def sample_subgraph(self, graph, features=None, timestamps=None, time_window=None):
        """根據指定的採樣方法採樣子圖
        
        參數:
            graph: 原始圖 (DGL或NetworkX格式)
            features: 節點特徵
            timestamps: 時間戳
            time_window: 時間窗口 (開始時間, 結束時間)
            
        返回:
            sampled_graph: 採樣的子圖
            sampled_features: 採樣節點的特徵
        """
        # 確保圖格式兼容性
        if isinstance(graph, dgl.DGLGraph):
            # 使用DGL圖
            self.is_dgl = True
            # 如果是DGL圖，可能需要提取一些信息
            num_nodes = graph.num_nodes()
            edges = graph.edges()
        elif isinstance(graph, nx.Graph):
            # 轉換為DGL圖
            self.is_dgl = False
            dgl_graph = dgl.from_networkx(graph)
            num_nodes = dgl_graph.num_nodes()
            edges = dgl_graph.edges()
        else:
            raise TypeError("圖必須是DGL.DGLGraph或NetworkX.Graph類型")
        
        # 選擇採樣方法
        if self.sampling_method == 'graphsaint':
            return self._graphsaint_sampling(graph, features, timestamps, time_window)
        elif self.sampling_method == 'cluster-gcn':
            return self._cluster_gcn_sampling(graph, features, timestamps, time_window)
        elif self.sampling_method == 'frontier':
            return self._frontier_sampling(graph, features, timestamps, time_window)
        elif self.sampling_method == 'historical':
            return self._historical_sampling(graph, features, timestamps, time_window)
        else:
            logger.warning(f"未知的採樣方法: {self.sampling_method}，使用默認GraphSAINT")
            return self._graphsaint_sampling(graph, features, timestamps, time_window)
    
    def _graphsaint_sampling(self, graph, features=None, timestamps=None, time_window=None):
        """GraphSAINT採樣方法
        
        實現基於節點和基於隨機行走的混合策略，針對時間動態圖進行優化
        """
        logger.info("執行GraphSAINT採樣...")
        
        # 檢查是否需要進行時間過濾
        if time_window is not None:
            start_time, end_time = time_window
            # 假設時間戳在邊的時間戳屬性上
            time_filtered_graph = self._filter_graph_by_time(graph, timestamps, start_time, end_time)
        else:
            time_filtered_graph = graph
        
        # 獲取圖的節點數和邊
        if isinstance(time_filtered_graph, dgl.DGLGraph):
            num_nodes = time_filtered_graph.num_nodes()
            edges = time_filtered_graph.edges()
            src_nodes, dst_nodes = edges[0].numpy(), edges[1].numpy()
        else:
            num_nodes = time_filtered_graph.number_of_nodes()
            edges = list(time_filtered_graph.edges())
            src_nodes, dst_nodes = zip(*edges) if edges else ([], [])
        
        # 節點採樣階段
        # 計算節點重要性分數，可能基於度數或其他中心度度量
        node_importance = self._calculate_node_importance(time_filtered_graph)
        
        # 根據重要性進行節點採樣 (優先採樣重要節點)
        if self.importance_sampling and len(node_importance) > 0:
            importance_sum = sum(node_importance.values())
            if importance_sum > 0:
                sample_probs = [node_importance.get(i, 0) / importance_sum for i in range(num_nodes)]
                sampled_nodes = np.random.choice(
                    num_nodes, 
                    size=min(self.sample_size, num_nodes),
                    replace=False, 
                    p=sample_probs
                )
            else:
                sampled_nodes = np.random.choice(
                    num_nodes, 
                    size=min(self.sample_size, num_nodes),
                    replace=False
                )
        else:
            # 均勻隨機採樣
            sampled_nodes = np.random.choice(
                num_nodes, 
                size=min(self.sample_size, num_nodes),
                replace=False
            )
        
        sampled_nodes_set = set(sampled_nodes)
        
        # 隨機行走採樣階段 - 從採樣節點開始隨機行走
        additional_nodes = set()
        
        for start_node in np.random.choice(sampled_nodes, size=min(self.num_walks, len(sampled_nodes)), replace=False):
            current_node = start_node
            for _ in range(self.walk_length):
                # 獲取當前節點的鄰居
                if isinstance(time_filtered_graph, dgl.DGLGraph):
                    neighbors = time_filtered_graph.successors(current_node).numpy()
                else:
                    neighbors = list(time_filtered_graph.neighbors(current_node))
                
                if len(neighbors) == 0:
                    break
                    
                # 選擇下一個節點
                next_node = np.random.choice(neighbors)
                additional_nodes.add(next_node)
                current_node = next_node
        
        # 合併節點集
        all_sampled_nodes = sampled_nodes_set.union(additional_nodes)
        
        # 檢查記憶緩衝區中是否有額外重要節點應包含
        if self.use_memory and len(self.memory_buffer['nodes']) > 0:
            # 獲取記憶中的高重要性節點
            memory_nodes = set(self.memory_buffer['nodes'])
            importance_threshold = np.percentile(self.memory_buffer['importance'], 75)  # 75th分位數
            
            high_importance_memory_nodes = {
                node for i, node in enumerate(self.memory_buffer['nodes'])
                if self.memory_buffer['importance'][i] >= importance_threshold
            }
            
            # 添加高重要性記憶節點
            all_sampled_nodes = all_sampled_nodes.union(high_importance_memory_nodes)
            
            # 更新記憶訪問統計
            overlap_nodes = memory_nodes.intersection(all_sampled_nodes)
            self.memory_stats['hit_count'] += len(overlap_nodes)
            self.memory_stats['miss_count'] += len(all_sampled_nodes - memory_nodes)
        
        # 構建採樣子圖
        if isinstance(time_filtered_graph, dgl.DGLGraph):
            # 使用DGL構建子圖
            sampled_graph = dgl.node_subgraph(time_filtered_graph, list(all_sampled_nodes))
            
            # 提取特徵
            if features is not None:
                if isinstance(features, torch.Tensor):
                    sampled_features = features[list(all_sampled_nodes)]
                elif isinstance(features, np.ndarray):
                    sampled_features = features[list(all_sampled_nodes)]
                else:
                    sampled_features = None
            else:
                sampled_features = None
        else:
            # 使用NetworkX構建子圖
            sampled_graph = time_filtered_graph.subgraph(all_sampled_nodes)
            
            # 提取特徵
            if features is not None:
                sampled_features = {node: features[node] for node in all_sampled_nodes if node in features}
            else:
                sampled_features = None
        
        # 更新記憶緩衝區
        if self.use_memory:
            self._update_memory(list(all_sampled_nodes), features, timestamps)
        
        # 添加位置嵌入 (如果啟用)
        if self.use_position_embedding:
            sampled_graph = self._add_position_embeddings(sampled_graph)
        
        logger.info(f"GraphSAINT採樣完成: 節點數={len(all_sampled_nodes)}")
        return sampled_graph, sampled_features
    
    def _cluster_gcn_sampling(self, graph, features=None, timestamps=None, time_window=None):
        """Cluster-GCN採樣方法
        
        通過圖聚類產生子圖，對時間動態圖的實現進行優化
        """
        logger.info("執行Cluster-GCN採樣...")
        
        # 檢查是否需要進行時間過濾
        if time_window is not None:
            start_time, end_time = time_window
            time_filtered_graph = self._filter_graph_by_time(graph, timestamps, start_time, end_time)
        else:
            time_filtered_graph = graph
        
        # 將圖轉換為NetworkX格式進行聚類
        if isinstance(time_filtered_graph, dgl.DGLGraph):
            nx_graph = dgl.to_networkx(time_filtered_graph)
        else:
            nx_graph = time_filtered_graph
        
        # 如果圖太大，先隨機採樣一部分節點
        max_nodes_for_clustering = 10000
        if nx_graph.number_of_nodes() > max_nodes_for_clustering:
            nodes = list(nx_graph.nodes())
            sampled_nodes = np.random.choice(nodes, size=max_nodes_for_clustering, replace=False)
            nx_graph = nx_graph.subgraph(sampled_nodes)
        
        # 獲取鄰接矩陣
        adj_matrix = nx.adjacency_matrix(nx_graph)
        
        # 執行圖聚類
        num_clusters = min(self.num_clusters, nx_graph.number_of_nodes())
        
        try:
            # 嘗試使用光譜聚類
            clustering = SpectralClustering(
                n_clusters=num_clusters,
                affinity='precomputed',
                assign_labels='discretize',
                random_state=42
            )
            # 使用鄰接矩陣作為相似性矩陣
            cluster_labels = clustering.fit_predict(adj_matrix)
        except Exception as e:
            logger.warning(f"光譜聚類失敗: {str(e)}，回退到K-means")
            # 回退到基於節點特徵的K-means
            if features is not None:
                if isinstance(features, torch.Tensor):
                    node_features = features.numpy()
                else:
                    node_features = features
                
                # 確保只使用所選節點的特徵
                if isinstance(node_features, dict):
                    feature_matrix = np.array([node_features[n] for n in nx_graph.nodes()])
                else:
                    nodes_list = list(nx_graph.nodes())
                    feature_matrix = node_features[nodes_list]
                
                # 執行K-means
                clustering = KMeans(n_clusters=num_clusters, random_state=42)
                cluster_labels = clustering.fit_predict(feature_matrix)
            else:
                # 如果沒有特徵，使用度數作為一維特徵
                degrees = np.array([deg for _, deg in nx_graph.degree()])
                clustering = KMeans(n_clusters=num_clusters, random_state=42)
                cluster_labels = clustering.fit_predict(degrees.reshape(-1, 1))
        
        # 隨機選擇一個或多個聚類
        unique_clusters = np.unique(cluster_labels)
        selected_clusters = np.random.choice(
            unique_clusters, 
            size=min(3, len(unique_clusters)),  # 選擇1-3個聚類
            replace=False
        )
        
        # 獲取所選聚類的節點
        nodes_list = list(nx_graph.nodes())
        sampled_nodes = set()
        
        for cluster_id in selected_clusters:
            cluster_nodes = [nodes_list[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
            sampled_nodes.update(cluster_nodes)
        
        # 限制總節點數
        if len(sampled_nodes) > self.sample_size:
            sampled_nodes = set(random.sample(list(sampled_nodes), self.sample_size))
        
        # 檢查記憶緩衝區中是否有額外重要節點
        if self.use_memory and len(self.memory_buffer['nodes']) > 0:
            # 加入重要記憶節點
            importance_scores = np.array(self.memory_buffer['importance'])
            if len(importance_scores) > 0:
                threshold = np.percentile(importance_scores, 80)  # 80th分位數
                important_memory_indices = [i for i, score in enumerate(importance_scores) if score >= threshold]
                important_memory_nodes = [self.memory_buffer['nodes'][i] for i in important_memory_indices]
                
                # 限制添加的記憶節點數量
                max_memory_nodes = min(len(important_memory_nodes), self.sample_size // 10)
                if max_memory_nodes > 0:
                    memory_nodes_to_add = set(random.sample(important_memory_nodes, max_memory_nodes))
                    
                    # 更新統計
                    self.memory_stats['hit_count'] += len(memory_nodes_to_add)
                    
                    # 添加到採樣節點
                    sampled_nodes.update(memory_nodes_to_add)
        
        # 構建採樣子圖
        if isinstance(time_filtered_graph, dgl.DGLGraph):
            sampled_graph = dgl.node_subgraph(time_filtered_graph, list(sampled_nodes))
            
            # 提取特徵
            if features is not None:
                if isinstance(features, torch.Tensor):
                    sampled_features = features[list(sampled_nodes)]
                elif isinstance(features, np.ndarray):
                    sampled_features = features[list(sampled_nodes)]
                else:
                    sampled_features = None
            else:
                sampled_features = None
        else:
            sampled_graph = time_filtered_graph.subgraph(sampled_nodes)
            
            # 提取特徵
            if features is not None:
                sampled_features = {node: features[node] for node in sampled_nodes if node in features}
            else:
                sampled_features = None
        
        # 更新記憶緩衝區
        if self.use_memory:
            self._update_memory(list(sampled_nodes), features, timestamps)
        
        # 添加位置嵌入
        if self.use_position_embedding:
            sampled_graph = self._add_position_embeddings(sampled_graph)
            
        logger.info(f"Cluster-GCN採樣完成: 節點數={len(sampled_nodes)}")
        return sampled_graph, sampled_features
    
    def _frontier_sampling(self, graph, features=None, timestamps=None, time_window=None):
        """Frontier採樣方法
        
        實現基於層次的鄰居採樣，專注於保持結構信息
        """
        logger.info("執行Frontier採樣...")
        
        # 檢查是否需要進行時間過濾
        if time_window is not None:
            start_time, end_time = time_window
            time_filtered_graph = self._filter_graph_by_time(graph, timestamps, start_time, end_time)
        else:
            time_filtered_graph = graph
        
        # 獲取節點數和基本圖信息
        if isinstance(time_filtered_graph, dgl.DGLGraph):
            num_nodes = time_filtered_graph.num_nodes()
            
            # 從高度中心的節點開始
            if time_filtered_graph.num_edges() > 0:
                # 計算度數
                degrees = time_filtered_graph.in_degrees() + time_filtered_graph.out_degrees()
                degrees = degrees.numpy()
            else:
                degrees = np.zeros(num_nodes)
        else:
            num_nodes = time_filtered_graph.number_of_nodes()
            # 計算度數
            degrees = np.array([deg for _, deg in time_filtered_graph.degree()])
        
        # 選擇初始節點 - 優先選擇高度數節點
        if np.sum(degrees) > 0:
            node_probs = degrees / np.sum(degrees)
            seed_nodes = np.random.choice(
                num_nodes, 
                size=min(100, num_nodes),  # 選100個種子點
                replace=False, 
                p=node_probs
            )
        else:
            seed_nodes = np.random.choice(
                num_nodes, 
                size=min(100, num_nodes),
                replace=False
            )
        
        # 檢查記憶緩衝區是否有高重要性節點可以作為種子
        if self.use_memory and len(self.memory_buffer['nodes']) > 0:
            memory_importance = np.array(self.memory_buffer['importance'])
            if len(memory_importance) > 0:
                threshold = np.percentile(memory_importance, 90)  # 90th分位數
                important_indices = [i for i, score in enumerate(memory_importance) if score >= threshold]
                
                # 限制使用的記憶節點數
                max_memory_seeds = min(len(important_indices), 20)
                if max_memory_seeds > 0:
                    memory_seed_indices = random.sample(important_indices, max_memory_seeds)
                    memory_seeds = [self.memory_buffer['nodes'][i] for i in memory_seed_indices]
                    
                    # 確保記憶種子節點在當前圖中
                    valid_memory_seeds = []
                    for node in memory_seeds:
                        if isinstance(time_filtered_graph, dgl.DGLGraph):
                            if node < time_filtered_graph.num_nodes():
                                valid_memory_seeds.append(node)
                        else:
                            if node in time_filtered_graph:
                                valid_memory_seeds.append(node)
                    
                    # 更新種子節點
                    if valid_memory_seeds:
                        seed_nodes = np.concatenate([seed_nodes, valid_memory_seeds])
                        self.memory_stats['hit_count'] += len(valid_memory_seeds)
        
        # 從種子節點開始進行層次採樣
        sampled_nodes = set(seed_nodes)
        frontier = set(seed_nodes)
        visited = set(seed_nodes)
        
        # 進行3層採樣
        for layer in range(3):
            next_frontier = set()
            
            for node in frontier:
                # 獲取鄰居
                if isinstance(time_filtered_graph, dgl.DGLGraph):
                    in_neighbors = time_filtered_graph.predecessors(node).numpy()
                    out_neighbors = time_filtered_graph.successors(node).numpy()
                    neighbors = np.concatenate([in_neighbors, out_neighbors])
                else:
                    neighbors = list(time_filtered_graph.neighbors(node))
                
                # 對鄰居進行採樣 - 根據層次降低採樣比例
                sample_ratio = 0.5 ** (layer + 1)  # 採樣比例隨層數遞減
                num_samples = max(1, int(len(neighbors) * sample_ratio))
                
                if len(neighbors) > 0:
                    # 優先選擇未訪問的節點
                    unvisited = [n for n in neighbors if n not in visited]
                    if unvisited:
                        sampled_neighbors = random.sample(
                            unvisited, 
                            min(num_samples, len(unvisited))
                        )
                    else:
                        sampled_neighbors = random.sample(
                            list(neighbors), 
                            min(num_samples, len(neighbors))
                        )
                    
                    next_frontier.update(sampled_neighbors)
                    visited.update(sampled_neighbors)
                    sampled_nodes.update(sampled_neighbors)
            
            frontier = next_frontier
            
            # 檢查是否達到目標樣本大小
            if len(sampled_nodes) >= self.sample_size:
                break
        
        # 限制最終節點數量
        if len(sampled_nodes) > self.sample_size:
            sampled_nodes = set(random.sample(list(sampled_nodes), self.sample_size))
        
        # 構建採樣子圖
        if isinstance(time_filtered_graph, dgl.DGLGraph):
            sampled_graph = dgl.node_subgraph(time_filtered_graph, list(sampled_nodes))
            
            # 提取特徵
            if features is not None:
                if isinstance(features, torch.Tensor):
                    sampled_features = features[list(sampled_nodes)]
                elif isinstance(features, np.ndarray):
                    sampled_features = features[list(sampled_nodes)]
                else:
                    sampled_features = None
            else:
                sampled_features = None
        else:
            sampled_graph = time_filtered_graph.subgraph(sampled_nodes)
            
            # 提取特徵
            if features is not None:
                sampled_features = {node: features[node] for node in sampled_nodes if node in features}
            else:
                sampled_features = None
        
        # 更新記憶緩衝區
        if self.use_memory:
            self._update_memory(list(sampled_nodes), features, timestamps)
        
        # 添加位置嵌入
        if self.use_position_embedding:
            sampled_graph = self._add_position_embeddings(sampled_graph)
            
        logger.info(f"Frontier採樣完成: 節點數={len(sampled_nodes)}")
        return sampled_graph, sampled_features
    
    def _historical_sampling(self, graph, features=None, timestamps=None, time_window=None):
        """歷史感知採樣方法
        
        專門為時間動態圖設計，保留重要的歷史信息
        """
        logger.info("執行歷史感知採樣...")
        
        # 檢查時間窗口
        if time_window is None:
            logger.warning("未提供時間窗口，無法進行有效的歷史採樣")
            # 回退到GraphSAINT
            return self._graphsaint_sampling(graph, features, timestamps)
        
        current_start, current_end = time_window
        
        # 計算歷史窗口 - 比當前窗口更早
        window_duration = current_end - current_start
        history_start = max(0, current_start - window_duration)  # 使用與當前窗口相同大小的歷史窗口
        history_end = current_start
        
        # 提取歷史子圖
        historical_graph = self._filter_graph_by_time(graph, timestamps, history_start, history_end)
        
        # 提取當前子圖
        current_graph = self._filter_graph_by_time(graph, timestamps, current_start, current_end)
        
        # 從當前圖開始採樣 (使用GraphSAINT採樣)
        # 確保採樣大小適應歷史和當前的混合
        current_sample_size = int(self.sample_size * 0.7)  # 70%用於當前圖
        
        # 使用GraphSAINT採樣當前子圖
        if isinstance(current_graph, dgl.DGLGraph) and current_graph.num_nodes() > 0:
            current_importance = self._calculate_node_importance(current_graph)
            
            # 優先採樣重要節點
            if current_importance and sum(current_importance.values()) > 0:
                importance_sum = sum(current_importance.values())
                current_nodes = list(range(current_graph.num_nodes()))
                sample_probs = [current_importance.get(i, 0) / importance_sum for i in current_nodes]
                
                current_sampled_nodes = np.random.choice(
                    current_nodes,
                    size=min(current_sample_size, len(current_nodes)),
                    replace=False,
                    p=sample_probs
                )
            else:
                # 隨機均勻採樣
                current_nodes = list(range(current_graph.num_nodes()))
                current_sampled_nodes = np.random.choice(
                    current_nodes,
                    size=min(current_sample_size, len(current_nodes)),
                    replace=False
                )
        elif isinstance(current_graph, nx.Graph) and current_graph.number_of_nodes() > 0:
            current_importance = self._calculate_node_importance(current_graph)
            
            # 優先採樣重要節點
            if current_importance and sum(current_importance.values()) > 0:
                importance_sum = sum(current_importance.values())
                current_nodes = list(current_graph.nodes())
                node_to_idx = {node: i for i, node in enumerate(current_nodes)}
                
                sample_probs = [current_importance.get(node, 0) / importance_sum for node in current_nodes]