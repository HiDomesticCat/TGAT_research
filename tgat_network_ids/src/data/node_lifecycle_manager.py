#!/usr/bin/env python
# coding: utf-8 -*-

"""
節點生命週期管理模組

實現智能節點重要性評估、活躍度追蹤和記憶機制，解決時間動態圖中節點過早清理的問題。
主要功能包括:
1. 自適應不活躍閾值
2. 多準則複合重要性評分
3. 節點休眠與重新激活機制
4. 輕量級頻繁模式挖掘
"""

import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime
import networkx as nx
from collections import defaultdict, Counter, deque
import heapq
from typing import Dict, List, Tuple, Set, Union, Optional, Any
import torch
import math
from scipy.stats import entropy

# 配置日誌
logger = logging.getLogger(__name__)

class NodeLifecycleManager:
    """節點生命週期管理器
    
    實現智能化的節點重要性評估與記憶機制，防止重要節點被過早清理。
    """
    
    def __init__(self, config=None):
        """初始化節點生命週期管理器
        
        參數:
            config (dict): 配置字典
        """
        self.config = config or {}
        
        # 活躍度追蹤配置
        self.activity_window = self.config.get('activity_window', 5)  # 活躍窗口大小（時間單位）
        self.activity_decay = self.config.get('activity_decay', 0.8)  # 活躍度衰減因子
        self.min_activity_level = self.config.get('min_activity_level', 0.01)  # 最小活躍度閾值
        self.dynamic_activity_threshold = self.config.get('dynamic_activity_threshold', True)  # 是否動態調整閾值
        
        # 重要性評分配置
        self.importance_factors = self.config.get('importance_factors', {
            'degree_centrality': 0.25,        # 度中心性權重
            'betweenness_estimate': 0.20,     # 介數中心性估計權重
            'recent_activity': 0.15,          # 近期活動權重
            'cumulative_activity': 0.10,      # 累積活動權重
            'feature_importance': 0.15,       # 特徵重要性權重
            'neighborhood_activity': 0.10,    # 鄰域活動權重
            'anomaly_score': 0.05             # 異常得分權重
        })
        
        # 重新激活配置
        self.enable_reactivation = self.config.get('enable_reactivation', True)  # 是否啟用重新激活機制
        self.reactivation_threshold = self.config.get('reactivation_threshold', 0.7)  # 重新激活閾值
        self.hibernation_pool_size = self.config.get('hibernation_pool_size', 1000)  # 休眠池大小
        self.min_hibernation_time = self.config.get('min_hibernation_time', 2.0)  # 最小休眠時間（時間單位）
        
        # 頻繁模式挖掘配置
        self.pattern_mining_enabled = self.config.get('pattern_mining_enabled', True)  # 是否啟用模式挖掘
        self.pattern_min_support = self.config.get('pattern_min_support', 3)  # 最小支持度
        self.pattern_max_length = self.config.get('pattern_max_length', 5)  # 最大模式長度
        
        # 初始化內部狀態
        self._initialize_state()
        
        logger.info("初始化節點生命週期管理器")
        logger.info(f"活躍窗口: {self.activity_window}, 衰減因子: {self.activity_decay}")
        logger.info(f"重要性因子: {list(self.importance_factors.keys())}")
        logger.info(f"重新激活機制: {self.enable_reactivation}")
    
    def _initialize_state(self):
        """初始化內部狀態"""
        # 活躍度追蹤
        self.node_activity = {}  # 節點活躍度
        self.last_access_time = {}  # 最後訪問時間
        self.access_history = defaultdict(list)  # 訪問歷史
        self.current_time = 0.0  # 當前時間
        
        # 重要性追蹤
        self.node_importance = {}  # 節點重要性
        self.importance_history = defaultdict(list)  # 重要性歷史
        
        # 鄰域追蹤
        self.node_neighbors = defaultdict(set)  # 節點鄰居
        self.neighbor_activity = defaultdict(float)  # 鄰居活躍度
        
        # 休眠池
        self.hibernation_pool = {}  # 休眠節點池
        self.hibernation_time = {}  # 休眠開始時間
        
        # 頻繁模式
        self.frequent_patterns = []  # 發現的頻繁模式
        self.pattern_support = {}  # 模式支持度
        self.node_pattern_membership = defaultdict(list)  # 節點所屬的模式
        
        # 統計數據
        self.stats = {
            'total_nodes_seen': 0,
            'active_nodes': 0,
            'hibernated_nodes': 0,
            'reactivated_nodes': 0,
            'permanently_removed_nodes': 0,
            'dynamic_threshold_adjustments': 0,
            'current_activity_threshold': self.min_activity_level
        }
    
    def update_time(self, current_time=None):
        """更新當前時間"""
        if current_time is None:
            self.current_time += 1.0
        else:
            self.current_time = float(current_time)
        return self.current_time
    
    def register_activity(self, node_id, timestamp=None, features=None, neighbors=None):
        """註冊節點活動
        
        參數:
            node_id: 節點ID
            timestamp: 活動時間戳，若為None則使用當前時間
            features: 節點特徵，可選
            neighbors: 節點鄰居，可選
        """
        # 更新當前時間
        current_time = self.update_time(timestamp)
        
        # 追蹤新節點
        if node_id not in self.node_activity:
            self.node_activity[node_id] = 0.0
            self.stats['total_nodes_seen'] += 1
        
        # 更新活躍度與訪問記錄
        self.node_activity[node_id] = min(1.0, self.node_activity[node_id] + 0.2)
        self.last_access_time[node_id] = current_time
        
        # 維護固定長度的訪問歷史
        self.access_history[node_id].append(current_time)
        if len(self.access_history[node_id]) > 10:  # 保留最近10次活動
            self.access_history[node_id] = self.access_history[node_id][-10:]
        
        # 更新節點鄰居
        if neighbors is not None:
            self.node_neighbors[node_id] = set(neighbors)
            # 更新鄰居的活躍度
            for neighbor in neighbors:
                self.neighbor_activity[neighbor] = min(1.0, self.neighbor_activity.get(neighbor, 0.0) + 0.05)
        
        # 檢查是否需要從休眠池中重新激活
        self._check_reactivation(node_id)
        
        # 更新統計信息
        self.stats['active_nodes'] = len(self.node_activity)
    
    def update_importance(self, node_id, features=None, graph=None):
        """更新節點重要性
        
        參數:
            node_id: 節點ID
            features: 節點特徵，可選
            graph: 圖結構，可選
            
        返回:
            float: 更新後的重要性值
        """
        # 如果節點不在活躍集中，先檢查休眠池
        if node_id not in self.node_activity:
            if node_id in self.hibernation_pool:
                # 如果在休眠池中且達到最小休眠時間，嘗試重新激活
                if self.current_time - self.hibernation_time[node_id] >= self.min_hibernation_time:
                    self._reactivate_node(node_id)
                else:
                    # 否則保持休眠狀態
                    return self.hibernation_pool[node_id]
            else:
                # 節點不存在於系統中
                return 0.0
        
        # 計算多因素重要性
        importance = 0.0
        importance_factors = {}
        
        # 1. 度中心性 - 如果有圖結構
        if graph is not None and hasattr(graph, 'degree'):
            try:
                degree = graph.degree(node_id)
                max_degree = max(dict(graph.degree()).values()) if graph.number_of_edges() > 0 else 1
                degree_centrality = degree / max_degree if max_degree > 0 else 0
                importance += degree_centrality * self.importance_factors.get('degree_centrality', 0.0)
                importance_factors['degree_centrality'] = degree_centrality
            except:
                pass
        
        # 2. 近期活動頻率
        recent_activity = 0.0
        if node_id in self.access_history and self.access_history[node_id]:
            # 計算最近訪問的頻率
            recent_times = self.access_history[node_id]
            if len(recent_times) > 1:
                # 計算平均訪問間隔
                intervals = np.diff(recent_times)
                avg_interval = np.mean(intervals)
                # 將間隔轉換為頻率的評分
                if avg_interval > 0:
                    recent_activity = min(1.0, 1.0 / avg_interval)
                else:
                    recent_activity = 1.0
            else:
                recent_activity = 0.5  # 只有一次訪問時的默認值
        importance += recent_activity * self.importance_factors.get('recent_activity', 0.0)
        importance_factors['recent_activity'] = recent_activity
        
        # 3. 累積活動得分
        current_activity = self.node_activity.get(node_id, 0.0)
        importance += current_activity * self.importance_factors.get('cumulative_activity', 0.0)
        importance_factors['cumulative_activity'] = current_activity
        
        # 4. 特徵重要性（如果提供了特徵）
        feature_importance = 0.0
        if features is not None:
            # 簡單示例：特徵方差作為重要性指標
            try:
                if isinstance(features, np.ndarray) or isinstance(features, list):
                    feature_vec = np.array(features)
                    feature_variance = np.var(feature_vec)
                    # 標準化方差得分
                    feature_importance = min(1.0, feature_variance / 10.0)
                elif isinstance(features, dict):
                    feature_vec = np.array(list(features.values()))
                    feature_variance = np.var(feature_vec)
                    feature_importance = min(1.0, feature_variance / 10.0)
            except:
                pass
        importance += feature_importance * self.importance_factors.get('feature_importance', 0.0)
        importance_factors['feature_importance'] = feature_importance
        
        # 5. 鄰域活躍度
        neighborhood_activity = self.neighbor_activity.get(node_id, 0.0)
        importance += neighborhood_activity * self.importance_factors.get('neighborhood_activity', 0.0)
        importance_factors['neighborhood_activity'] = neighborhood_activity
        
        # 6. 介數中心性估計 (如果有圖結構)
        betweenness_estimate = 0.0
        if graph is not None and node_id in self.node_neighbors:
            try:
                # 使用局部介數中心性的簡單估計
                neighbors = self.node_neighbors[node_id]
                if len(neighbors) >= 2:
                    # 計算鄰居之間的連接比例
                    connected_pairs = 0
                    total_pairs = 0
                    
                    for i, n1 in enumerate(neighbors):
                        for j, n2 in enumerate(neighbors):
                            if i < j:  # 避免重複計算
                                total_pairs += 1
                                if graph.has_edge(n1, n2):
                                    connected_pairs += 1
                    
                    if total_pairs > 0:
                        # 未連接的比例可作為介數中心性的簡單估計
                        betweenness_estimate = 1.0 - (connected_pairs / total_pairs)
            except:
                pass
        importance += betweenness_estimate * self.importance_factors.get('betweenness_estimate', 0.0)
        importance_factors['betweenness_estimate'] = betweenness_estimate
        
        # 7. 節點是否參與頻繁模式
        pattern_membership = 0.0
        if self.pattern_mining_enabled and node_id in self.node_pattern_membership:
            pattern_count = len(self.node_pattern_membership[node_id])
            pattern_membership = min(1.0, pattern_count / 5.0)  # 標準化：最多5個模式
        importance += pattern_membership * self.importance_factors.get('pattern_membership', 0.0)
        importance_factors['pattern_membership'] = pattern_membership
        
        # 8. 異常得分 - 基於訪問時間間隔的變異性
        anomaly_score = 0.0
        if node_id in self.access_history and len(self.access_history[node_id]) > 2:
            intervals = np.diff(self.access_history[node_id])
            if len(intervals) > 0:
                # 計算變異係數 (CV) 作為時間模式的異常性指標
                cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
                anomaly_score = min(1.0, cv / 2.0)  # 標準化：CV > 2 視為高異常性
        importance += anomaly_score * self.importance_factors.get('anomaly_score', 0.0)
        importance_factors['anomaly_score'] = anomaly_score
        
        # 儲存節點重要性與因素詳情
        self.node_importance[node_id] = importance
        
        # 保存重要性歷史
        self.importance_history[node_id].append((self.current_time, importance))
        if len(self.importance_history[node_id]) > 10:
            self.importance_history[node_id] = self.importance_history[node_id][-10:]
        
        return importance
    
    def decay_activities(self):
        """對所有節點的活躍度應用衰減"""
        nodes_to_hibernate = []
        current_threshold = self._get_activity_threshold()
        
        for node_id in list(self.node_activity.keys()):
            # 應用活躍度衰減
            time_since_last_access = self.current_time - self.last_access_time.get(node_id, 0)
            decay_factor = self.activity_decay ** min(time_since_last_access, self.activity_window)
            self.node_activity[node_id] *= decay_factor
            
            # 檢查是否需要休眠
            if self.node_activity[node_id] < current_threshold:
                nodes_to_hibernate.append(node_id)
        
        # 處理需要休眠的節點
        for node_id in nodes_to_hibernate:
            self._hibernate_node(node_id)
    
    def get_important_nodes(self, top_k=None, threshold=None):
        """獲取重要節點
        
        參數:
            top_k: 返回前K個重要節點，如果為None則使用閾值
            threshold: 重要性閾值，如果為None則返回所有節點
            
        返回:
            dict: {節點ID: 重要性}的字典
        """
        if not self.node_importance:
            return {}
        
        # 按重要性降序排序
        sorted_nodes = sorted(self.node_importance.items(), key=lambda x: x[1], reverse=True)
        
        # 如果有指定前K個
        if top_k is not None:
            sorted_nodes = sorted_nodes[:top_k]
        
        # 如果有指定閾值
        if threshold is not None:
            sorted_nodes = [(node, importance) for node, importance in sorted_nodes if importance >= threshold]
        
        return dict(sorted_nodes)
    
    def _get_activity_threshold(self):
        """獲取當前活躍度閾值，可能是動態調整的"""
        if not self.dynamic_activity_threshold or len(self.node_activity) < 10:
            return self.min_activity_level
        
        # 根據當前節點活躍度分佈動態調整閾值
        activity_values = list(self.node_activity.values())
        
        # 計算活躍度的四分位數
        q1 = np.percentile(activity_values, 25)
        q2 = np.percentile(activity_values, 50)  # 中位數
        
        # 使用較低的四分位數和最小閾值中的較大值
        dynamic_threshold = max(self.min_activity_level, q1 * 0.5)
        
        # 記錄閾值變化
        if abs(dynamic_threshold - self.stats['current_activity_threshold']) > 0.01:
            self.stats['dynamic_threshold_adjustments'] += 1
            self.stats['current_activity_threshold'] = dynamic_threshold
        
        return dynamic_threshold
    
    def _hibernate_node(self, node_id):
        """將節點移入休眠池"""
        if node_id not in self.node_activity:
            return
        
        # 計算該節點的重要性（如果尚未計算）
        if node_id not in self.node_importance:
            self.update_importance(node_id)
        
        # 將節點移至休眠池
        importance = self.node_importance.get(node_id, 0.0)
        self.hibernation_pool[node_id] = importance
        self.hibernation_time[node_id] = self.current_time
        
        # 從活躍集中移除
        del self.node_activity[node_id]
        if node_id in self.node_importance:
            del self.node_importance[node_id]
        
        # 更新統計
        self.stats['active_nodes'] = len(self.node_activity)
        self.stats['hibernated_nodes'] = len(self.hibernation_pool)
        
        # 如果休眠池太大，移除最不重要的節點
        if len(self.hibernation_pool) > self.hibernation_pool_size:
            self._prune_hibernation_pool()
    
    def _reactivate_node(self, node_id):
        """從休眠池中重新激活節點"""
        if not self.enable_reactivation or node_id not in self.hibernation_pool:
            return False
        
        # 恢復節點狀態
        importance = self.hibernation_pool[node_id]
        initial_activity = 0.3  # 重新激活時的初始活躍度
        
        self.node_activity[node_id] = initial_activity
        self.node_importance[node_id] = importance
        self.last_access_time[node_id] = self.current_time
        
        # 從休眠池移除
        del self.hibernation_pool[node_id]
        del self.hibernation_time[node_id]
        
        # 更新統計
        self.stats['active_nodes'] = len(self.node_activity)
        self.stats['hibernated_nodes'] = len(self.hibernation_pool)
        self.stats['reactivated_nodes'] += 1
        
        return True
    
    def _check_reactivation(self, node_id):
        """檢查節點是否應該從休眠池重新激活"""
        return self._reactivate_node(node_id)
    
    def _prune_hibernation_pool(self):
        """清理休眠池，移除最不重要的節點"""
        # 按重要性排序休眠節點
        sorted_nodes = sorted(self.hibernation_pool.items(), key=lambda x: x[1])
        
        # 計算需要移除的數量 - 移除10%或至少1個
        remove_count = max(1, int(len(sorted_nodes) * 0.1))
        nodes_to_remove = [node for node, _ in sorted_nodes[:remove_count]]
        
        # 移除節點
        for node_id in nodes_to_remove:
            del self.hibernation_pool[node_id]
            del self.hibernation_time[node_id]
            self.stats['permanently_removed_nodes'] += 1
    
    def update_frequent_patterns(self, node_sequence):
        """更新頻繁節點訪問模式
        
        參數:
            node_sequence: 節點訪問序列
        """
        if not self.pattern_mining_enabled or len(node_sequence) < 2:
            return
        
        # 使用滑動窗口提取子序列
        for window_size in range(2, min(self.pattern_max_length + 1, len(node_sequence) + 1)):
            for i in range(len(node_sequence) - window_size + 1):
                pattern = tuple(node_sequence[i:i+window_size])
                
                # 更新模式支持度
                self.pattern_support[pattern] = self.pattern_support.get(pattern, 0) + 1
                
                # 如果支持度達到閾值，將其添加到頻繁模式列表
                if self.pattern_support[pattern] >= self.pattern_min_support:
                    if pattern not in self.frequent_patterns:
                        self.frequent_patterns.append(pattern)
                        
                        # 更新節點的模式成員資格
                        for node in pattern:
                            if pattern not in self.node_pattern_membership[node]:
                                self.node_pattern_membership[node].append(pattern)
    
    def get_node_status_report(self):
        """獲取節點狀態報告"""
        return {
            'active_nodes': len(self.node_activity),
            'hibernated_nodes': len(self.hibernation_pool),
            'total_nodes_managed': self.stats['total_nodes_seen'],
            'reactivated_nodes': self.stats['reactivated_nodes'],
            'permanently_removed_nodes': self.stats['permanently_removed_nodes'],
            'dynamic_threshold_adjustments': self.stats['dynamic_threshold_adjustments'],
            'current_activity_threshold': self.stats['current_activity_threshold'],
            'frequent_patterns_found': len(self.frequent_patterns)
        }


class TemporalNodeLifecycleManager(NodeLifecycleManager):
    """時間動態圖的節點生命週期管理器
    
    擴展基本管理器以處理時間屬性和動態圖
    """
    
    def __init__(self, config=None):
        """初始化時間動態圖節點管理器"""
        super().__init__(config)
        
        # 時間特定配置
        self.time_decay_factor = self.config.get('time_decay_factor', 0.95)  # 時間衰減係數
        self.temporal_window_size = self.config.get('temporal_window_size', 10.0)  # 時間窗口大小
        self.recency_boost = self.config.get('recency_boost', True)  # 是否增強最近節點的重要性
        
        # 時間相關狀態
        self.node_timestamps = {}  # 節點關聯的時間戳
        self.node_time_ranges = {}  # 節點活動的時間範圍
        self.time_window_nodes = defaultdict(set)  # 時間窗口中的節點
        
        logger.info("初始化時間動態圖節點生命週期管理器")
    
    def register_temporal_activity(self, node_id, timestamp, features=None, neighbors=None):
        """註冊節點的時間活動
        
        參數:
            node_id: 節點ID
            timestamp: 活動時間戳
            features: 節點特徵，可選
            neighbors: 節點鄰居，可選
        """
        # 更新基本活動
        self.register_activity(node_id, timestamp, features, neighbors)
        
        # 更新時間相關資訊
        self.node_timestamps[node_id] = timestamp
        
        # 更新節點的時間範圍
        if node_id not in self.node_time_ranges:
            self.node_time_ranges[node_id] = [timestamp, timestamp]  # [開始時間, 結束時間]
        else:
            self.node_time_ranges[node_id][1] = timestamp  # 更新結束時間
        
        # 更新當前時間窗口的節點集
        window_start = timestamp - self.temporal_window_size
        current_window = (window_start, timestamp)
        self.time_window_nodes[current_window].add(node_id)
        
        # 清理過期的時間窗口
        self._clean_old_time_windows(window_start)
    
    def update_temporal_importance(self, node_id, timestamp, features=None, graph=None):
        """更新時間相關節點重要性
        
        參數:
            node_id: 節點ID
            timestamp: 當前時間戳
            features: 節點特徵，可選
            graph: 圖結構，可選
            
        返回:
            float: 更新後的重要性值
        """
        # 先計算基本重要性
        base_importance = self.update_importance(node_id, features, graph)
        
        # 時間相關調整因子
        temporal_factor = 1.0
        
        # 考慮時間衰減 - 距離當前時間越遠，重要性越低
        if node_id in self.node_timestamps:
            node_time = self.node_timestamps[node_id]
            time_diff = timestamp - node_time
            
            # 使用指數衰減
            if time_diff > 0:
                temporal_factor = self.time_decay_factor ** min(time_diff, self.temporal_window_size)
        
        # 考慮節點持續時間 - 長期存在的節點可能更重要
        if node_id in self.node_time_ranges:
            start_time, end_time = self.node_time_ranges[node_id]
            duration = end_time - start_time
            
            # 持續時間因子 - 最多提升20%
            duration_factor = min(0.2, duration / self.temporal_window_size)
            temporal_factor += duration_factor
        
        # 最近節點增強 - 為最近訪問的節點增加重要性
        if self.recency_boost and node_id in self.node_timestamps:
            node_time = self.node_timestamps[node_id]
            
            # 如果節點在最近的10%時間窗口內
            if timestamp - node_time < self.temporal_window_size * 0.1:
                temporal_factor += 0.15  # 最多增加15%重要性
        
        # 應用時間因子調整重要性
        adjusted_importance = base_importance * temporal_factor
        
        # 更新節點重要性
        self.node_importance[node_id] = adjusted_importance
        
        return adjusted_importance
    
    def get_nodes_in_time_range(self, start_time, end_time):
        """獲取特定時間範圍內的節點
        
        參數:
            start_time: 開始時間
            end_time: 結束時間
            
        返回:
            set: 時間範圍內的節點集合
        """
        nodes_in_range = set()
        
        # 檢查每個節點的時間範圍
        for node_id, time_range in self.node_time_ranges.items():
            node_start, node_end = time_range
            
            # 檢查節點的時間範圍是否與查詢範圍有重疊
            if not (node_end < start_time or node_start > end_time):
                nodes_in_range.add(node_id)
        
        return nodes_in_range
    
    def _clean_old_time_windows(self, current_window_start):
        """清理舊的時間窗口"""
        windows_to_remove = []
        
        for window in self.time_window_nodes.keys():
            window_start, window_end = window
            
            # 如果窗口完全在當前窗口之前，移除它
            if window_end < current_window_start:
                windows_to_remove.append(window)
        
        # 移除過期窗口
        for window in windows_to_remove:
            del self.time_window_nodes[window]


def create_
