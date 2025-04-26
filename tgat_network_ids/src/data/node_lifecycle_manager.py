#!/usr/bin/env python
# coding: utf-8 -*-

"""
實現節點生命週期管理功能

該模組用於跟踪圖中節點的活躍時間，實現高效的記憶體管理和節點活躍性檢查。
主要功能包括:
1. 追踪節點的生命週期
2. 管理節點在不同時間窗口下的活躍狀態
3. 實現高效的節點清理機制
4. 提供時間窗口查詢功能
"""

import numpy as np
import pandas as pd
import logging
from collections import defaultdict
from datetime import datetime
import bisect
import heapq

logger = logging.getLogger(__name__)

class NodeLifecycleManager:
    """
    節點生命週期管理器 - 高效記憶體和活躍性管理
    
    該類負責追踪和管理動態圖中節點的活躍時間，
    並提供高效的時間查詢和節點清理功能。
    """
    
    def __init__(self, max_window_size=3600, cleanup_frequency=1000):
        """
        初始化節點生命週期管理器
        
        參數:
            max_window_size (int): 默認最大窗口大小 (秒)
            cleanup_frequency (int): 清理頻率，每處理多少節點執行一次清理
        """
        # 基本設定
        self.max_window_size = max_window_size
        self.cleanup_frequency = cleanup_frequency
        
        # 跟踪節點出現與消失時間
        self.node_first_seen = {}  # 節點ID -> 首次出現時間
        self.node_last_seen = {}   # 節點ID -> 最後出現時間
        self.node_activity_count = {}  # 節點ID -> 活動次數
        
        # 時間窗口相關
        self.time_window_nodes = defaultdict(set)  # 時間窗口 -> 節點集合
        self.window_size = max_window_size  # 當前使用的時間窗口大小
        
        # 優化資源使用
        self.inactive_nodes = set()  # 不再活躍的節點集合
        self.operation_count = 0     # 操作計數器，用於觸發清理
        
        # 時間區間索引
        self.time_points = []  # 所有時間點的有序列表
        
        logger.info(f"初始化節點生命週期管理器: 窗口大小={max_window_size}, 清理頻率={cleanup_frequency}")
    
    def add_node(self, node_id, time):
        """
        添加節點及其出現時間
        
        參數:
            node_id: 節點ID
            time: 節點出現時間
        """
        # 更新節點出現記錄
        if node_id in self.node_first_seen:
            self.node_last_seen[node_id] = max(self.node_last_seen[node_id], time)
            self.node_activity_count[node_id] = self.node_activity_count.get(node_id, 0) + 1
        else:
            self.node_first_seen[node_id] = time
            self.node_last_seen[node_id] = time
            self.node_activity_count[node_id] = 1
        
        # 從非活躍節點集合中移除（如果存在）
        if node_id in self.inactive_nodes:
            self.inactive_nodes.remove(node_id)
        
        # 將節點添加到時間窗口
        window_key = int(time // self.window_size)
        self.time_window_nodes[window_key].add(node_id)
        
        # 更新時間點索引
        if time not in self.time_points:
            bisect.insort(self.time_points, time)
        
        # 增加操作計數，可能觸發清理
        self.operation_count += 1
        if self.operation_count >= self.cleanup_frequency:
            self._perform_cleanup()
            self.operation_count = 0
    
    def add_nodes(self, node_ids, times):
        """
        批量添加節點及其出現時間
        
        參數:
            node_ids: 節點ID列表
            times: 對應的時間列表
        """
        if len(node_ids) != len(times):
            raise ValueError("節點ID列表和時間列表的長度必須相同")
            
        for node_id, time in zip(node_ids, times):
            self.add_node(node_id, time)
    
    def remove_node(self, node_id):
        """
        從管理器中移除節點
        
        參數:
            node_id: 要移除的節點ID
        """
        # 從所有記錄中移除節點
        if node_id in self.node_first_seen:
            del self.node_first_seen[node_id]
        
        if node_id in self.node_last_seen:
            del self.node_last_seen[node_id]
            
        if node_id in self.node_activity_count:
            del self.node_activity_count[node_id]
        
        # 從時間窗口中移除
        for nodes in self.time_window_nodes.values():
            if node_id in nodes:
                nodes.remove(node_id)
        
        # 添加到非活躍集合
        self.inactive_nodes.add(node_id)
    
    def remove_nodes(self, node_ids):
        """
        批量移除節點
        
        參數:
            node_ids: 要移除的節點ID列表
        """
        for node_id in node_ids:
            self.remove_node(node_id)
    
    def get_nodes_in_window(self, window_start, window_end):
        """
        獲取指定時間窗口內的所有節點
        
        參數:
            window_start: 窗口開始時間
            window_end: 窗口結束時間
            
        返回:
            set: 該時間窗口內活躍的節點集合
        """
        # 將時間範圍轉換為窗口鍵
        start_window = int(window_start // self.window_size)
        end_window = int(window_end // self.window_size)
        
        # 收集所有窗口中的節點
        result_nodes = set()
        for window in range(start_window, end_window + 1):
            if window in self.time_window_nodes:
                result_nodes.update(self.time_window_nodes[window])
        
        # 過濾節點：只保留在指定時間範圍內活躍的節點
        filtered_nodes = {
            node_id for node_id in result_nodes
            if (node_id in self.node_first_seen and 
                window_start <= self.node_last_seen.get(node_id, 0) and 
                self.node_first_seen.get(node_id, float('inf')) <= window_end)
        }
        
        return filtered_nodes
    
    def get_active_nodes(self, current_time, lookback_window=None):
        """
        獲取當前活躍的節點
        
        參數:
            current_time: 當前時間點
            lookback_window: 回看窗口大小，默認使用max_window_size
            
        返回:
            set: 活躍節點集合
        """
        if lookback_window is None:
            lookback_window = self.max_window_size
        
        window_start = current_time - lookback_window
        return self.get_nodes_in_window(window_start, current_time)
    
    def get_node_first_time(self, node_id):
        """獲取節點首次出現時間"""
        return self.node_first_seen.get(node_id, None)
    
    def get_node_last_time(self, node_id):
        """獲取節點最後出現時間"""
        return self.node_last_seen.get(node_id, None)
    
    def get_node_activity_span(self, node_id):
        """獲取節點活躍持續時間"""
        if node_id in self.node_first_seen and node_id in self.node_last_seen:
            return self.node_last_seen[node_id] - self.node_first_seen[node_id]
        return None
    
    def get_node_activity_count(self, node_id):
        """獲取節點活動次數"""
        return self.node_activity_count.get(node_id, 0)
    
    def is_node_active(self, node_id, current_time, lookback_window=None):
        """
        檢查節點是否活躍
        
        參數:
            node_id: 節點ID
            current_time: 當前時間
            lookback_window: 回看窗口大小
            
        返回:
            bool: 節點是否活躍
        """
        if lookback_window is None:
            lookback_window = self.max_window_size
            
        if node_id not in self.node_last_seen:
            return False
            
        last_seen = self.node_last_seen[node_id]
        return (current_time - last_seen) <= lookback_window
    
    def get_all_nodes(self):
        """獲取所有曾經出現的節點"""
        return set(self.node_first_seen.keys())
    
    def get_currently_tracked_nodes(self):
        """獲取當前正在追踪的節點（排除已標記為非活躍的）"""
        return set(self.node_first_seen.keys()) - self.inactive_nodes
    
    def get_all_time_points(self):
        """獲取所有已記錄的時間點"""
        return self.time_points.copy()
    
    def adjust_window_size(self, new_size):
        """
        調整時間窗口大小
        
        參數:
            new_size: 新的窗口大小
        """
        if new_size <= 0:
            raise ValueError("窗口大小必須大於零")
            
        old_size = self.window_size
        self.window_size = new_size
        
        # 如果窗口大小變化很大，需要重新組織時間窗口
        if abs(new_size - old_size) / old_size > 0.5:
            self._reorganize_time_windows()
    
    def _reorganize_time_windows(self):
        """重新組織時間窗口"""
        # 保存所有節點和它們的時間
        nodes_with_times = []
        for node_id in self.get_all_nodes():
            if node_id in self.node_first_seen:
                nodes_with_times.append((node_id, self.node_first_seen[node_id]))
        
        # 清除現有窗口
        self.time_window_nodes.clear()
        
        # 重新分配節點到窗口
        for node_id, time in nodes_with_times:
            window_key = int(time // self.window_size)
            self.time_window_nodes[window_key].add(node_id)
    
    def _perform_cleanup(self):
        """執行清理操作，移除不再需要的窗口和節點"""
        # 查找所有窗口的最大時間
        if not self.time_points:
            return
            
        latest_time = max(self.time_points)
        earliest_relevant_time = latest_time - self.max_window_size
        
        # 找出可以安全移除的窗口
        earliest_relevant_window = int(earliest_relevant_time // self.window_size)
        windows_to_remove = [
            window for window in self.time_window_nodes.keys()
            if window < earliest_relevant_window
        ]
        
        # 移除過時的窗口
        for window in windows_to_remove:
            del self.time_window_nodes[window]
    
    def get_node_statistics(self):
        """獲取節點統計信息"""
        stats = {
            'total_nodes': len(self.node_first_seen),
            'active_nodes': len(self.get_all_nodes()) - len(self.inactive_nodes),
            'inactive_nodes': len(self.inactive_nodes),
            'time_windows': len(self.time_window_nodes),
            'time_range': (min(self.time_points) if self.time_points else 0,
                          max(self.time_points) if self.time_points else 0)
        }
        return stats
    
    def __repr__(self):
        stats = self.get_node_statistics()
        return (f"NodeLifecycleManager(total_nodes={stats['total_nodes']}, "
                f"active_nodes={stats['active_nodes']}, "
                f"time_windows={stats['time_windows']})")

def create_node_lifecycle_manager(max_window_size=3600, cleanup_frequency=1000):
    """
    建立並返回一個節點生命週期管理器
    
    參數:
        max_window_size: 最大時間窗口大小（秒）
        cleanup_frequency: 清理頻率
        
    返回:
        NodeLifecycleManager: 初始化好的管理器實例
    """
    manager = NodeLifecycleManager(max_window_size, cleanup_frequency)
    return manager
