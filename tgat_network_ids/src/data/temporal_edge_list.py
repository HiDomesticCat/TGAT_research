#!/usr/bin/env python
# coding: utf-8 -*-

"""
時間邊列表模組

實現高效的時間邊列表數據結構，用於時序圖的快速窗口查詢和內存優化。
主要功能：
1. 按時間戳索引高效存儲邊
2. 支持時間窗口快速查詢
3. 自動清理過期邊以節省內存
4. 提供時間統計和記憶體使用報告
"""

import bisect
from collections import defaultdict
import logging
import time
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
import gc

# 配置日誌
logger = logging.getLogger(__name__)

class TemporalEdgeList:
    """高效的時間邊列表實現，支援快速時間窗口查詢"""
    
    def __init__(self, clean_interval: int = 1000):
        """
        初始化時間邊列表
        
        參數:
            clean_interval: 多少條邊後嘗試自動清理 (default: 1000)
        """
        # 按時間戳索引的邊列表
        self.time_indexed_edges = defaultdict(list)
        # 時間戳排序列表
        self.timestamps = []
        # 邊計數器
        self.edge_count = 0
        # 清理計數器
        self.clean_counter = 0
        # 清理閾值
        self.clean_interval = clean_interval
        # 創建時間
        self.creation_time = time.time()
        # 最後更新時間
        self.last_update_time = self.creation_time
        # 最後清理時間
        self.last_clean_time = self.creation_time
        
        logger.info(f"初始化時間邊列表: 清理間隔={clean_interval}")
        
    def add_edge(self, src: int, dst: int, timestamp: float, features: Any = None) -> int:
        """
        添加一個時間邊
        
        參數:
            src: 源節點ID
            dst: 目標節點ID
            timestamp: 時間戳
            features: 邊特徵 (可選)
            
        返回:
            當前邊總數
        """
        if timestamp not in self.time_indexed_edges:
            bisect.insort(self.timestamps, timestamp)
        
        self.time_indexed_edges[timestamp].append((src, dst, features))
        self.edge_count += 1
        self.last_update_time = time.time()
        self.clean_counter += 1
        
        # 如果達到清理閾值，嘗試自動清理
        if self.clean_counter >= self.clean_interval:
            self._auto_clean()
            self.clean_counter = 0
        
        return self.edge_count
    
    def add_edges_batch(self, edges: List[Tuple[int, int, float, Any]]) -> int:
        """
        批量添加時間邊
        
        參數:
            edges: 邊列表，每個元素是 (src, dst, timestamp, features)
            
        返回:
            當前邊總數
        """
        # 按時間戳分組
        timestamp_groups = defaultdict(list)
        for src, dst, timestamp, features in edges:
            timestamp_groups[timestamp].append((src, dst, features))
        
        # 添加到時間索引
        for timestamp, edge_list in timestamp_groups.items():
            if timestamp not in self.time_indexed_edges:
                bisect.insort(self.timestamps, timestamp)
            self.time_indexed_edges[timestamp].extend(edge_list)
        
        # 更新計數器
        added_count = len(edges)
        self.edge_count += added_count
        self.last_update_time = time.time()
        self.clean_counter += added_count
        
        # 如果達到清理閾值，嘗試自動清理
        if self.clean_counter >= self.clean_interval:
            self._auto_clean()
            self.clean_counter = 0
        
        return self.edge_count
    
    def get_edges_in_window(self, start_time: float, end_time: float) -> List[Tuple[int, int, float, Any]]:
        """
        獲取時間窗口內的所有邊
        
        參數:
            start_time: 起始時間
            end_time: 結束時間
            
        返回:
            窗口內的所有邊 [(src, dst, timestamp, features), ...]
        """
        # 如果沒有時間戳，返回空列表
        if not self.timestamps:
            return []
        
        # 二分搜索找到時間範圍
        start_idx = bisect.bisect_left(self.timestamps, start_time)
        end_idx = bisect.bisect_right(self.timestamps, end_time)
        
        # 提取該時間範圍內的所有邊
        edges = []
        for i in range(start_idx, end_idx):
            timestamp = self.timestamps[i]
            edges.extend((src, dst, timestamp, features) 
                        for src, dst, features in self.time_indexed_edges[timestamp])
        
        logger.debug(f"找到時間窗口 [{start_time}, {end_time}] 內的 {len(edges)} 條邊")
        return edges
    
    def get_recent_edges(self, time_window: float) -> List[Tuple[int, int, float, Any]]:
        """
        獲取最近的邊
        
        參數:
            time_window: 時間窗口大小
            
        返回:
            最近時間窗口內的所有邊
        """
        if not self.timestamps:
            return []
        
        latest_time = self.timestamps[-1]
        start_time = latest_time - time_window
        
        return self.get_edges_in_window(start_time, latest_time)
    
    def clear_old_edges(self, time_threshold: float) -> int:
        """
        清理舊邊以釋放記憶體
        
        參數:
            time_threshold: 時間閾值，早於此時間的邊將被清理
            
        返回:
            清理的邊數量
        """
        if not self.timestamps:
            return 0
        
        # 找到閾值的位置
        idx = bisect.bisect_left(self.timestamps, time_threshold)
        
        # 如果沒有舊邊，直接返回
        if idx == 0:
            return 0
        
        # 記錄要清理的邊數量
        num_edges_to_clear = sum(len(self.time_indexed_edges[t]) for t in self.timestamps[:idx])
        
        # 清理舊邊
        for t in self.timestamps[:idx]:
            del self.time_indexed_edges[t]
        
        # 更新時間戳列表
        self.timestamps = self.timestamps[idx:]
        
        # 更新邊計數
        self.edge_count -= num_edges_to_clear
        
        # 更新最後清理時間
        self.last_clean_time = time.time()
        
        # 強制垃圾回收
        gc.collect()
        
        logger.info(f"清理 {num_edges_to_clear} 條早於 {time_threshold} 的邊")
        return num_edges_to_clear
    
    def _auto_clean(self, retention_window: float = 3600 * 24) -> int:
        """
        自動清理過期邊
        
        參數:
            retention_window: 保留窗口大小（秒）
            
        返回:
            清理的邊數量
        """
        if not self.timestamps:
            return 0
        
        # 使用最新時間減去保留窗口作為清理閾值
        latest_time = self.timestamps[-1]
        clean_threshold = latest_time - retention_window
        
        # 清理
        return self.clear_old_edges(clean_threshold)
    
    def get_time_span(self) -> Tuple[Optional[float], Optional[float]]:
        """
        獲取時間跨度
        
        返回:
            (最早時間, 最晚時間) 或 (None, None)
        """
        if not self.timestamps:
            return None, None
        
        return self.timestamps[0], self.timestamps[-1]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        獲取邊列表的統計信息
        
        返回:
            統計信息字典
        """
        earliest, latest = self.get_time_span()
        active_edges = sum(len(edges) for edges in self.time_indexed_edges.values())
        
        return {
            "total_edges_added": self.edge_count,
            "active_edges": active_edges,
            "active_timestamps": len(self.timestamps),
            "earliest_timestamp": earliest,
            "latest_timestamp": latest,
            "time_span": latest - earliest if earliest is not None and latest is not None else 0,
            "creation_time": self.creation_time,
            "last_update_time": self.last_update_time,
            "last_clean_time": self.last_clean_time,
            "uptime": time.time() - self.creation_time
        }
    
    def memory_usage(self) -> Dict[str, Union[int, float]]:
        """
        估計記憶體使用量
        
        返回:
            記憶體使用統計字典
        """
        # 時間戳列表的記憶體使用
        timestamps_memory = len(self.timestamps) * 8  # 浮點數大約8字節
        
        # 邊的記憶體使用（粗略估計）
        edges_memory = 0
        for edges in self.time_indexed_edges.values():
            # 每條邊估計為(int, int, object) 約24字節
            edges_memory += len(edges) * 24
        
        # 字典和其他結構的開銷
        overhead = 1024  # 粗略估計
        
        total = timestamps_memory + edges_memory + overhead
        
        return {
            "timestamps_memory_bytes": timestamps_memory,
            "edges_memory_bytes": edges_memory,
            "overhead_bytes": overhead,
            "total_memory_bytes": total,
            "total_memory_mb": total / (1024 * 1024)
        }
    
    def __len__(self) -> int:
        """返回活動邊數量"""
        return sum(len(edges) for edges in self.time_indexed_edges.values())
    
    def __str__(self) -> str:
        """返回描述字符串"""
        stats = self.get_statistics()
        return (f"TemporalEdgeList: {stats['active_edges']} active edges, "
                f"time span: {stats['time_span']:.2f}s, "
                f"{stats['active_timestamps']} timestamps")
