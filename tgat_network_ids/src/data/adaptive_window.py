#!/usr/bin/env python
# coding: utf-8 -*-

"""
自適應時間窗口機制

為時間圖分析提供多尺度、自適應的時間窗口選擇機制。
支援不同類型攻擊模式的檢測，包括：
1. 快速爆發型攻擊 (毫秒-秒級別)
2. 持續性滲透 (分鐘-小時級別)
3. 低慢攻擊 (小時-天級別)
4. 高級持續威脅APT (天-週-月級別)
"""

import numpy as np
import pandas as pd
import networkx as nx
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.cluster import DBSCAN
from tqdm import tqdm

# 配置日誌
logger = logging.getLogger(__name__)

class AdaptiveWindowManager:
    """自適應時間窗口管理器
    
    根據流量特性、攻擊模式和時間分佈動態調整時間窗口大小，支援多尺度分析。
    """
    
    def __init__(self, config=None):
        """初始化自適應窗口管理器
        
        參數:
            config (dict): 配置字典，包含窗口相關設定
        """
        self.config = config or {}
        
        # 默認窗口層級與大小設定 (秒為單位)
        self.window_scales = self.config.get('window_scales', {
            'micro': 1,           # 1秒 (快速爆發型事件)
            'small': 60,          # 1分鐘 (短時連續事件)
            'medium': 300,        # 5分鐘 (標準時間範圍)
            'large': 3600,        # 1小時 (長時間模式)
            'macro': 86400,       # 1天 (低慢攻擊/潛伏期攻擊)
            'extended': 604800    # 1週 (高級持續威脅APT)
        })
        
        # 默認啟用的窗口層級
        self.enabled_scales = self.config.get('enabled_scales', ['micro', 'medium', 'large'])
        
        # 動態窗口自適應設置
        self.enable_auto_adjust = self.config.get('enable_auto_adjust', True)
        self.min_events_threshold = self.config.get('min_events_threshold', 5)
        self.activity_threshold = self.config.get('activity_threshold', 0.01)  # 最低活動閾值
        self.max_window_count = self.config.get('max_window_count', 100)  # 每個尺度最大窗口數
        
        # 特殊模式檢測設置
        self.detect_bursts = self.config.get('detect_bursts', True)  # 檢測突發活動
        self.detect_periodic = self.config.get('detect_periodic', True)  # 檢測週期性活動
        self.detect_low_slow = self.config.get('detect_low_slow', True)  # 檢測低慢攻擊
        
        # 監控與統計
        self.window_stats = {}
        self.window_cache = {}
        
        logger.info(f"初始化自適應窗口管理器，啟用窗口尺度: {self.enabled_scales}")
        
    def select_optimal_windows(self, timestamps, features=None, edge_data=None):
        """選擇最佳的時間窗口組合
        
        根據時間分佈特性和可選特徵/邊緣數據自動選擇最佳窗口尺度組合
        
        參數:
            timestamps (array-like): 時間戳數組或列表
            features (DataFrame, optional): 額外特徵數據
            edge_data (list, optional): 邊關係數據
            
        返回:
            dict: 最佳窗口組合，按尺度分組
        """
        if len(timestamps) < self.min_events_threshold:
            logger.warning(f"事件數量過少 ({len(timestamps)})，使用默認中等窗口")
            return {'medium': [self.window_scales['medium']]}
        
        # 確保時間戳為數值型時間序列
        times = self._normalize_timestamps(timestamps)
        
        # 計算時間間隔統計
        time_diffs = np.diff(np.sort(times))
        
        if len(time_diffs) < 2:
            logger.warning("時間點不足，使用默認中等窗口尺度")
            scale_name = 'medium'
            return {scale_name: [self.window_scales[scale_name]]}
        
        # 計算基本統計指標
        min_diff = np.min(time_diffs)
        max_diff = np.max(time_diffs)
        median_diff = np.median(time_diffs)
        q1_diff = np.percentile(time_diffs, 25)
        q3_diff = np.percentile(time_diffs, 75)
        iqr = q3_diff - q1_diff
        
        logger.info(f"時間差統計: 最小={min_diff:.2f}s, 中位數={median_diff:.2f}s, 最大={max_diff:.2f}s")
        
        # 檢測時間尺度特徵
        is_bursty = self._detect_burst_pattern(time_diffs)
        is_periodic = self._detect_periodic_pattern(times)
        is_low_slow = self._detect_low_slow_pattern(time_diffs, features)
        
        if is_bursty:
            logger.info("檢測到突發性活動模式")
        if is_periodic:
            logger.info("檢測到週期性活動模式")
        if is_low_slow:
            logger.info("檢測到低慢攻擊模式")
            
        # 根據時間特性選擇窗口尺度
        selected_scales = {}
        
        # 選擇微觀(micro)窗口 - 基於最小間隔或突發特性
        if 'micro' in self.enabled_scales and (min_diff < 2.0 or is_bursty):
            base_window = max(0.5, min_diff)  # 確保至少0.5秒
            micro_windows = [base_window, base_window * 2, base_window * 5]
            selected_scales['micro'] = micro_windows
            
        # 選擇小型(small)窗口 - 基於第一四分位數或突發特性
        if 'small' in self.enabled_scales and (q1_diff < 30 or is_bursty):
            small_window = max(2.0, q1_diff)
            selected_scales['small'] = [small_window, small_window * 2, small_window * 5]
            
        # 選擇中型(medium)窗口 - 基於中位數或週期特性
        if 'medium' in self.enabled_scales:
            medium_window = max(10.0, median_diff)
            medium_windows = [medium_window]
            if is_periodic:
                # 添加檢測到的週期作為窗口大小
                period = self._estimate_period(times)
                if period > 0:
                    medium_windows.append(period)
            selected_scales['medium'] = medium_windows
            
        # 選擇大型(large)窗口 - 基於第三四分位數或低慢特性
        if 'large' in self.enabled_scales and (q3_diff > 60 or is_low_slow):
            large_window = max(60.0, q3_diff * 2)
            selected_scales['large'] = [large_window, large_window * 2]
            
        # 選擇宏觀(macro)窗口 - 基於最大間隔或低慢特性
        if 'macro' in self.enabled_scales and (max_diff > 3600 or is_low_slow):
            macro_window = max(3600.0, max_diff / 2)
            selected_scales['macro'] = [macro_window]
            
        # 確保至少有一個窗口尺度被選中
        if not selected_scales:
            scale_name = 'medium'
            logger.info(f"無法確定最佳窗口尺度，使用默認 '{scale_name}' 尺度")
            selected_scales[scale_name] = [self.window_scales[scale_name]]
            
        # 記錄選擇結果
        scale_info = []
        for scale, windows in selected_scales.items():
            scale_info.append(f"{scale}:[{', '.join(f'{w:.1f}s' for w in windows)}]")
        logger.info(f"選擇的窗口尺度: {'; '.join(scale_info)}")
            
        return selected_scales

    def generate_adaptive_windows(self, timestamps, selected_scales=None):
        """生成自適應時間窗口序列
        
        參數:
            timestamps (array-like): 時間戳數組
            selected_scales (dict, optional): 預選窗口尺度，若未提供則自動選擇
            
        返回:
            list: 生成的時間窗口列表，每個窗口為(開始時間, 結束時間, 尺度名稱, 窗口大小)
        """
        if len(timestamps) < self.min_events_threshold:
            logger.warning(f"事件數量過少 ({len(timestamps)})")
            return []
            
        # 確保時間戳為數值型
        times = self._normalize_timestamps(timestamps)
        
        # 如果未提供尺度，自動選擇
        if selected_scales is None:
            selected_scales = self.select_optimal_windows(times)
        
        # 計算時間範圍
        start_time = np.min(times)
        end_time = np.max(times)
        total_duration = end_time - start_time
        
        windows = []
        
        # 對每個選擇的尺度生成窗口
        for scale, window_sizes in selected_scales.items():
            for window_size in window_sizes:
                # 計算此尺度的窗口數量，但限制最大窗口數
                if total_duration <= 0:
                    n_windows = 1
                else:
                    n_windows = min(int(np.ceil(total_duration / window_size)), self.max_window_count)
                
                # 如果窗口數太多，調整窗口大小
                if n_windows > self.max_window_count:
                    adjusted_size = total_duration / self.max_window_count
                    logger.info(f"窗口 '{scale}' 數量({n_windows})超過限制，調整大小: {window_size:.1f}s → {adjusted_size:.1f}s")
                    window_size = adjusted_size
                    n_windows = self.max_window_count
                
                # 生成均勻分佈的窗口
                for i in range(n_windows):
                    win_start = start_time + i * window_size
                    win_end = win_start + window_size
                    
                    # 確保最後一個窗口涵蓋結束時間
                    if i == n_windows - 1:
                        win_end = max(win_end, end_time)
                        
                    windows.append((win_start, win_end, scale, window_size))
                
        # 按開始時間排序
        windows.sort(key=lambda x: x[0])
        
        logger.info(f"生成了 {len(windows)} 個自適應時間窗口，跨越 {scale_info} 尺度")
        return windows

    def create_windows_for_pattern(self, timestamps, pattern_type):
        """為特定模式類型創建專用窗口
        
        參數:
            timestamps (array-like): 時間戳數組
            pattern_type (str): 模式類型，如 'burst', 'periodic', 'low_slow'
            
        返回:
            list: 生成的時間窗口列表
        """
        times = self._normalize_timestamps(timestamps)
        
        if pattern_type == 'burst':
            # 對突發事件使用小窗口、高密度
            windows = self._create_burst_windows(times)
        elif pattern_type == 'periodic':
            # 對週期性事件使用與週期匹配的窗口
            windows = self._create_periodic_windows(times)
        elif pattern_type == 'low_slow':
            # 對低慢攻擊使用大窗口、低密度
            windows = self._create_low_slow_windows(times)
        else:
            logger.warning(f"未知的模式類型: {pattern_type}，使用默認窗口")
            scales = self.select_optimal_windows(times)
            windows = self.generate_adaptive_windows(times, scales)
            
        return windows

    def _normalize_timestamps(self, timestamps):
        """標準化時間戳為數值型秒數據"""
        if isinstance(timestamps[0], (int, float)):
            # 已經是數值型，假設為秒
            return np.array(timestamps, dtype=float)
        elif isinstance(timestamps[0], str):
            # 字符串格式時間戳，嘗試解析
            try:
                dt_objects = pd.to_datetime(timestamps)
                # 轉換為Unix時間戳 (秒)
                return dt_objects.astype('int64') / 1e9
            except:
                logger.error("無法轉換字符串時間戳")
                raise ValueError("無法解析時間戳格式")
        elif isinstance(timestamps[0], (datetime, pd.Timestamp)):
            # datetime對象，轉換為Unix時間戳
            if isinstance(timestamps, pd.Series):
                return timestamps.astype('int64') / 1e9
            else:
                return np.array([(t - datetime(1970, 1, 1)).total_seconds() 
                                for t in timestamps])
        else:
            logger.error(f"不支持的時間戳類型: {type(timestamps[0])}")
            raise ValueError("不支持的時間戳格式")

    def _detect_burst_pattern(self, time_diffs):
        """檢測突發活動模式"""
        if not self.detect_bursts or len(time_diffs) < 10:
            return False
            
        # 計算時間差的變異係數 (CV)
        cv = np.std(time_diffs) / np.mean(time_diffs)
        
        # 突發活動通常有高變異係數
        is_bursty = cv > 2.0
        
        # 檢查是否有明顯的集群
        if len(time_diffs) >= 50:
            try:
                # 使用DBSCAN檢測時間集群
                time_diffs_2d = time_diffs.reshape(-1, 1)
                clustering = DBSCAN(eps=np.median(time_diffs)/2, min_samples=5).fit(time_diffs_2d)
                labels = clustering.labels_
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
                # 如果有多個明顯集群，可能是突發模式
                is_bursty = is_bursty or n_clusters >= 2
            except Exception as e:
                logger.warning(f"DBSCAN聚類分析失敗: {str(e)}")
        
        return is_bursty

    def _detect_periodic_pattern(self, times):
        """檢測週期性活動模式"""
        if not self.detect_periodic or len(times) < 20:
            return False
            
        try:
            # 使用FFT或自相關分析尋找週期性
            sorted_times = np.sort(times)
            time_diffs = np.diff(sorted_times)
            
            # 計算自相關
            n = len(time_diffs)
            max_lag = min(n//3, 100)  # 最大延遲為數據長度的1/3或100
            
            autocorr = np.zeros(max_lag)
            mean = np.mean(time_diffs)
            var = np.var(time_diffs)
            
            for lag in range(1, max_lag + 1):
                # 計算自相關係數
                autocorr[lag-1] = np.sum((time_diffs[:n-lag] - mean) * 
                                          (time_diffs[lag:] - mean)) / ((n - lag) * var)
            
            # 尋找自相關峰值
            peaks = []
            for i in range(1, len(autocorr) - 1):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.2:
                    peaks.append(i)
                    
            # 如果有明顯的自相關峰值，表明有週期性
            return len(peaks) > 0
        except Exception as e:
            logger.warning(f"週期性檢測失敗: {str(e)}")
            return False

    def _detect_low_slow_pattern(self, time_diffs, features=None):
        """檢測低慢攻擊模式"""
        if not self.detect_low_slow:
            return False
            
        # 低慢攻擊通常表現為低頻率但長期持續的模式
        if len(time_diffs) < 10:
            return False
            
        # 基本檢測: 查看是否有異常大的時間間隔
        q3 = np.percentile(time_diffs, 75)
        max_diff = np.max(time_diffs)
        
        is_low_slow = max_diff > 10 * q3
        
        # 進階檢測: 如果有額外特徵，查看活動模式
        if features is not None and isinstance(features, pd.DataFrame):
            try:
                # 檢查是否有重複IP地址但時間間隔長
                ip_cols = [col for col in features.columns if 'ip' in col.lower()]
                
                if ip_cols:
                    ip_col = ip_cols[0]  # 使用第一個IP列
                    
                    # 計算每個IP的時間分佈
                    ip_times = defaultdict(list)
                    
                    for idx, ip in enumerate(features[ip_col]):
                        if idx < len(time_diffs):
                            ip_times[ip].append(time_diffs[idx])
                    
                    # 檢查是否有IP的活動模式符合低慢特徵
                    for ip, ip_diffs in ip_times.items():
                        if len(ip_diffs) < 3:
                            continue
                            
                        ip_diffs = np.array(ip_diffs)
                        max_ip_diff = np.max(ip_diffs)
                        mean_ip_diff = np.mean(ip_diffs)
                        
                        # 低慢特徵: 大的最大間隔和相對較小的平均間隔
                        if max_ip_diff > 5 * mean_ip_diff and max_ip_diff > 300:  # 5分鐘以上
                            is_low_slow = True
                            break
                            
            except Exception as e:
                logger.warning(f"低慢攻擊特徵分析失敗: {str(e)}")
        
        return is_low_slow

    def _estimate_period(self, times):
        """估計時間序列的週期"""
        try:
            sorted_times = np.sort(times)
            time_diffs = np.diff(sorted_times)
            
            # 計算自相關
            n = len(time_diffs)
            max_lag = min(n//3, 100)
            
            autocorr = np.zeros(max_lag)
            mean = np.mean(time_diffs)
            var = np.var(time_diffs)
            
            for lag in range(1, max_lag + 1):
                autocorr[lag-1] = np.sum((time_diffs[:n-lag] - mean) * 
                                          (time_diffs[lag:] - mean)) / ((n - lag) * var)
            
            # 尋找第一個顯著峰值
            for i in range(1, len(autocorr) - 1):
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.2:
                    # 返回峰值對應的週期
                    return float(np.median(time_diffs) * (i + 1))
                    
            return 0  # 未發現明顯週期
        except Exception as e:
            logger.warning(f"週期估計失敗: {str(e)}")
            return 0

    def _create_burst_windows(self, times):
        """為突發事件創建專用窗口"""
        start_time = np.min(times)
        end_time = np.max(times)
        
        # 對數據進行聚類以識別突發區域
        times_2d = times.reshape(-1, 1)
        
        # 自適應eps參數
        time_range = end_time - start_time
        eps = time_range * 0.01  # 默認使用總時間範圍的1%
        
        try:
            # DBSCAN聚類
            clustering = DBSCAN(eps=eps, min_samples=3).fit(times_2d)
            labels = clustering.labels_
            
            # 為每個非噪聲聚類創建窗口
            windows = []
            unique_clusters = set(labels) - {-1}  # 排除噪聲點
            
            for cluster_id in unique_clusters:
                # 獲取該聚類的時間點
                cluster_times = times[labels == cluster_id]
                
                if len(cluster_times) < 3:
                    continue
                    
                # 計算聚類的開始和結束時間
                c_start = np.min(cluster_times)
                c_end = np.max(cluster_times)
                c_duration = c_end - c_start
                
                # 確保窗口至少有最小大小
                min_window = 1.0  # 1秒
                if c_duration < min_window:
                    # 擴展窗口
                    padding = (min_window - c_duration) / 2
                    c_start -= padding
                    c_end += padding
                    
                # 添加聚類窗口
                windows.append((c_start, c_end, 'burst', c_duration))
                
                # 添加擴展窗口以捕獲上下文
                extended_start = max(start_time, c_start - c_duration)
                extended_end = min(end_time, c_end + c_duration)
                windows.append((extended_start, extended_end, 'burst_context', c_duration * 2))
                
            return windows
        except Exception as e:
            logger.warning(f"突發窗口創建失敗: {str(e)}")
            
            # 回退到簡單窗口
            duration = end_time - start_time
            window_size = max(1.0, duration / 10)  # 將總持續時間分為最多10個窗口
            
            windows = []
            for i in range(10):
                win_start = start_time + i * window_size
                win_end = win_start + window_size
                
                # 確保不超過數據範圍
                if win_start >= end_time:
                    break
                win_end = min(win_end, end_time)
                
                windows.append((win_start, win_end, 'burst', window_size))
                
            return windows

    def _create_periodic_windows(self, times):
        """為週期性事件創建專用窗口"""
        # 估計週期
        period = self._estimate_period(times)
        
        if period <= 0:
            # 無法識別週期，使用默認窗口
            logger.info("無法識別明確週期，將使用基本窗口")
            scales = self.select_optimal_windows(times)
            return self.generate_adaptive_windows(times, scales)
            
        start_time = np.min(times)
        end_time = np.max(times)
        duration = end_time - start_time
        
        windows = []
        
        # 創建與週期匹配的窗口
        num_windows = int(np.ceil(duration / period))
        
        for i in range(num_windows):
            win_start = start_time + i * period
            win_end = win_start + period
            
            # 確保窗口不超出數據範圍
            if win_start >= end_time:
                break
            win_end = min(win_end, end_time)
            
            windows.append((win_start, win_end, 'periodic', period))
            
        # 增加部分重疊窗口以捕捉跨越週期邊界的模式
        for i in range(num_windows - 1):
            win_start = start_time + (i + 0.5) * period  # 從半週期開始
            win_end = win_start + period
            
            if win_start >= end_time:
                break
            win_end = min(win_end, end_time)
            
            windows.append((win_start, win_end, 'periodic_overlap', period))
            
        return windows

    def _create_low_slow_windows(self, times):
        """為低慢攻擊創建專用窗口"""
        start_time = np.min(times)
        end_time = np.max(times)
        duration = end_time - start_time
        
        # 低慢攻擊通常持續較長時間，使用較大窗口
        windows = []
        
        # 基本大窗口 - 捕捉整體模式
        large_window_size = max(3600, duration / 5)  # 至少1小時
        
        for i in range(5):  # 最多5個大窗口
            win_start = start_time + i * large_window_size
            win_end = win_start + large_window_size
            
            if win_start >= end_time:
                break
            win_end = min(win_end, end_time)
            
            windows.append((win_start, win_end, 'low_slow_large', large_window_size))
            
        # 滑動窗口 - 更細粒度地捕捉模式變化
        medium_window_size = large_window_size / 3
        step_size = medium_window_size / 2  # 50%重疊
        
        current_start = start_time
        while current_start < end_time:
            win_end = min(current_start + medium_window_size, end_time)
            windows.append((current_start, win_end, 'low_slow_medium', medium_window_size))
            current_start += step_size
            
        return windows
