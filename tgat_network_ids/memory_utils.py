#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
記憶體優化工具模組 (Enhanced Version)

此模組提供全面的記憶體優化相關工具函數，包括：
1. 記憶體使用監控與分析
2. 記憶體映射文件操作
3. 主動記憶體管理與釋放
4. GPU 記憶體優化與控制
5. 記憶體使用報告生成與視覺化
6. 自適應記憶體管理策略
7. 記憶體洩漏檢測
8. 記憶體使用優化建議
"""

import os
import gc
import time
import psutil
import numpy as np
import pandas as pd
import torch
import logging
import weakref
import json
import tracemalloc
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from functools import wraps
from contextlib import contextmanager
from typing import Dict, List, Tuple, Union, Optional, Callable, Any

# 設置日誌
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局變量
_MEMORY_MONITOR_INSTANCE = None  # 單例模式的記憶體監控器實例

class MemoryMonitor:
    """
    記憶體使用監控類
    
    提供實時記憶體使用監控、報告生成和視覺化功能。
    支持 CPU 和 GPU 記憶體監控，可設置監控間隔和報告格式。
    """
    
    def __init__(self, interval: int = 10, report_dir: str = './memory_reports', 
                 enable_gpu: bool = True, detailed_monitoring: bool = False,
                 alert_threshold: float = 90.0):
        """
        初始化記憶體監控器
        
        參數:
            interval (int): 監控間隔（秒）
            report_dir (str): 報告保存目錄
            enable_gpu (bool): 是否監控 GPU 記憶體
            detailed_monitoring (bool): 是否啟用詳細監控（包括進程和系統級別）
            alert_threshold (float): 記憶體使用率警告閾值（百分比）
        """
        self.interval = interval
        self.report_dir = report_dir
        self.enable_gpu = enable_gpu and torch.cuda.is_available()
        self.detailed_monitoring = detailed_monitoring
        self.alert_threshold = alert_threshold
        self.monitoring = False
        self.memory_usage = []
        self.gpu_memory_usage = []
        self.system_memory_usage = []
        self.timestamps = []
        self.peak_memory = 0
        self.peak_gpu_memory = 0
        self.alerts = []
        
        # 創建報告目錄
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        
        # 初始化 tracemalloc 用於詳細監控
        if detailed_monitoring:
            tracemalloc.start()
    
    def start(self):
        """
        開始記憶體監控
        
        啟動背景線程進行定期記憶體使用監控，並記錄數據用於後續分析。
        """
        if self.monitoring:
            logger.warning("記憶體監控已在運行中")
            return
        
        self.monitoring = True
        self.memory_usage = []
        self.gpu_memory_usage = []
        self.system_memory_usage = []
        self.timestamps = []
        self.peak_memory = 0
        self.peak_gpu_memory = 0
        self.alerts = []
        
        logger.info(f"開始記憶體監控，間隔: {self.interval} 秒")
        
        # 在背景執行監控
        import threading
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop(self):
        """
        停止記憶體監控
        
        停止監控線程，並生成最終報告和視覺化。
        """
        if not self.monitoring:
            logger.warning("記憶體監控未運行")
            return
        
        self.monitoring = False
        logger.info("停止記憶體監控")
        
        # 等待監控線程結束
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2)
        
        # 生成報告
        self.generate_report()
        
        # 如果啟用了詳細監控，停止 tracemalloc
        if self.detailed_monitoring:
            tracemalloc.stop()
    
    def _monitor_loop(self):
        """
        監控循環
        
        定期收集記憶體使用數據，並在超過閾值時發出警告。
        """
        while self.monitoring:
            # 獲取當前記憶體使用情況
            cpu_usage = self._get_cpu_memory_usage()
            system_usage = self._get_system_memory_usage()
            
            self.memory_usage.append(cpu_usage)
            self.system_memory_usage.append(system_usage)
            self.timestamps.append(datetime.now())
            
            # 更新峰值記憶體使用
            self.peak_memory = max(self.peak_memory, cpu_usage)
            
            # 檢查是否超過警告閾值
            if system_usage['percent'] > self.alert_threshold:
                alert_msg = f"記憶體使用率警告: {system_usage['percent']:.2f}% > {self.alert_threshold}%"
                logger.warning(alert_msg)
                self.alerts.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'message': alert_msg,
                    'memory_usage': cpu_usage,
                    'system_percent': system_usage['percent']
                })
                
                # 主動清理記憶體
                clean_memory()
            
            # GPU 記憶體監控
            if self.enable_gpu:
                gpu_usage = self._get_gpu_memory_usage()
                self.gpu_memory_usage.append(gpu_usage)
                self.peak_gpu_memory = max(self.peak_gpu_memory, gpu_usage)
            
            # 記錄日誌
            if len(self.memory_usage) % 10 == 0:
                logger.info(f"當前記憶體使用: {cpu_usage:.2f} MB (峰值: {self.peak_memory:.2f} MB)")
                logger.info(f"系統記憶體使用率: {system_usage['percent']:.2f}%")
                
                if self.enable_gpu:
                    logger.info(f"當前 GPU 記憶體使用: {gpu_usage:.2f} MB (峰值: {self.peak_gpu_memory:.2f} MB)")
                
                # 詳細監控
                if self.detailed_monitoring:
                    self._log_detailed_memory_usage()
            
            # 等待下一次監控
            time.sleep(self.interval)
    
    def _get_cpu_memory_usage(self) -> float:
        """
        獲取 CPU 記憶體使用情況
        
        返回:
            float: 當前進程的記憶體使用量 (MB)
        """
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # 轉換為 MB
    
    def _get_system_memory_usage(self) -> Dict[str, float]:
        """
        獲取系統記憶體使用情況
        
        返回:
            Dict[str, float]: 系統記憶體使用信息
        """
        memory = psutil.virtual_memory()
        return {
            'total': memory.total / (1024 * 1024),  # MB
            'available': memory.available / (1024 * 1024),  # MB
            'used': memory.used / (1024 * 1024),  # MB
            'percent': memory.percent
        }
    
    def _get_gpu_memory_usage(self) -> float:
        """
        獲取 GPU 記憶體使用情況
        
        返回:
            float: 當前 GPU 記憶體使用量 (MB)
        """
        if not self.enable_gpu:
            return 0
        
        try:
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # 轉換為 MB
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # 轉換為 MB
            return allocated
        except:
            return 0
    
    def _log_detailed_memory_usage(self):
        """
        記錄詳細的記憶體使用情況
        
        使用 tracemalloc 分析記憶體分配情況，找出主要的記憶體使用來源。
        """
        if not tracemalloc.is_tracing():
            return
        
        # 獲取當前記憶體快照
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        logger.info("記憶體分配 Top 5:")
        for i, stat in enumerate(top_stats[:5], 1):
            logger.info(f"#{i}: {stat.size / 1024:.1f} KB - {stat.traceback.format()[0]}")
    
    def generate_report(self) -> Dict[str, Any]:
        """
        生成記憶體使用報告
        
        創建詳細的記憶體使用報告，包括統計數據、圖表和警告信息。
        
        返回:
            Dict[str, Any]: 報告摘要
        """
        if not self.memory_usage:
            logger.warning("沒有記憶體使用數據，無法生成報告")
            return {}
        
        # 生成報告文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.report_dir, f"memory_report_{timestamp}.json")
        plot_file = os.path.join(self.report_dir, f"memory_plot_{timestamp}.png")
        
        # 準備報告數據
        report_data = {
            "timestamp": timestamp,
            "duration": (self.timestamps[-1] - self.timestamps[0]).total_seconds(),
            "interval": self.interval,
            "memory_usage": {
                "min": min(self.memory_usage),
                "max": max(self.memory_usage),
                "avg": sum(self.memory_usage) / len(self.memory_usage),
                "final": self.memory_usage[-1],
                "peak": self.peak_memory
            },
            "system_memory": {
                "avg_percent": sum(m['percent'] for m in self.system_memory_usage) / len(self.system_memory_usage),
                "final_percent": self.system_memory_usage[-1]['percent'],
                "total_mb": self.system_memory_usage[-1]['total']
            },
            "timestamps": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in self.timestamps],
            "memory_values": self.memory_usage,
            "alerts": self.alerts
        }
        
        # 添加 GPU 記憶體信息
        if self.enable_gpu and self.gpu_memory_usage:
            report_data["gpu_memory_usage"] = {
                "min": min(self.gpu_memory_usage),
                "max": max(self.gpu_memory_usage),
                "avg": sum(self.gpu_memory_usage) / len(self.gpu_memory_usage),
                "final": self.gpu_memory_usage[-1],
                "peak": self.peak_gpu_memory
            }
            report_data["gpu_memory_values"] = self.gpu_memory_usage
        
        # 保存報告
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=4)
        
        logger.info(f"記憶體使用報告已保存至: {report_file}")
        
        # 繪製記憶體使用圖表
        plt.figure(figsize=(12, 8))
        
        # CPU 記憶體使用
        plt.subplot(2, 1, 1)
        plt.plot(self.timestamps, self.memory_usage, label='進程記憶體使用')
        plt.axhline(y=self.peak_memory, color='r', linestyle='--', label=f'峰值: {self.peak_memory:.2f} MB')
        plt.xlabel('時間')
        plt.ylabel('記憶體使用 (MB)')
        plt.title('進程記憶體使用趨勢')
        plt.grid(True)
        plt.legend()
        
        # 系統記憶體使用率
        plt.subplot(2, 1, 2)
        system_percents = [m['percent'] for m in self.system_memory_usage]
        plt.plot(self.timestamps, system_percents, label='系統記憶體使用率', color='green')
        plt.axhline(y=self.alert_threshold, color='r', linestyle='--', label=f'警告閾值: {self.alert_threshold}%')
        plt.xlabel('時間')
        plt.ylabel('使用率 (%)')
        plt.title('系統記憶體使用率趨勢')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(plot_file)
        
        # 如果有 GPU 數據，繪製 GPU 記憶體使用圖表
        if self.enable_gpu and self.gpu_memory_usage:
            gpu_plot_file = os.path.join(self.report_dir, f"gpu_memory_plot_{timestamp}.png")
            plt.figure(figsize=(12, 6))
            plt.plot(self.timestamps, self.gpu_memory_usage, label='GPU 記憶體使用', color='orange')
            plt.axhline(y=self.peak_gpu_memory, color='r', linestyle='--', label=f'峰值: {self.peak_gpu_memory:.2f} MB')
            plt.xlabel('時間')
            plt.ylabel('記憶體使用 (MB)')
            plt.title('GPU 記憶體使用趨勢')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(gpu_plot_file)
            logger.info(f"GPU 記憶體使用圖表已保存至: {gpu_plot_file}")
        
        logger.info(f"記憶體使用圖表已保存至: {plot_file}")
        
        # 返回報告摘要
        summary = {
            "process_memory": {
                "min": min(self.memory_usage),
                "max": max(self.memory_usage),
                "avg": sum(self.memory_usage) / len(self.memory_usage),
                "final": self.memory_usage[-1],
                "peak": self.peak_memory
            },
            "system_memory_percent": {
                "avg": sum(m['percent'] for m in self.system_memory_usage) / len(self.system_memory_usage),
                "final": self.system_memory_usage[-1]['percent']
            },
            "alerts_count": len(self.alerts)
        }
        
        if self.enable_gpu and self.gpu_memory_usage:
            summary["gpu_memory"] = {
                "peak": self.peak_gpu_memory,
                "final": self.gpu_memory_usage[-1]
            }
        
        return summary

def get_memory_monitor(interval: int = 10, report_dir: str = './memory_reports', 
                      enable_gpu: bool = True, detailed_monitoring: bool = False,
                      alert_threshold: float = 90.0) -> MemoryMonitor:
    """
    獲取記憶體監控器實例（單例模式）
    
    參數:
        interval (int): 監控間隔（秒）
        report_dir (str): 報告保存目錄
        enable_gpu (bool): 是否監控 GPU 記憶體
        detailed_monitoring (bool): 是否啟用詳細監控
        alert_threshold (float): 記憶體使用率警告閾值（百分比）
        
    返回:
        MemoryMonitor: 記憶體監控器實例
    """
    global _MEMORY_MONITOR_INSTANCE
    
    if _MEMORY_MONITOR_INSTANCE is None:
        _MEMORY_MONITOR_INSTANCE = MemoryMonitor(
            interval=interval,
            report_dir=report_dir,
            enable_gpu=enable_gpu,
            detailed_monitoring=detailed_monitoring,
            alert_threshold=alert_threshold
        )
    
    return _MEMORY_MONITOR_INSTANCE

def memory_mapped_array(shape: Tuple[int, ...], dtype: np.dtype = np.float32, 
                        filename: Optional[str] = None, mode: str = 'w+') -> np.memmap:
    """
    創建記憶體映射數組
    
    將大型數組存儲在磁盤上，通過記憶體映射進行訪問，減少內存使用。
    
    參數:
        shape (Tuple[int, ...]): 數組形狀
        dtype (np.dtype): 數據類型
        filename (Optional[str]): 文件名，如果為 None 則使用臨時文件
        mode (str): 文件模式，'r' 只讀，'w+' 讀寫
        
    返回:
        np.memmap: 記憶體映射數組
    """
    if filename is None:
        import tempfile
        temp_dir = tempfile.gettempdir()
        filename = os.path.join(temp_dir, f"memmap_{int(time.time())}.dat")
    
    # 確保目錄存在
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # 創建記憶體映射數組
    memmap_array = np.memmap(filename, dtype=dtype, mode=mode, shape=shape)
    
    # 保存形狀信息，以便後續加載
    shape_file = f"{filename}.shape.npy"
    np.save(shape_file, np.array(shape))
    
    logger.info(f"創建記憶體映射數組: {filename}, 形狀: {shape}, 類型: {dtype}")
    
    return memmap_array

def load_memory_mapped_array(filename: str, mode: str = 'r') -> np.memmap:
    """
    加載記憶體映射數組
    
    參數:
        filename (str): 記憶體映射文件路徑
        mode (str): 文件模式，'r' 只讀，'r+' 讀寫
        
    返回:
        np.memmap: 記憶體映射數組
    """
    # 加載形狀信息
    shape_file = f"{filename}.shape.npy"
    if not os.path.exists(shape_file):
        raise FileNotFoundError(f"找不到形狀文件: {shape_file}")
    
    shape = tuple(np.load(shape_file))
    
    # 加載記憶體映射數組
    memmap_array = np.memmap(filename, mode=mode, shape=shape)
    
    logger.info(f"加載記憶體映射數組: {filename}, 形狀: {shape}")
    
    return memmap_array

def save_dataframe_chunked(df: pd.DataFrame, filename: str, chunk_size: int = 10000, 
                          compression: Optional[str] = None, index: bool = False) -> None:
    """
    分塊保存 DataFrame 到 CSV 文件
    
    將大型 DataFrame 分塊保存，減少內存使用。
    
    參數:
        df (pd.DataFrame): 數據框
        filename (str): 文件名
        chunk_size (int): 每塊的行數
        compression (Optional[str]): 壓縮格式，如 'gzip', 'bz2', 'xz', 'zip'
        index (bool): 是否保存索引
    """
    # 確保目錄存在
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # 計算塊數
    n_chunks = (len(df) + chunk_size - 1) // chunk_size
    
    logger.info(f"分塊保存 DataFrame: {filename}, 總行數: {len(df)}, 塊大小: {chunk_size}, 塊數: {n_chunks}")
    
    # 分塊保存
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        
        chunk = df.iloc[start_idx:end_idx]
        
        # 第一塊包含表頭，後續塊不包含
        header = i == 0
        mode = 'w' if i == 0 else 'a'
        
        chunk.to_csv(filename, mode=mode, header=header, index=index, compression=compression)
        
        # 釋放記憶體
        del chunk
        gc.collect()
        
        logger.info(f"已保存塊 {i+1}/{n_chunks}, 行數: {end_idx - start_idx}")

def load_dataframe_chunked(filename: str, chunk_size: int = 10000, 
                          compression: Optional[str] = None, dtype: Optional[Dict] = None,
                          optimize_memory: bool = True) -> pd.DataFrame:
    """
    分塊加載 CSV 文件到 DataFrame
    
    分塊加載大型 CSV 文件，減少內存使用。
    
    參數:
        filename (str): 文件名
        chunk_size (int): 每塊的行數
        compression (Optional[str]): 壓縮格式，如 'gzip', 'bz2', 'xz', 'zip'
        dtype (Optional[Dict]): 列數據類型
        optimize_memory (bool): 是否優化記憶體使用
        
    返回:
        pd.DataFrame: 合併後的數據框
    """
    logger.info(f"分塊加載 DataFrame: {filename}, 塊大小: {chunk_size}")
    
    # 使用 pandas 的 chunked 讀取
    chunks = []
    total_rows = 0
    
    for i, chunk in enumerate(pd.read_csv(filename, chunksize=chunk_size, 
                                         compression=compression, dtype=dtype, 
                                         low_memory=False)):
        # 優化記憶體使用
        if optimize_memory:
            chunk = optimize_dataframe_memory(chunk, verbose=False)
        
        chunks.append(chunk)
        total_rows += len(chunk)
        logger.info(f"已加載塊 {i+1}，行數: {len(chunk)}, 總行數: {total_rows}")
        
        # 定期清理記憶體
        if (i + 1) % 5 == 0:
            clean_memory()
    
    # 合併所有塊
    logger.info(f"合併 {len(chunks)} 個數據塊...")
    df = pd.concat(chunks, ignore_index=True)
    
    # 釋放記憶體
    del chunks
    clean_memory()
    
    logger.info(f"完成加載，總行數: {len(df)}")
    
    return df

def optimize_dataframe_memory(df: pd.DataFrame, category_threshold: int = 50, 
                             verbose: bool = True, deep_optimization: bool = False) -> pd.DataFrame:
    """
    優化 DataFrame 記憶體使用
    
    通過選擇最佳數據類型和轉換為分類類型來減少 DataFrame 的記憶體使用。
    
    參數:
        df (pd.DataFrame): 數據框
        category_threshold (int): 唯一值數量閾值，低於此值的列將轉換為分類類型
        verbose (bool): 是否輸出詳細信息
        deep_optimization (bool): 是否進行深度優化（更積極的類型轉換）
        
    返回:
        pd.DataFrame: 優化後的數據框
    """
    start_mem = df.memory_usage(deep=True).sum() / (1024 * 1024)
    if verbose:
        logger.info(f"原始 DataFrame 記憶體使用: {start_mem:.2f} MB")
    
    # 複製 DataFrame 以避免修改原始數據
    df_optimized = df.copy()
    
    # 遍歷所有列
    for col in df_optimized.columns:
        # 獲取列數據類型
        col_type = df_optimized[col].dtype
        
        # 數值類型優化
        if pd.api.types.is_integer_dtype(col_type):
            # 檢查值範圍
            col_min = df_optimized[col].min()
            col_max = df_optimized[col].max()
            
            # 根據值範圍選擇最小的數據類型
            if col_min >= 0:
                if col_max < 2**8:
                    df_optimized[col] = df_optimized[col].astype(np.uint8)
                elif col_max < 2**16:
                    df_optimized[col] = df_optimized[col].astype(np.uint16)
                elif col_max < 2**32:
                    df_optimized[col] = df_optimized[col].astype(np.uint32)
            else:
                if col_min > -2**7 and col_max < 2**7:
                    df_optimized[col] = df_optimized[col].astype(np.int8)
                elif col_min > -2**15 and col_max < 2**15:
                    df_optimized[col] = df_optimized[col].astype(np.int16)
                elif col_min > -2**31 and col_max < 2**31:
                    df_optimized[col] = df_optimized[col].astype(np.int32)
        
        # 浮點類型優化
        elif pd.api.types.is_float_dtype(col_type):
            # 檢查是否可以轉換為較小的浮點類型
            if deep_optimization:
                # 檢查值範圍，決定是否可以使用 float16
                col_min = df_optimized[col].min()
                col_max = df_optimized[col].max()
                
                # float16 範圍約為 ±65504
                if col_min > -65504 and col_max < 65504:
                    df_optimized[col] = df_optimized[col].astype(np.float16)
                else:
                    df_optimized[col] = df_optimized[col].astype(np.float32)
            else:
                # 標準優化：使用 float32 代替 float64
                df_optimized[col] = df_optimized[col].astype(np.float32)
        
        # 字符串類型優化
        elif col_type == object:
            # 檢查是否可以轉換為分類類型
            n_unique = df_optimized[col].nunique()
            if n_unique < category_threshold:
                df_optimized[col] = df_optimized[col].astype('category')
            elif deep_optimization:
                # 嘗試推斷更好的數據類型
                try:
                    # 嘗試轉換為日期時間
                    df_optimized[col] = pd.to_datetime(df_optimized[col], errors='ignore')
                except:
                    pass
    
    # 計算優化後的記憶體使用
    end_mem = df_optimized.memory_usage(deep=True).sum() / (1024 * 1024)
    reduction = 100 * (start_mem - end_mem) / start_mem
    
    if verbose:
        logger.info(f"優化後 DataFrame 記憶體使用: {end_mem:.2f} MB")
        logger.info(f"記憶體減少: {reduction:.2f}%")
    
    return df_optimized

def clean_memory(aggressive: bool = False) -> Dict[str, float]:
    """
    主動清理記憶體
    
    強制執行垃圾回收並清理未使用的記憶體。
    
    參數:
        aggressive (bool): 是否進行積極的記憶體清理
        
    返回:
        Dict[str, float]: 清理前後的記憶體使用信息
    """
    # 獲取清理前的記憶體使用
    before = get_memory_usage()
    
    # 強制執行垃圾回收
    gc.collect()
    
    # 如果啟用了積極清理，執行更多清理操作
    if aggressive:
        # 多次執行垃圾回收
        for _ in range(3):
            gc.collect()
        
        # 清理 Python 解釋器緩存
        import sys
        sys.exc_clear() if hasattr(sys, 'exc_clear') else None
        
        # 清理 numpy 緩存
        np.clear_cache() if hasattr(np, 'clear_cache') else None
    
    # 如果使用 CUDA，清理 CUDA 緩存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        # 如果啟用了積極清理，執行更多 GPU 清理操作
        if aggressive and hasattr(torch.cuda, 'synchronize'):
            torch.cuda.synchronize()
    
    # 獲取清理後的記憶體使用
    after = get_memory_usage()
    
    # 計算清理效果
    cpu_diff = before['cpu_memory_mb'] - after['cpu_memory_mb']
    gpu_diff = before.get('gpu_memory_mb', 0) - after.get('gpu_memory_mb', 0)
    
    logger.info("已執行記憶體清理")
    if cpu_diff > 0 or gpu_diff > 0:
        logger.info(f"  釋放 CPU 記憶體: {cpu_diff:.2f} MB")
        if torch.cuda.is_available():
            logger.info(f"  釋放 GPU 記憶體: {gpu_diff:.2f} MB")
    
    return {
        'before': before,
        'after': after,
        'cpu_diff': cpu_diff,
        'gpu_diff': gpu_diff
    }

def limit_gpu_memory(limit_mb: int = 0, device: int = 0) -> None:
    """
    限制 GPU 記憶體使用
    
    參數:
        limit_mb (int): 記憶體限制 (MB)，0 表示不限制
        device (int): GPU 設備 ID
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA 不可用，無法限制 GPU 記憶體")
        return
    
    if limit_mb <= 0:
        logger.info("不限制 GPU 記憶體使用")
        return
    
    try:
        # 獲取設備總記憶體
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)  # MB
        
        # 確保限制不超過總記憶體
        if limit_mb > total_memory:
            logger.warning(f"限制 ({limit_mb} MB) 超過設備總記憶體 ({total_memory:.0f} MB)，將使用 80% 的總記憶體")
            limit_mb = int(total_memory * 0.8)
        
        # 設置 CUDA 記憶體分配器
        fraction = limit_mb / total_memory
        torch.cuda.set_per_process_memory_fraction(fraction, device)
        logger.info(f"已限制 GPU {device} 記憶體使用: {limit_mb} MB ({fraction:.1%} 的總記憶體)")
    except Exception as e:
        logger.error(f"限制 GPU 記憶體時發生錯誤: {str(e)}")

@contextmanager
def track_memory_usage(name: str = "操作", detailed: bool = False) -> None:
    """
    記憶體使用追蹤上下文管理器
    
    用於追蹤代碼塊的記憶體使用情況。
    
    參數:
        name (str): 操作名稱
        detailed (bool): 是否顯示詳細信息
        
    用法:
        with track_memory_usage("數據加載"):
            data = load_large_dataset()
    """
    # 記錄開始時的記憶體使用
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / (1024 * 1024)
    
    # 記錄 GPU 記憶體（如果可用）
    start_gpu_mem = 0
    if torch.cuda.is_available():
        start_gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)
    
    start_time = time.time()
    
    if detailed:
        logger.info(f"開始 {name}...")
        logger.info(f"  初始 CPU 記憶體: {start_mem:.2f} MB")
        if torch.cuda.is_available():
            logger.info(f"  初始 GPU 記憶體: {start_gpu_mem:.2f} MB")
    
    try:
        yield
    finally:
        # 記錄結束時的記憶體使用
        end_mem = process.memory_info().rss / (1024 * 1024)
        end_time = time.time()
        
        # 計算記憶體使用變化
        delta_mem = end_mem - start_mem
        elapsed_time = end_time - start_time
        
        # 記錄 GPU 記憶體變化
        delta_gpu_mem = 0
        if torch.cuda.is_available():
            end_gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)
            delta_gpu_mem = end_gpu_mem - start_gpu_mem
        
        logger.info(f"{name} 完成 (耗時: {elapsed_time:.2f}s)")
        logger.info(f"  記憶體變化: {delta_mem:+.2f} MB (從 {start_mem:.2f} MB 到 {end_mem:.2f} MB)")
        
        if torch.cuda.is_available():
            logger.info(f"  GPU 記憶體變化: {delta_gpu_mem:+.2f} MB (從 {start_gpu_mem:.2f} MB 到 {end_gpu_mem:.2f} MB)")

def memory_usage_decorator(func: Callable) -> Callable:
    """
    記憶體使用裝飾器
    
    用於追蹤函數的記憶體使用情況。
    
    參數:
        func: 要監控的函數
        
    返回:
        callable: 包裝後的函數
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        with track_memory_usage(func.__name__):
            return func(*args, **kwargs)
    
    return wrapper

def get_memory_usage() -> Dict[str, float]:
    """
    獲取當前記憶體使用情況
    
    返回:
        Dict[str, float]: 記憶體使用信息
    """
    # CPU 記憶體
    process = psutil.Process(os.getpid())
    cpu_mem = process.memory_info().rss / (1024 * 1024)
    
    # 系統記憶體
    system_mem = psutil.virtual_memory()
    
    # GPU 記憶體
    gpu_mem = 0
    gpu_mem_reserved = 0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)
        gpu_mem_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    
    return {
        "cpu_memory_mb": cpu_mem,
        "system_memory_total_mb": system_mem.total / (1024 * 1024),
        "system_memory_used_mb": system_mem.used / (1024 * 1024),
        "system_memory_percent": system_mem.percent,
        "gpu_memory_mb": gpu_mem,
        "gpu_memory_reserved_mb": gpu_mem_reserved,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def print_memory_usage(detailed: bool = False) -> None:
    """
    打印當前記憶體使用情況
    
    參數:
        detailed (bool): 是否顯示詳細信息
    """
    mem_info = get_memory_usage()
    
    logger.info("當前記憶體使用情況:")
    logger.info(f"  CPU 記憶體: {mem_info['cpu_memory_mb']:.2f} MB")
    logger.info(f"  系統記憶體: {mem_info['system_memory_used_mb']:.2f} MB / {mem_info['system_memory_total_mb']:.2f} MB ({mem_info['system_memory_percent']:.1f}%)")
    
    if torch.cuda.is_available():
        logger.info(f"  GPU 記憶體: {mem_info['gpu_memory_mb']:.2f} MB (已分配)")
        logger.info(f"  GPU 記憶體: {mem_info['gpu_memory_reserved_mb']:.2f} MB (已保留)")
        
        if detailed:
            # 顯示每個 GPU 的詳細信息
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  GPU {i} ({props.name}):")
                logger.info(f"    總記憶體: {props.total_memory / (1024 * 1024):.2f} MB")
                logger.info(f"    處理器數量: {props.multi_processor_count}")
                logger.info(f"    CUDA 能力: {props.major}.{props.minor}")

def get_memory_optimization_suggestions() -> List[str]:
    """
    獲取記憶體優化建議
    
    根據當前系統狀態提供記憶體優化建議。
    
    返回:
        List[str]: 優化建議列表
    """
    suggestions = []
    
    # 獲取記憶體使用情況
    mem_info = get_memory_usage()
    
    # 系統記憶體使用率高
    if mem_info['system_memory_percent'] > 80:
        suggestions.append("系統記憶體使用率高 (> 80%)，建議減小批次大小或使用記憶體映射")
        suggestions.append("考慮啟用增量式資料加載 (incremental_loading=True)")
        suggestions.append("使用 save_preprocessed=True 保存預處理結果，避免重複計算")
    
    # GPU 記憶體使用高
    if torch.cuda.is_available() and mem_info['gpu_memory_mb'] > 1000:
        suggestions.append("GPU 記憶體使用較高 (> 1GB)，建議使用混合精度訓練 (use_mixed_precision=True)")
        suggestions.append("啟用梯度檢查點 (use_gradient_checkpointing=True) 減少激活值記憶體使用")
        suggestions.append("考慮使用梯度累積 (use_gradient_accumulation=True) 減少批次大小")
    
    # 一般建議
    suggestions.append("使用 optimize_dataframe_memory() 優化 DataFrame 記憶體使用")
    suggestions.append("定期調用 clean_memory() 清理未使用的記憶體")
    suggestions.append("對大型數據集使用 memory_mapped_array() 進行記憶體映射")
    suggestions.append("使用 track_memory_usage() 上下文管理器追蹤關鍵操作的記憶體使用")
    
    # 根據系統狀態添加更多建議
    if psutil.virtual_memory().available < 2 * 1024 * 1024 * 1024:  # 小於 2GB 可用
        suggestions.append("系統可用記憶體不足 (< 2GB)，建議關閉其他應用程序釋放記憶體")
        suggestions.append("考慮使用更小的資料子集進行訓練")
    
    return suggestions

def print_optimization_suggestions() -> None:
    """打印記憶體優化建議"""
    suggestions = get_memory_optimization_suggestions()
    
    logger.info("記憶體優化建議:")
    for i, suggestion in enumerate(suggestions, 1):
        logger.info(f"  {i}. {suggestion}")

def detect_memory_leaks(iterations: int = 5, func: Optional[Callable] = None, *args, **kwargs) -> Dict[str, Any]:
    """
    檢測記憶體洩漏
    
    通過多次執行函數並監控記憶體使用變化來檢測潛在的記憶體洩漏。
    
    參數:
        iterations (int): 執行迭代次數
        func (Optional[Callable]): 要測試的函數，如果為 None 則僅監控記憶體使用
        *args, **kwargs: 傳遞給函數的參數
        
    返回:
        Dict[str, Any]: 記憶體洩漏檢測結果
    """
    logger.info(f"開始記憶體洩漏檢測 ({iterations} 次迭代)...")
    
    # 啟用 tracemalloc
    tracemalloc.start()
    
    memory_usage = []
    snapshot1 = None
    
    # 執行多次迭代
    for i in range(iterations):
        # 清理記憶體
        clean_memory(aggressive=True)
        
        # 記錄開始時的記憶體使用
        if i == 0:
            snapshot1 = tracemalloc.take_snapshot()
        
        # 執行函數
        if func is not None:
            func(*args, **kwargs)
        
        # 記錄記憶體使用
        memory_info = get_memory_usage()
        memory_usage.append(memory_info['cpu_memory_mb'])
        
        logger.info(f"  迭代 {i+1}/{iterations}: {memory_info['cpu_memory_mb']:.2f} MB")
    
    # 獲取最終快照
    snapshot2 = tracemalloc.take_snapshot()
    
    # 停止 tracemalloc
    tracemalloc.stop()
    
    # 分析記憶體使用趨勢
    memory_trend = np.polyfit(range(iterations), memory_usage, 1)[0]
    
    # 比較快照
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    
    # 準備結果
    result = {
        'memory_usage': memory_usage,
        'memory_trend': memory_trend,
        'has_leak': memory_trend > 1.0,  # 如果每次迭代增加超過 1MB，則可能存在洩漏
        'top_differences': []
    }
    
    # 記錄頂部差異
    logger.info("記憶體分配差異 Top 10:")
    for i, stat in enumerate(top_stats[:10], 1):
        trace = stat.traceback.format()
        size = stat.size / 1024  # KB
        if size > 0:  # 只關注增加的部分
            logger.info(f"  #{i}: {size:.1f} KB - {trace[0]}")
            result['top_differences'].append({
                'size_kb': size,
                'trace': trace[0]
            })
    
    # 結論
    if result['has_leak']:
        logger.warning(f"檢測到可能的記憶體洩漏: 每次迭代增加約 {memory_trend:.2f} MB")
    else:
        logger.info("未檢測到明顯的記憶體洩漏")
    
    return result

def adaptive_batch_size(initial_batch_size: int, memory_threshold: float = 0.8, 
                       min_batch_size: int = 16, step_size: float = 0.5) -> int:
    """
    自適應批次大小
    
    根據當前記憶體使用情況動態調整批次大小。
    
    參數:
        initial_batch_size (int): 初始批次大小
        memory_threshold (float): 記憶體使用率閾值 (0.0-1.0)
        min_batch_size (int): 最小批次大小
        step_size (float): 調整步長係數
        
    返回:
        int: 調整後的批次大小
    """
    # 獲取當前記憶體使用情況
    mem_info = get_memory_usage()
    system_memory_percent = mem_info['system_memory_percent'] / 100.0
    
    # 如果 GPU 可用，也考慮 GPU 記憶體
    gpu_memory_percent = 0.0
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_percent = allocated / total
        except:
            pass
    
    # 使用較高的記憶體使用率
    memory_percent = max(system_memory_percent, gpu_memory_percent)
    
    # 根據記憶體使用率調整批次大小
    if memory_percent > memory_threshold:
        # 記憶體使用率高，減小批次大小
        reduction_factor = 1.0 - step_size * ((memory_percent - memory_threshold) / (1.0 - memory_threshold))
        new_batch_size = max(min_batch_size, int(initial_batch_size * reduction_factor))
        
        logger.info(f"記憶體使用率 ({memory_percent:.1%}) 超過閾值 ({memory_threshold:.1%})，"
                   f"將批次大小從 {initial_batch_size} 減小到 {new_batch_size}")
        return new_batch_size
    else:
        # 記憶體使用率低，可以使用初始批次大小
        return initial_batch_size

# 主程式測試
if __name__ == "__main__":
    # 測試記憶體監控
    monitor = MemoryMonitor(interval=1, report_dir='./test_memory_reports', detailed_monitoring=True)
    monitor.start()
    
    # 創建一些數據以測試記憶體使用
    data = []
    for i in range(10):
        # 分配一些記憶體
        arr = np.random.randn(1000, 1000)
        data.append(arr)
        
        # 等待一秒
        time.sleep(1)
    
    # 停止監控並生成報告
    monitor.stop()
    
    # 測試記憶體映射數組
    memmap_arr = memory_mapped_array((1000, 1000))
    memmap_arr[:] = np.random.randn(1000, 1000)
    
    # 測試 DataFrame 記憶體優化
    df = pd.DataFrame({
        'int_col': np.random.randint(0, 100, 10000),
        'float_col': np.random.randn(10000),
        'str_col': ['str_' + str(i % 10) for i in range(10000)]
    })
    
    df_optimized = optimize_dataframe_memory(df, deep_optimization=True)
    
    # 測試記憶體使用追蹤
    with track_memory_usage("大型數組操作", detailed=True):
        # 分配一些記憶體
        arr = np.random.randn(2000, 2000)
        time.sleep(1)
        del arr
    
    # 測試記憶體使用情況打印
    print_memory_usage(detailed=True)
    
    # 測試優化建議
    print_optimization_suggestions()
    
    # 測試記憶體洩漏檢測
    def leaky_function():
        # 模擬記憶體洩漏
        global leaky_list
        if not 'leaky_list' in globals():
            leaky_list = []
        leaky_list.append(np.random.randn(100, 100))
        time.sleep(0.5)
    
    detect_memory_leaks(iterations=3, func=leaky_function)
    
    # 測試自適應批次大小
    batch_size = adaptive_batch_size(128, memory_threshold=0.5)
    print(f"自適應批次大小: {batch_size}")
    
    # 清理記憶體
    clean_memory(aggressive=True)
