#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
記憶體優化工具模組

此模組提供記憶體優化相關工具函數，包括：
1. 記憶體使用監控與分析
2. 記憶體映射文件操作
3. 主動記憶體管理與釋放
4. GPU 記憶體優化與控制
5. 記憶體使用報告生成
"""

import os
import gc
import time
import psutil
import numpy as np
import torch
import logging
import json
import tracemalloc
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
                 enable_gpu: bool = True, alert_threshold: float = 90.0):
        """
        初始化記憶體監控器
        
        參數:
            interval (int): 監控間隔（秒）
            report_dir (str): 報告保存目錄
            enable_gpu (bool): 是否監控 GPU 記憶體
            alert_threshold (float): 記憶體使用率警告閾值（百分比）
        """
        self.interval = interval
        self.report_dir = report_dir
        self.enable_gpu = enable_gpu and torch.cuda.is_available()
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
        
        # 初始化 tracemalloc
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
        
        # 停止 tracemalloc
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
        plt.plot(self.timestamps, self.memory_usage, label='Process Memory Usage')
        plt.axhline(y=self.peak_memory, color='r', linestyle='--', label=f'Peak: {self.peak_memory:.2f} MB')
        plt.xlabel('Time')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Process Memory Usage Trend')
        plt.grid(True)
        plt.legend()
        
        # 系統記憶體使用率
        plt.subplot(2, 1, 2)
        system_percents = [m['percent'] for m in self.system_memory_usage]
        plt.plot(self.timestamps, system_percents, label='System Memory Usage', color='green')
        plt.axhline(y=self.alert_threshold, color='r', linestyle='--', label=f'Alert Threshold: {self.alert_threshold}%')
        plt.xlabel('Time')
        plt.ylabel('Usage (%)')
        plt.title('System Memory Usage Trend')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(plot_file)
        
        # 如果有 GPU 數據，繪製 GPU 記憶體使用圖表
        if self.enable_gpu and self.gpu_memory_usage:
            gpu_plot_file = os.path.join(self.report_dir, f"gpu_memory_plot_{timestamp}.png")
            plt.figure(figsize=(12, 6))
            plt.plot(self.timestamps, self.gpu_memory_usage, label='GPU Memory Usage', color='orange')
            plt.axhline(y=self.peak_gpu_memory, color='r', linestyle='--', label=f'Peak: {self.peak_gpu_memory:.2f} MB')
            plt.xlabel('Time')
            plt.ylabel('Memory Usage (MB)')
            plt.title('GPU Memory Usage Trend')
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
                      enable_gpu: bool = True, alert_threshold: float = 90.0) -> MemoryMonitor:
    """
    獲取記憶體監控器實例（單例模式）
    
    參數:
        interval (int): 監控間隔（秒）
        report_dir (str): 報告保存目錄
        enable_gpu (bool): 是否監控 GPU 記憶體
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

@contextmanager
def track_memory_usage(name: str = "操作") -> None:
    """
    記憶體使用追蹤上下文管理器
    
    用於追蹤代碼塊的記憶體使用情況。
    
    參數:
        name (str): 操作名稱
        
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
    
    logger.info(f"開始 {name}...")
    
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
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024)
    
    return {
        "cpu_memory_mb": cpu_mem,
        "system_memory_total_mb": system_mem.total / (1024 * 1024),
        "system_memory_used_mb": system_mem.used / (1024 * 1024),
        "system_memory_percent": system_mem.percent,
        "gpu_memory_mb": gpu_mem,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def print_memory_usage() -> None:
    """
    打印當前記憶體使用情況
    """
    mem_info = get_memory_usage()
    
    logger.info("當前記憶體使用情況:")
    logger.info(f"  CPU 記憶體: {mem_info['cpu_memory_mb']:.2f} MB")
    logger.info(f"  系統記憶體: {mem_info['system_memory_used_mb']:.2f} MB / {mem_info['system_memory_total_mb']:.2f} MB ({mem_info['system_memory_percent']:.1f}%)")
    
    if torch.cuda.is_available():
        logger.info(f"  GPU 記憶體: {mem_info['gpu_memory_mb']:.2f} MB (已分配)")

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
    
    if cpu_diff > 1 or gpu_diff > 1:  # 只在有明顯效果時記錄
        logger.info("已執行記憶體清理")
        if cpu_diff > 1:
            logger.info(f"  釋放 CPU 記憶體: {cpu_diff:.2f} MB")
        if torch.cuda.is_available() and gpu_diff > 1:
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


def save_dataframe_chunked(df: pd.DataFrame, path: str, chunk_size: int = 10000, 
                          compression: Optional[str] = None) -> None:
    """
    分塊保存 DataFrame 到 CSV 文件
    
    對於大型 DataFrame，分塊保存可以減少記憶體使用。
    
    參數:
        df (pd.DataFrame): 要保存的 DataFrame
        path (str): 保存路徑
        chunk_size (int): 每個塊的行數
        compression (str, optional): 壓縮格式，如 'gzip', 'bz2', 'zip' 等
    """
    # 確保目錄存在
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    # 獲取總行數
    total_rows = len(df)
    
    # 計算塊數
    num_chunks = (total_rows + chunk_size - 1) // chunk_size
    
    logger.info(f"分塊保存 DataFrame 到 {path}，共 {total_rows} 行，分為 {num_chunks} 個塊")
    
    # 分塊保存
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_rows)
        
        # 獲取當前塊
        chunk = df.iloc[start_idx:end_idx]
        
        # 保存當前塊
        if i == 0:
            # 第一個塊，包含表頭
            chunk.to_csv(path, mode='w', index=False, compression=compression)
        else:
            # 後續塊，不包含表頭
            chunk.to_csv(path, mode='a', header=False, index=False, compression=compression)
        
        logger.info(f"已保存塊 {i+1}/{num_chunks}，行 {start_idx} 到 {end_idx-1}")
        
        # 定期清理記憶體
        if (i + 1) % 5 == 0:
            clean_memory()


def load_dataframe_chunked(path: str, chunk_size: int = 10000, 
                          dtype: Optional[Dict] = None, 
                          compression: Optional[str] = None) -> pd.DataFrame:
    """
    分塊加載 CSV 文件到 DataFrame
    
    對於大型 CSV 文件，分塊加載可以減少記憶體使用。
    
    參數:
        path (str): CSV 文件路徑
        chunk_size (int): 每個塊的行數
        dtype (Dict, optional): 列的數據類型
        compression (str, optional): 壓縮格式，如 'gzip', 'bz2', 'zip' 等
        
    返回:
        pd.DataFrame: 加載的 DataFrame
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    
    logger.info(f"分塊加載 CSV 文件: {path}")
    
    # 使用 pandas 的 chunking 功能
    chunks = []
    for i, chunk in enumerate(pd.read_csv(path, chunksize=chunk_size, dtype=dtype, compression=compression)):
        chunks.append(chunk)
        logger.info(f"已加載塊 {i+1}，形狀: {chunk.shape}")
        
        # 定期清理記憶體
        if (i + 1) % 5 == 0:
            clean_memory()
    
    # 合併所有塊
    logger.info(f"合併 {len(chunks)} 個塊")
    df = pd.concat(chunks, ignore_index=True)
    
    # 釋放記憶體
    del chunks
    clean_memory()
    
    logger.info(f"加載完成，DataFrame 形狀: {df.shape}")
    
    return df


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    優化 DataFrame 的記憶體使用
    
    通過將列轉換為更合適的數據類型來減少記憶體使用。
    
    參數:
        df (pd.DataFrame): 要優化的 DataFrame
        
    返回:
        pd.DataFrame: 優化後的 DataFrame
    """
    start_mem = df.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.info(f"優化前 DataFrame 記憶體使用: {start_mem:.2f} MB")
    
    # 複製 DataFrame 以避免修改原始數據
    df_optimized = df.copy()
    
    # 處理數值型列
    for col in df_optimized.select_dtypes(include=['int']).columns:
        # 獲取列的最小值和最大值
        col_min = df_optimized[col].min()
        col_max = df_optimized[col].max()
        
        # 根據數據範圍選擇合適的數據類型
        if col_min >= 0:
            if col_max < 2**8:
                df_optimized[col] = df_optimized[col].astype(np.uint8)
            elif col_max < 2**16:
                df_optimized[col] = df_optimized[col].astype(np.uint16)
            elif col_max < 2**32:
                df_optimized[col] = df_optimized[col].astype(np.uint32)
            else:
                df_optimized[col] = df_optimized[col].astype(np.uint64)
        else:
            if col_min > -2**7 and col_max < 2**7:
                df_optimized[col] = df_optimized[col].astype(np.int8)
            elif col_min > -2**15 and col_max < 2**15:
                df_optimized[col] = df_optimized[col].astype(np.int16)
            elif col_min > -2**31 and col_max < 2**31:
                df_optimized[col] = df_optimized[col].astype(np.int32)
            else:
                df_optimized[col] = df_optimized[col].astype(np.int64)
    
    # 處理浮點型列
    for col in df_optimized.select_dtypes(include=['float']).columns:
        # 嘗試使用 float32 而不是 float64
        df_optimized[col] = df_optimized[col].astype(np.float32)
    
    # 處理對象型列
    for col in df_optimized.select_dtypes(include=['object']).columns:
        # 如果唯一值較少，使用分類類型
        if df_optimized[col].nunique() / len(df_optimized) < 0.5:
            df_optimized[col] = df_optimized[col].astype('category')
    
    # 計算優化後的記憶體使用
    end_mem = df_optimized.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.info(f"優化後 DataFrame 記憶體使用: {end_mem:.2f} MB")
    logger.info(f"記憶體使用減少: {(start_mem - end_mem) / start_mem * 100:.2f}%")
    
    return df_optimized


def get_memory_optimization_suggestions() -> Dict[str, Any]:
    """
    獲取記憶體優化建議
    
    根據當前系統狀態提供記憶體優化建議。
    
    返回:
        Dict[str, Any]: 優化建議
    """
    # 獲取記憶體使用情況
    mem_info = get_memory_usage()
    
    suggestions = {
        "high_memory_usage": False,
        "high_gpu_usage": False,
        "suggestions": []
    }
    
    # 系統記憶體使用率高
    if mem_info['system_memory_percent'] > 80:
        suggestions["high_memory_usage"] = True
        suggestions["suggestions"].extend([
            "減小批次大小或使用記憶體映射",
            "啟用增量式資料加載 (incremental_loading=True)",
            "使用 save_preprocessed=True 保存預處理結果，避免重複計算"
        ])
    
    # GPU 記憶體使用高
    if torch.cuda.is_available() and mem_info['gpu_memory_mb'] > 1000:
        suggestions["high_gpu_usage"] = True
        suggestions["suggestions"].extend([
            "使用混合精度訓練 (use_mixed_precision=True)",
            "啟用梯度檢查點 (use_gradient_checkpointing=True) 減少激活值記憶體使用",
            "考慮使用梯度累積 (use_gradient_accumulation=True) 減少批次大小"
        ])
    
    # 一般建議
    suggestions["suggestions"].extend([
        "使用子圖採樣減少圖的大小 (use_subgraph_sampling=True)",
        "定期調用 clean_memory() 清理未使用的記憶體",
        "對大型數據集使用 memory_mapped_array() 進行記憶體映射",
        "使用動態批次大小 (use_dynamic_batch_size=True) 自動調整批次大小"
    ])
    
    return suggestions


def print_optimization_suggestions() -> None:
    """
    打印記憶體優化建議
    
    根據當前系統狀態提供記憶體優化建議。
    """
    # 獲取優化建議
    suggestions = get_memory_optimization_suggestions()
    
    logger.info("記憶體優化建議:")
    
    # 系統記憶體使用率高
    if suggestions["high_memory_usage"]:
        logger.info("  1. 系統記憶體使用率高 (> 80%)，建議減小批次大小或使用記憶體映射")
        logger.info("  2. 啟用增量式資料加載 (incremental_loading=True)")
        logger.info("  3. 使用 save_preprocessed=True 保存預處理結果，避免重複計算")
    
    # GPU 記憶體使用高
    if suggestions["high_gpu_usage"]:
        logger.info("  4. GPU 記憶體使用較高 (> 1GB)，建議使用混合精度訓練 (use_mixed_precision=True)")
        logger.info("  5. 啟用梯度檢查點 (use_gradient_checkpointing=True) 減少激活值記憶體使用")
        logger.info("  6. 考慮使用梯度累積 (use_gradient_accumulation=True) 減少批次大小")
    
    # 一般建議
    logger.info("  7. 使用子圖採樣減少圖的大小 (use_subgraph_sampling=True)")
    logger.info("  8. 定期調用 clean_memory() 清理未使用的記憶體")
    logger.info("  9. 對大型數據集使用 memory_mapped_array() 進行記憶體映射")
    logger.info("  10. 使用動態批次大小 (use_dynamic_batch_size=True) 自動調整批次大小")


def detect_memory_leaks(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """
    檢測函數執行過程中的記憶體洩漏
    
    參數:
        func (Callable): 要檢測的函數
        *args: 函數的位置參數
        **kwargs: 函數的關鍵字參數
        
    返回:
        Dict[str, Any]: 記憶體洩漏檢測結果
    """
    # 啟動 tracemalloc
    tracemalloc.start()
    
    # 記錄開始時的記憶體使用
    start_mem = get_memory_usage()
    
    # 執行函數
    result = func(*args, **kwargs)
    
    # 記錄結束時的記憶體使用
    end_mem = get_memory_usage()
    
    # 獲取記憶體快照
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    # 停止 tracemalloc
    tracemalloc.stop()
    
    # 計算記憶體變化
    mem_diff = {
        'cpu_memory_mb': end_mem['cpu_memory_mb'] - start_mem['cpu_memory_mb'],
        'system_memory_percent': end_mem['system_memory_percent'] - start_mem['system_memory_percent'],
        'gpu_memory_mb': end_mem['gpu_memory_mb'] - start_mem['gpu_memory_mb']
    }
    
    # 記錄可能的記憶體洩漏
    potential_leaks = []
    for stat in top_stats[:10]:
        potential_leaks.append({
            'size': stat.size / 1024,  # KB
            'source': str(stat.traceback)
        })
    
    # 強制執行垃圾回收
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 記錄垃圾回收後的記憶體使用
    post_gc_mem = get_memory_usage()
    
    # 計算垃圾回收的效果
    gc_effect = {
        'cpu_memory_mb': end_mem['cpu_memory_mb'] - post_gc_mem['cpu_memory_mb'],
        'system_memory_percent': end_mem['system_memory_percent'] - post_gc_mem['system_memory_percent'],
        'gpu_memory_mb': end_mem['gpu_memory_mb'] - post_gc_mem['gpu_memory_mb']
    }
    
    # 判斷是否存在記憶體洩漏
    has_leak = (gc_effect['cpu_memory_mb'] < 0.5 * mem_diff['cpu_memory_mb']) and (mem_diff['cpu_memory_mb'] > 10)
    
    # 記錄結果
    leak_result = {
        'function_name': func.__name__,
        'memory_diff': mem_diff,
        'gc_effect': gc_effect,
        'has_leak': has_leak,
        'potential_leaks': potential_leaks
    }
    
    # 記錄日誌
    if has_leak:
        logger.warning(f"檢測到可能的記憶體洩漏: {func.__name__}")
        logger.warning(f"  記憶體變化: {mem_diff['cpu_memory_mb']:.2f} MB")
        logger.warning(f"  垃圾回收效果: {gc_effect['cpu_memory_mb']:.2f} MB")
        for i, leak in enumerate(potential_leaks[:3], 1):
            logger.warning(f"  可能的洩漏源 #{i}: {leak['size']:.2f} KB - {leak['source']}")
    else:
        logger.info(f"未檢測到記憶體洩漏: {func.__name__}")
        logger.info(f"  記憶體變化: {mem_diff['cpu_memory_mb']:.2f} MB")
        logger.info(f"  垃圾回收效果: {gc_effect['cpu_memory_mb']:.2f} MB")
    
    return leak_result
