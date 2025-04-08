#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
記憶體優化工具模組

此模組提供記憶體優化相關的工具函數，包括：
1. 記憶體使用監控
2. 記憶體映射文件操作
3. 主動記憶體管理
4. GPU 記憶體優化
5. 記憶體使用報告生成
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
from datetime import datetime
import matplotlib.pyplot as plt
from functools import wraps

# 設置日誌
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryMonitor:
    """記憶體使用監控類"""
    
    def __init__(self, interval=10, report_dir='./memory_reports', enable_gpu=True):
        """
        初始化記憶體監控器
        
        參數:
            interval (int): 監控間隔（秒）
            report_dir (str): 報告保存目錄
            enable_gpu (bool): 是否監控 GPU 記憶體
        """
        self.interval = interval
        self.report_dir = report_dir
        self.enable_gpu = enable_gpu and torch.cuda.is_available()
        self.monitoring = False
        self.memory_usage = []
        self.timestamps = []
        
        # 創建報告目錄
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
    
    def start(self):
        """開始記憶體監控"""
        if self.monitoring:
            logger.warning("記憶體監控已在運行中")
            return
        
        self.monitoring = True
        self.memory_usage = []
        self.timestamps = []
        
        logger.info(f"開始記憶體監控，間隔: {self.interval} 秒")
        
        # 在背景執行監控
        import threading
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop(self):
        """停止記憶體監控"""
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
    
    def _monitor_loop(self):
        """監控循環"""
        while self.monitoring:
            # 獲取當前記憶體使用情況
            cpu_usage = self._get_cpu_memory_usage()
            self.memory_usage.append(cpu_usage)
            self.timestamps.append(datetime.now())
            
            # 記錄日誌
            if len(self.memory_usage) % 10 == 0:
                logger.info(f"當前記憶體使用: {cpu_usage:.2f} MB")
                
                if self.enable_gpu:
                    gpu_usage = self._get_gpu_memory_usage()
                    logger.info(f"當前 GPU 記憶體使用: {gpu_usage:.2f} MB")
            
            # 等待下一次監控
            time.sleep(self.interval)
    
    def _get_cpu_memory_usage(self):
        """獲取 CPU 記憶體使用情況"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # 轉換為 MB
    
    def _get_gpu_memory_usage(self):
        """獲取 GPU 記憶體使用情況"""
        if not self.enable_gpu:
            return 0
        
        try:
            return torch.cuda.memory_allocated() / (1024 * 1024)  # 轉換為 MB
        except:
            return 0
    
    def generate_report(self):
        """生成記憶體使用報告"""
        if not self.memory_usage:
            logger.warning("沒有記憶體使用數據，無法生成報告")
            return
        
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
                "final": self.memory_usage[-1]
            },
            "timestamps": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in self.timestamps],
            "memory_values": self.memory_usage
        }
        
        # 保存報告
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=4)
        
        logger.info(f"記憶體使用報告已保存至: {report_file}")
        
        # 繪製記憶體使用圖表
        plt.figure(figsize=(12, 6))
        plt.plot(self.timestamps, self.memory_usage)
        plt.xlabel('時間')
        plt.ylabel('記憶體使用 (MB)')
        plt.title('記憶體使用趨勢')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plot_file)
        
        logger.info(f"記憶體使用圖表已保存至: {plot_file}")
        
        # 返回報告摘要
        return {
            "min": min(self.memory_usage),
            "max": max(self.memory_usage),
            "avg": sum(self.memory_usage) / len(self.memory_usage),
            "final": self.memory_usage[-1]
        }

def memory_mapped_array(shape, dtype=np.float32, filename=None, mode='w+'):
    """
    創建記憶體映射數組
    
    參數:
        shape (tuple): 數組形狀
        dtype (np.dtype): 數據類型
        filename (str, optional): 文件名，如果為 None 則使用臨時文件
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
    
    logger.info(f"創建記憶體映射數組: {filename}, 形狀: {shape}, 類型: {dtype}")
    
    return memmap_array

def save_dataframe_chunked(df, filename, chunk_size=10000, compression=None):
    """
    分塊保存 DataFrame 到 CSV 文件
    
    參數:
        df (pd.DataFrame): 數據框
        filename (str): 文件名
        chunk_size (int): 每塊的行數
        compression (str, optional): 壓縮格式，如 'gzip', 'bz2', 'xz', 'zip'
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
        
        chunk.to_csv(filename, mode=mode, header=header, index=False, compression=compression)
        
        # 釋放記憶體
        del chunk
        gc.collect()
        
        logger.info(f"已保存塊 {i+1}/{n_chunks}, 行數: {end_idx - start_idx}")

def load_dataframe_chunked(filename, chunk_size=10000, compression=None, dtype=None):
    """
    分塊加載 CSV 文件到 DataFrame
    
    參數:
        filename (str): 文件名
        chunk_size (int): 每塊的行數
        compression (str, optional): 壓縮格式，如 'gzip', 'bz2', 'xz', 'zip'
        dtype (dict, optional): 列數據類型
        
    返回:
        pd.DataFrame: 合併後的數據框
    """
    logger.info(f"分塊加載 DataFrame: {filename}, 塊大小: {chunk_size}")
    
    # 使用 pandas 的 chunked 讀取
    chunks = []
    for chunk in pd.read_csv(filename, chunksize=chunk_size, compression=compression, dtype=dtype, low_memory=False):
        chunks.append(chunk)
        logger.info(f"已加載塊，行數: {len(chunk)}")
    
    # 合併所有塊
    df = pd.concat(chunks, ignore_index=True)
    
    logger.info(f"完成加載，總行數: {len(df)}")
    
    return df

def optimize_dataframe_memory(df, category_threshold=50, verbose=True):
    """
    優化 DataFrame 記憶體使用
    
    參數:
        df (pd.DataFrame): 數據框
        category_threshold (int): 唯一值數量閾值，低於此值的列將轉換為分類類型
        verbose (bool): 是否輸出詳細信息
        
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
        if col_type in [np.int64, np.int32]:
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
        elif col_type in [np.float64, np.float32]:
            df_optimized[col] = df_optimized[col].astype(np.float32)
        
        # 字符串類型優化
        elif col_type == object:
            # 檢查是否可以轉換為分類類型
            n_unique = df_optimized[col].nunique()
            if n_unique < category_threshold:
                df_optimized[col] = df_optimized[col].astype('category')
    
    # 計算優化後的記憶體使用
    end_mem = df_optimized.memory_usage(deep=True).sum() / (1024 * 1024)
    reduction = 100 * (start_mem - end_mem) / start_mem
    
    if verbose:
        logger.info(f"優化後 DataFrame 記憶體使用: {end_mem:.2f} MB")
        logger.info(f"記憶體減少: {reduction:.2f}%")
    
    return df_optimized

def clean_memory():
    """主動清理記憶體"""
    # 強制執行垃圾回收
    gc.collect()
    
    # 如果使用 CUDA，清理 CUDA 緩存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("已執行記憶體清理")

def limit_gpu_memory(limit_mb=2048):
    """
    限制 GPU 記憶體使用
    
    參數:
        limit_mb (int): 記憶體限制 (MB)，0 表示不限制
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA 不可用，無法限制 GPU 記憶體")
        return
    
    if limit_mb <= 0:
        logger.info("不限制 GPU 記憶體使用")
        return
    
    try:
        # 設置 CUDA 記憶體分配器
        torch.cuda.set_per_process_memory_fraction(limit_mb / torch.cuda.get_device_properties(0).total_memory)
        logger.info(f"已限制 GPU 記憶體使用: {limit_mb} MB")
    except Exception as e:
        logger.error(f"限制 GPU 記憶體時發生錯誤: {str(e)}")

def memory_usage_decorator(func):
    """
    記憶體使用裝飾器
    
    參數:
        func: 要監控的函數
        
    返回:
        callable: 包裝後的函數
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 記錄開始時的記憶體使用
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)
        
        # 執行函數
        result = func(*args, **kwargs)
        
        # 記錄結束時的記憶體使用
        end_mem = process.memory_info().rss / (1024 * 1024)
        
        # 計算記憶體使用變化
        delta_mem = end_mem - start_mem
        
        logger.info(f"{func.__name__} 記憶體使用: {delta_mem:.2f} MB (開始: {start_mem:.2f} MB, 結束: {end_mem:.2f} MB)")
        
        return result
    
    return wrapper

def get_memory_usage():
    """
    獲取當前記憶體使用情況
    
    返回:
        dict: 記憶體使用信息
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
        "gpu_memory_mb": gpu_mem
    }

def print_memory_usage():
    """打印當前記憶體使用情況"""
    mem_info = get_memory_usage()
    
    logger.info("當前記憶體使用情況:")
    logger.info(f"  CPU 記憶體: {mem_info['cpu_memory_mb']:.2f} MB")
    logger.info(f"  系統記憶體: {mem_info['system_memory_used_mb']:.2f} MB / {mem_info['system_memory_total_mb']:.2f} MB ({mem_info['system_memory_percent']:.1f}%)")
    
    if torch.cuda.is_available():
        logger.info(f"  GPU 記憶體: {mem_info['gpu_memory_mb']:.2f} MB")

def get_memory_optimization_suggestions():
    """
    獲取記憶體優化建議
    
    返回:
        list: 優化建議列表
    """
    suggestions = []
    
    # 獲取記憶體使用情況
    mem_info = get_memory_usage()
    
    # 系統記憶體使用率高
    if mem_info['system_memory_percent'] > 80:
        suggestions.append("系統記憶體使用率高，建議減小批次大小或使用記憶體映射")
    
    # GPU 記憶體使用高
    if torch.cuda.is_available() and mem_info['gpu_memory_mb'] > 1000:
        suggestions.append("GPU 記憶體使用較高，建議使用混合精度訓練或梯度檢查點")
    
    # 一般建議
    suggestions.append("使用 optimize_dataframe_memory() 優化 DataFrame 記憶體使用")
    suggestions.append("定期調用 clean_memory() 清理未使用的記憶體")
    suggestions.append("考慮使用記憶體映射文件處理大型數據集")
    
    return suggestions

def print_optimization_suggestions():
    """打印記憶體優化建議"""
    suggestions = get_memory_optimization_suggestions()
    
    logger.info("記憶體優化建議:")
    for i, suggestion in enumerate(suggestions, 1):
        logger.info(f"  {i}. {suggestion}")

# 主程式測試
if __name__ == "__main__":
    # 測試記憶體監控
    monitor = MemoryMonitor(interval=1, report_dir='./test_memory_reports')
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
    
    df_optimized = optimize_dataframe_memory(df)
    
    # 測試記憶體使用裝飾器
    @memory_usage_decorator
    def memory_intensive_function():
        # 分配一些記憶體
        arr = np.random.randn(2000, 2000)
        time.sleep(1)
        return arr.mean()
    
    result = memory_intensive_function()
    
    # 測試記憶體使用情況打印
    print_memory_usage()
    
    # 測試優化建議
    print_optimization_suggestions()
    
    # 清理記憶體
    clean_memory()
