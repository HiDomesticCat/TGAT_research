#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
工具函數模組

此模組包含:
1. 配置加載與保存
2. 性能評估指標
3. 資料預處理工具
4. 其他輔助功能
"""

import os
import yaml
import json
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import logging
import time
from datetime import datetime
import matplotlib.pyplot as plt
import random

# 設置日誌
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed):
    """
    設置隨機種子以確保可重現性
    
    參數:
        seed (int): 隨機種子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"已設置隨機種子: {seed}")

def get_device():
    """
    獲取可用的計算裝置
    
    返回:
        str: 裝置名稱 ('cuda' 或 'cpu')
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用裝置: {device}")
    if device == 'cuda':
        logger.info(f"CUDA 裝置: {torch.cuda.get_device_name(0)}")
    return device

def load_config(config_path):
    """
    載入YAML配置文件
    
    參數:
        config_path (str): 配置文件路徑
        
    返回:
        dict: 配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"已載入配置: {config_path}")
        return config
    except Exception as e:
        logger.error(f"載入配置時發生錯誤: {str(e)}")
        return {}

def save_config(config, config_path):
    """
    保存配置到YAML文件
    
    參數:
        config (dict): 配置字典
        config_path (str): 保存路徑
    """
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"已保存配置: {config_path}")
    except Exception as e:
        logger.error(f"保存配置時發生錯誤: {str(e)}")

def evaluate_predictions(y_true, y_pred, y_proba=None):
    """
    評估預測結果
    
    參數:
        y_true (array-like): 真實標籤
        y_pred (array-like): 預測標籤
        y_proba (array-like, optional): 預測概率 (用於ROC-AUC計算)
        
    返回:
        dict: 評估指標字典
    """
    metrics = {}
    
    # 基本分類指標
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # 混淆矩陣
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # 詳細分類報告
    metrics['report'] = classification_report(y_true, y_pred, zero_division=0)
    
    # 如果提供概率，計算 ROC-AUC
    if y_proba is not None:
        try:
            # 二分類情況
            if len(np.unique(y_true)) == 2:
                if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                    # 取正類別的概率
                    y_proba_binary = y_proba[:, 1]
                else:
                    y_proba_binary = y_proba
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba_binary)
            # 多分類情況
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
        except Exception as e:
            logger.warning(f"計算 ROC-AUC 時發生錯誤: {str(e)}")
    
    return metrics

def format_metrics(metrics):
    """
    格式化評估指標輸出
    
    參數:
        metrics (dict): 評估指標字典
        
    返回:
        str: 格式化的指標字符串
    """
    output = "模型評估指標:\n"
    output += f"準確率 (Accuracy): {metrics['accuracy']:.4f}\n"
    output += f"精確率 (Precision) - 巨集平均: {metrics['precision_macro']:.4f}\n"
    output += f"召回率 (Recall) - 巨集平均: {metrics['recall_macro']:.4f}\n"
    output += f"F1 分數 - 巨集平均: {metrics['f1_macro']:.4f}\n\n"
    
    output += f"精確率 (Precision) - 加權平均: {metrics['precision_weighted']:.4f}\n"
    output += f"召回率 (Recall) - 加權平均: {metrics['recall_weighted']:.4f}\n"
    output += f"F1 分數 - 加權平均: {metrics['f1_weighted']:.4f}\n"
    
    if 'roc_auc' in metrics:
        output += f"ROC-AUC: {metrics['roc_auc']:.4f}\n"
    
    output += "\n分類報告:\n"
    output += metrics['report']
    
    return output

def create_dir(directory):
    """
    創建目錄 (如果不存在)
    
    參數:
        directory (str): 目錄路徑
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"已創建目錄: {directory}")

def save_results(results, save_path):
    """
    保存結果到 JSON 文件
    
    參數:
        results (dict): 結果字典
        save_path (str): 保存路徑
    """
    try:
        # 轉換非序列化類型
        serializable_results = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                serializable_results[k] = v.tolist()
            elif isinstance(v, np.integer):
                serializable_results[k] = int(v)
            elif isinstance(v, np.floating):
                serializable_results[k] = float(v)
            else:
                serializable_results[k] = v
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=4)
        
        logger.info(f"已保存結果: {save_path}")
    except Exception as e:
        logger.error(f"保存結果時發生錯誤: {str(e)}")

def get_timestamp():
    """
    獲取當前時間戳記
    
    返回:
        str: 格式化的時間戳記
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def calculate_time_delta_features(df, ip_col, time_col, window_sizes=[1, 5, 10, 30, 60]):
    """
    計算時間差異特徵
    
    參數:
        df (pd.DataFrame): 輸入數據框
        ip_col (str): IP 列名
        time_col (str): 時間戳記列名
        window_sizes (list): 時間窗口大小列表 (秒)
        
    返回:
        pd.DataFrame: 加入時間差異特徵的數據框
    """
    # 確保時間列是 datetime 類型
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # 按 IP 和時間排序
    df = df.sort_values([ip_col, time_col]).reset_index(drop=True)
    
    # 對每個 IP，計算與前一個封包的時間差
    df['prev_time'] = df.groupby(ip_col)[time_col].shift(1)
    df['time_delta'] = (df[time_col] - df['prev_time']).dt.total_seconds()
    
    # 對於沒有前一個封包的情況，設為 NaN
    df['time_delta'] = df['time_delta'].fillna(-1)
    
    # 對每個時間窗口，計算窗口內的封包數
    for window in window_sizes:
        # 計算窗口結束時間
        df[f'window_{window}_end'] = df[time_col] + pd.Timedelta(seconds=window)
        
        # 創建一個臨時計數列
        df['tmp_count'] = 1
        
        # 計算每個窗口內的封包數
        result = []
        for ip in df[ip_col].unique():
            ip_df = df[df[ip_col] == ip].copy()
            counts = []
            
            for i, row in ip_df.iterrows():
                # 窗口結束時間
                window_end = row[f'window_{window}_end']
                
                # 計算窗口內的封包數
                count = ip_df[(ip_df[time_col] >= row[time_col]) & 
                              (ip_df[time_col] <= window_end)]['tmp_count'].sum()
                
                counts.append(count)
            
            ip_df[f'packet_count_{window}s'] = counts
            result.append(ip_df)
        
        df = pd.concat(result)
        
        # 刪除臨時列
        df = df.drop(['tmp_count', f'window_{window}_end'], axis=1)
    
    # 刪除臨時列
    df = df.drop(['prev_time'], axis=1)
    
    return df

def extract_packet_features(df, src_ip_col, dst_ip_col, time_col, features_to_aggregate=None):
    """
    提取網路封包特徵
    
    參數:
        df (pd.DataFrame): 輸入數據框
        src_ip_col (str): 源 IP 列名
        dst_ip_col (str): 目標 IP 列名
        time_col (str): 時間戳記列名
        features_to_aggregate (list, optional): 要聚合的特徵列表
        
    返回:
        pd.DataFrame: 包含聚合特徵的數據框
    """
    # 預設聚合特徵
    if features_to_aggregate is None:
        features_to_aggregate = [
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
            'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
            'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
            'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
            'Flow IAT Max', 'Flow IAT Min'
        ]
    
    # 檢查特徵是否存在
    existing_features = [f for f in features_to_aggregate if f in df.columns]
    logger.info(f"使用 {len(existing_features)}/{len(features_to_aggregate)} 個特徵進行聚合")
    
    # 確保時間列是 datetime 類型
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    
    # 按時間分組 (每分鐘)
    df['time_bin'] = df[time_col].dt.floor('1min')
    
    # 對每個 IP 對和時間段，計算特徵聚合值
    agg_dict = {}
    for feat in existing_features:
        if feat in df.columns:
            agg_dict[feat] = ['mean', 'max', 'min', 'std', 'count']
    
    # 按源 IP 聚合
    src_agg = df.groupby([src_ip_col, 'time_bin']).agg(agg_dict)
    src_agg.columns = [f'src_{feat}_{agg}' for feat, agg in src_agg.columns]
    src_agg = src_agg.reset_index()
    
    # 按目標 IP 聚合
    dst_agg = df.groupby([dst_ip_col, 'time_bin']).agg(agg_dict)
    dst_agg.columns = [f'dst_{feat}_{agg}' for feat, agg in dst_agg.columns]
    dst_agg = dst_agg.reset_index()
    
    # 按 IP 對聚合
    pair_agg = df.groupby([src_ip_col, dst_ip_col, 'time_bin']).agg(agg_dict)
    pair_agg.columns = [f'pair_{feat}_{agg}' for feat, agg in pair_agg.columns]
    pair_agg = pair_agg.reset_index()
    
    # 合併所有聚合
    result = df.merge(src_agg, on=[src_ip_col, 'time_bin'], how='left')
    result = result.merge(dst_agg, on=[dst_ip_col, 'time_bin'], how='left')
    result = result.merge(pair_agg, on=[src_ip_col, dst_ip_col, 'time_bin'], how='left')
    
    # 刪除時間分組列
    result = result.drop(['time_bin'], axis=1)
    
    return result

def plot_attack_distribution(df, label_col, save_path=None):
    """
    繪製攻擊分布圖
    
    參數:
        df (pd.DataFrame): 輸入數據框
        label_col (str): 標籤列名
        save_path (str, optional): 保存路徑
    """
    plt.figure(figsize=(12, 6))
    
    # 計算各類別數量
    value_counts = df[label_col].value_counts()
    
    # 繪製條形圖
    ax = value_counts.plot(kind='bar', color='skyblue')
    
    # 添加數值標籤
    for i, count in enumerate(value_counts):
        ax.text(i, count + (count * 0.01), f"{count}", ha='center')
    
    plt.title('攻擊類型分布')
    plt.xlabel('攻擊類型')
    plt.ylabel('數量')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"攻擊分布圖已保存: {save_path}")
    
    plt.show()

def time_execution(func):
    """
    函數執行時間裝飾器
    
    參數:
        func: 要計時的函數
        
    返回:
        callable: 包裝後的函數
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} 執行時間: {execution_time:.4f} 秒")
        return result
    return wrapper

# 主程式測試
if __name__ == "__main__":
    # 測試日誌功能
    logger.info("測試工具函數模組")
    
    # 測試裝置獲取
    device = get_device()
    print(f"計算裝置: {device}")
    
    # 測試隨機種子設置
    set_seed(42)
    
    # 測試簡單指標計算
    y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 0, 1, 1, 1]
    y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.6, 0.4], 
                      [0.7, 0.3], [0.3, 0.7], [0.4, 0.6], [0.2, 0.8]])[:, 1]
    
    metrics = evaluate_predictions(y_true, y_pred, y_proba)
    print(format_metrics(metrics))
    
    # 測試時間戳記生成
    print(f"當前時間戳記: {get_timestamp()}")
    
    # 測試計時裝飾器
    @time_execution
    def slow_function():
        time.sleep(1)
        return "完成"
    
    print(slow_function())