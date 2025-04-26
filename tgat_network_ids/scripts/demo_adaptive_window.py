#!/usr/bin/env python
# coding: utf-8 -*-

"""
時間自適應窗口機制演示腳本

此腳本演示如何使用自適應時間窗口機制處理不同類型的網路攻擊模式，
包括突發攻擊、週期性攻擊和低慢攻擊。
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

# 添加專案根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 導入自適應窗口管理器
from src.data.adaptive_window import AdaptiveWindowManager

# 配置日誌
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_synthetic_network_data(pattern_type='mixed', sample_count=1000):
    """生成合成網路流量數據用於演示
    
    參數:
        pattern_type: 攻擊模式類型 ('burst', 'periodic', 'low_slow', 'mixed')
        sample_count: 樣本數量
        
    返回:
        DataFrame: 包含時間戳和IP地址的合成數據
    """
    now = datetime.now()
    timestamps = []
    ip_sources = []
    ip_destinations = []
    
    if pattern_type == 'burst':
        # 生成突發攻擊模式 - 大量快速連續的事件，間隔幾次大間隔
        logger.info("生成突發攻擊模式數據")
        
        # 基本時間間隔 (秒)
        base_interval = 5
        
        # 模擬3-5次突發
        burst_count = random.randint(3, 5)
        
        samples_per_burst = sample_count // burst_count
        current_time = now
        
        for burst in range(burst_count):
            # 突發期
            for i in range(samples_per_burst):
                if i < samples_per_burst * 0.8:  # 突發期內的高頻部分
                    interval = random.uniform(0.01, 0.5)  # 10毫秒到500毫秒
                else:
                    interval = random.uniform(0.5, 2)  # 逐漸降低頻率
                
                current_time += timedelta(seconds=interval)
                timestamps.append(current_time)
                
                # 一致的來源IP和隨機目標IP
                src_ip = f"192.168.1.{burst+1}"
                dst_ip = f"10.0.0.{random.randint(1, 254)}"
                ip_sources.append(src_ip)
                ip_destinations.append(dst_ip)
            
            # 突發間的間隔
            if burst < burst_count - 1:
                quiet_interval = random.uniform(60, 300)  # 1-5分鐘的間隔
                current_time += timedelta(seconds=quiet_interval)
    
    elif pattern_type == 'periodic':
        # 生成週期性攻擊模式 - 以固定間隔發生的事件
        logger.info("生成週期性攻擊模式數據")
        
        # 定義基本週期 (以秒為單位)
        period = random.choice([30, 60, 120, 300])  # 30秒, 1分鐘, 2分鐘, 5分鐘
        logger.info(f"使用週期: {period}秒")
        
        # 添加少量噪聲
        noise_level = 0.1  # 10%的時間噪聲
        
        current_time = now
        for i in range(sample_count):
            # 添加週期性間隔
            noise = random.uniform(-period * noise_level, period * noise_level)
            interval = period + noise
            current_time += timedelta(seconds=interval)
            timestamps.append(current_time)
            
            # 生成具有重複模式的IP地址
            src_octets = [192, 168, (i % 5) + 1, (i % 10) + 1]
            dst_octets = [10, 0, (i % 8), (i % 20) + 100]
            
            src_ip = '.'.join(map(str, src_octets))
            dst_ip = '.'.join(map(str, dst_octets))
            
            ip_sources.append(src_ip)
            ip_destinations.append(dst_ip)
    
    elif pattern_type == 'low_slow':
        # 生成低慢攻擊模式 - 低頻率長時間的低調活動
        logger.info("生成低慢攻擊模式數據")
        
        # 初始時間間隔 (以分鐘為單位)
        base_interval = random.uniform(15, 60)  # 15-60分鐘
        
        # 隨機選擇幾個持續活動的來源IP
        active_ips = [f"172.16.{random.randint(1, 10)}.{random.randint(1, 100)}" for _ in range(3)]
        
        current_time = now
        for i in range(sample_count):
            # 變化的時間間隔，但總體保持較長
            jitter = random.uniform(-0.3, 0.5)  # -30% 到 +50% 的變化
            interval = base_interval * (1 + jitter)
            
            # 偶爾有更長的間隔
            if random.random() < 0.1:  # 10%的概率
                interval *= random.uniform(2, 5)  # 2-5倍的間隔
                
            current_time += timedelta(minutes=interval)
            timestamps.append(current_time)
            
            # 使用較少的來源IP，較多的目標IP
            src_ip = random.choice(active_ips)
            dst_ip = f"192.168.{random.randint(1, 10)}.{random.randint(1, 254)}"
            
            ip_sources.append(src_ip)
            ip_destinations.append(dst_ip)
    
    else:  # mixed pattern
        # 混合不同類型的攻擊模式
        logger.info("生成混合攻擊模式數據")
        
        # 生成每種類型的部分數據
        samples_burst = generate_synthetic_network_data('burst', sample_count // 3)
        samples_periodic = generate_synthetic_network_data('periodic', sample_count // 3)
        samples_low_slow = generate_synthetic_network_data('low_slow', sample_count // 3)
        
        # 組合數據
        combined_data = pd.concat([samples_burst, samples_periodic, samples_low_slow], ignore_index=True)
        # 按時間排序
        combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)
        
        # 僅取所需樣本數
        if len(combined_data) > sample_count:
            combined_data = combined_data.iloc[:sample_count]
            
        return combined_data
    
    # 創建DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'src_ip': ip_sources,
        'dst_ip': ip_destinations
    })
    
    # 添加一些基本特徵
    df['bytes'] = [random.randint(60, 1500) for _ in range(len(df))]
    df['protocol'] = [random.choice(['TCP', 'UDP', 'ICMP']) for _ in range(len(df))]
    df['flags'] = [random.choice(['SYN', 'ACK', 'SYN-ACK', 'FIN', 'RST']) for _ in range(len(df))]
    
    return df

def visualize_windows(timestamps, windows):
    """視覺化時間窗口
    
    參數:
        timestamps: 時間戳數組
        windows: 窗口列表，每個窗口為 (開始時間, 結束時間, 尺度名稱, 窗口大小)
    """
    # 轉換為相對時間 (小時)
    if isinstance(timestamps[0], (datetime, pd.Timestamp)):
        # 轉換為小時
        reference_time = min(timestamps)
        relative_times = [(t - reference_time).total_seconds() / 3600 for t in timestamps]
    else:
        # 假設已經是數值型
        reference_time = min(timestamps)
        relative_times = [(t - reference_time) / 3600 for t in timestamps]
    
    # 創建圖形
    plt.figure(figsize=(12, 6))
    
    # 繪製事件點
    plt.scatter(relative_times, [0.5] * len(relative_times), alpha=0.5, color='blue', label='事件')
    
    # 繪製窗口
    colors = {
        'micro': 'red',
        'small': 'orange',
        'medium': 'green',
        'large': 'purple',
        'macro': 'brown',
        'burst': 'magenta',
        'burst_context': 'pink',
        'periodic': 'cyan',
        'periodic_overlap': 'lightblue',
        'low_slow_large': 'darkgreen',
        'low_slow_medium': 'olive'
    }
    
    # 添加黑色實線表示數據持續時間
    plt.axhline(y=0.5, color='black', linestyle='-', alpha=0.3)
    
    # 按窗口類型組織數據以便於創建圖例
    windows_by_type = {}
    
    for start, end, scale, size in windows:
        # 轉換為相對時間
        rel_start = (start - reference_time) / 3600
        rel_end = (end - reference_time) / 3600
        
        # 為窗口類型分配顏色
        color = colors.get(scale, 'gray')
        
        # 繪製窗口
        plt.axvspan(rel_start, rel_end, alpha=0.15, color=color)
        
        # 添加到類型字典
        if scale not in windows_by_type:
            windows_by_type[scale] = (rel_start, rel_end, color)
    
    # 添加圖例
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor='blue', markersize=10, label='事件')]
    
    for scale, (_, _, color) in windows_by_type.items():
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.3, label=f'{scale}窗口'))
    
    plt.legend(handles=legend_elements, loc='upper right')
    
    # 設置軸標籤和標題
    plt.xlabel('相對時間 (小時)')
    plt.ylabel('事件')
    plt.title('自適應時間窗口視覺化')
    plt.grid(True, alpha=0.3)
    
    # 儲存圖形
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'adaptive_windows_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(output_path)
    
    logger.info(f"時間窗口視覺化已保存至: {output_path}")
    plt.close()

def main():
    """主函數：演示自適應時間窗口機制"""
    logger.info("開始自適應時間窗口演示...")
    
    # 配置自適應窗口管理器
    config = {
        'enabled_scales': ['micro', 'small', 'medium', 'large', 'macro'],
        'detect_bursts': True,
        'detect_periodic': True,
        'detect_low_slow': True
    }
    
    adaptive_mgr = AdaptiveWindowManager(config)
    
    # 演示幾種不同的攻擊模式
    pattern_types = ['burst', 'periodic', 'low_slow', 'mixed']
    
    for pattern in pattern_types:
        logger.info(f"\n處理 {pattern} 模式...")
        
        # 生成合成數據
        data = generate_synthetic_network_data(pattern, 200)
        
        # 提取時間戳
        timestamps = data['timestamp'].values
        
        # 可視化原始數據分佈
        plt.figure(figsize=(10, 4))
        
        # 將時間轉換為相對時間 (秒)
        reference_time = min(timestamps)
        relative_times = [(t - reference_time).total_seconds() for t in timestamps]
        
        plt.scatter(relative_times, [1] * len(relative_times), alpha=0.7)
        plt.title(f"{pattern.capitalize()} 模式時間分佈")
        plt.xlabel("相對時間 (秒)")
        plt.grid(True, alpha=0.3)
        
        # 儲存時間分佈圖
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
        os.makedirs(output_dir, exist_ok=True)
        dist_path = os.path.join(output_dir, f'time_distribution_{pattern}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(dist_path)
        plt.close()
        
        # 根據模式類型選擇窗口生成方法
        if pattern == 'mixed':
            # 混合模式：使用自動選擇的窗口
            optimal_scales = adaptive_mgr.select_optimal_windows(timestamps, data)
            windows = adaptive_mgr.generate_adaptive_windows(timestamps, optimal_scales)
        else:
            # 特定模式：使用專用窗口
            windows = adaptive_mgr.create_windows_for_pattern(timestamps, pattern)
        
        # 視覺化窗口
        visualize_windows(timestamps, windows)
        
        logger.info(f"共生成 {len(windows)} 個窗口用於 {pattern} 模式")
        scale_counts = {}
        for _, _, scale, _ in windows:
            scale_counts[scale] = scale_counts.get(scale, 0) + 1
            
        for scale, count in scale_counts.items():
            logger.info(f"  - {scale}: {count} 個窗口")
    
    logger.info("自適應時間窗口演示完成")

if __name__ == "__main__":
    main()
