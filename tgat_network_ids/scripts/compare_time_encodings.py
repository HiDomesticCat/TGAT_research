#!/usr/bin/env python
# coding: utf-8 -*-

"""
時間編碼方法比較腳本

比較不同時間編碼方法對TGAT模型性能的影響，包括記憶體使用、訓練速度與準確度。
支持的編碼方法：
1. 記憶體高效餘弦編碼 (默認)
2. 可學習時間嵌入
3. Time2Vec編碼
4. 傅立葉編碼
"""

import os
import sys
import yaml
import argparse
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import logging
from datetime import datetime

# 確保可以導入專案模組
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# 導入專案模組
from src.data.optimized_data_loader import EnhancedMemoryOptimizedDataLoader
from src.models.optimized_tgat_model import OptimizedTGATModel
from src.models.time_encoding import TimeEncodingFactory
from src.utils.memory_utils import print_memory_usage, track_memory_usage

# 配置日誌
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(f"time_encoding_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description="比較不同時間編碼方法的性能")
    
    # 基本參數
    parser.add_argument('--config', type=str, default='config/memory_optimized_config.yaml',
                        help='配置文件路徑')
    parser.add_argument('--data_path', type=str, default=None,
                        help='數據路徑，覆蓋配置文件中的設定')
    parser.add_argument('--use_gpu', action='store_true', default=None,
                        help='使用GPU，覆蓋配置文件設定')
    parser.add_argument('--epochs', type=int, default=20,
                        help='訓練輪數')
    
    # 時間編碼相關參數
    parser.add_argument('--encoding_types', type=str, default='memory_efficient,learnable,time2vec,fourier',
                        help='要比較的時間編碼方法，以逗號分隔')
    parser.add_argument('--encoding_dim', type=int, default=64,
                        help='時間編碼維度')
    parser.add_argument('--use_best_encoding', action='store_true',
                        help='使用比較後表現最佳的編碼方法進行最終訓練')
    
    # 輸出相關
    parser.add_argument('--output_dir', type=str, default=None,
                        help='輸出目錄，若未指定則使用時間戳建立')
    
    return parser.parse_args()

def load_config(config_path):
    """載入配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_output_dirs(args):
    """設置輸出目錄"""
    # 創建輸出基目錄
    if args.output_dir:
        base_output_dir = args.output_dir
    else:
        # 使用時間戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_output_dir = os.path.join(project_root, f'time_encoding_comparison_{timestamp}')
    
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 為每種編碼方法創建子目錄
    encoding_types = args.encoding_types.split(',')
    encoding_dirs = {}
    
    for enc_type in encoding_types:
        enc_dir = os.path.join(base_output_dir, enc_type)
        os.makedirs(enc_dir, exist_ok=True)
        
        # 創建模型、結果和可視化目錄
        os.makedirs(os.path.join(enc_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(enc_dir, 'results'), exist_ok=True)
        os.makedirs(os.path.join(enc_dir, 'plots'), exist_ok=True)
        
        encoding_dirs[enc_type] = enc_dir
    
    # 創建比較結果目錄
    comparison_dir = os.path.join(base_output_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    return base_output_dir, encoding_dirs, comparison_dir

@track_memory_usage
def train_and_evaluate(config, encoding_type, encoding_dim, output_dir, device, epochs=20):
    """使用指定的時間編碼方法訓練和評估模型
    
    參數:
        config (dict): 配置字典
        encoding_type (str): 時間編碼類型
        encoding_dim (int): 時間編碼維度
        output_dir (str): 輸出目錄
        device (torch.device): 訓練設備
        epochs (int): 訓練輪數
        
    返回:
        dict: 包含訓練結果和性能指標
    """
    logger.info(f"使用 {encoding_type} 編碼開始訓練...")
    
    # 配置時間編碼
    config['time_encoding'] = {
        'method': encoding_type,
        'dimension': encoding_dim
    }
    
    # 配置額外的編碼參數
    if encoding_type == 'fourier':
        config['time_encoding']['fourier_freqs'] = 32
        config['time_encoding']['learnable_freqs'] = True
    elif encoding_type == 'learnable':
        config['time_encoding']['num_bins'] = 2000
        config['time_encoding']['max_period'] = 10000.0
    
    # 記錄開始時間
    start_time = time.time()
    
    # 初始化數據載入器
    data_loader = EnhancedMemoryOptimizedDataLoader(config)
    
    # 載入數據
    data_loader.load_data()
    
    # 預處理數據
    features, target = data_loader.preprocess()
    
    # 拆分數據
    X_train, X_test, y_train, y_test = data_loader.split_data()
    
    # 初始化TGAT模型
    model_config = config.copy()
    model_config['model']['node_features'] = features.shape[1]
    model_config['model']['num_classes'] = len(np.unique(target))
    
    # 訓練配置
    model_config['training']['epochs'] = epochs
    
    # 初始化模型
    model = OptimizedTGATModel(model_config)
    model.to(device)
    
    # 訓練模型並計時
    train_start = time.time()
    training_stats = model.train_model(X_train, y_train, X_test, y_test, device)
    train_end = time.time()
    
    # 評估模型
    evaluation_results = model.evaluate(X_test, y_test)
    
    # 計算總時間
    total_time = time.time() - start_time
    training_time = train_end - train_start
    
    # 獲取最大記憶體使用量
    peak_memory = torch.cuda.max_memory_allocated(device) // (1024 * 1024) if torch.cuda.is_available() else 0
    
    # 記錄結果
    results = {
        'encoding_type': encoding_type,
        'encoding_dim': encoding_dim,
        'total_time': total_time,
        'training_time': training_time,
        'peak_memory_mb': peak_memory,
        'converged_epoch': len(training_stats['loss']),
        'final_loss': training_stats['loss'][-1] if training_stats['loss'] else None,
        'best_val_accuracy': max(training_stats['val_accuracy']) if training_stats['val_accuracy'] else None,
        'test_accuracy': evaluation_results.get('accuracy', 0),
        'test_f1_score': evaluation_results.get('f1_score', 0),
        'evaluation_results': evaluation_results,
        'training_history': training_stats
    }
    
    # 保存模型
    model_path = os.path.join(output_dir, 'models', f"model_{encoding_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
    model.save(model_path)
    logger.info(f"模型已保存至: {model_path}")
    
    # 保存結果
    results_path = os.path.join(output_dir, 'results', f"results_{encoding_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"結果已保存至: {results_path}")
    
    # 繪製學習曲線
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(training_stats['loss'], label='Training Loss')
    plt.plot(training_stats['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{encoding_type} Time Encoding - Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(training_stats['accuracy'], label='Training Accuracy')
    plt.plot(training_stats['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{encoding_type} Time Encoding - Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'plots', f"learning_curves_{encoding_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_path)
    plt.close()
    
    logger.info(f"完成 {encoding_type} 編碼訓練與評估")
    logger.info(f"測試準確率: {results['test_accuracy']:.4f}, F1分數: {results['test_f1_score']:.4f}")
    logger.info(f"訓練時間: {training_time:.2f} 秒, 峰值記憶體: {peak_memory} MB")
    
    return results

def visualize_comparison(all_results, comparison_dir):
    """視覺化不同編碼方法的比較"""
    # 提取關鍵指標
    encoding_types = []
    accuracies = []
    f1_scores = []
    train_times = []
    memory_usages = []
    
    for result in all_results:
        encoding_types.append(result['encoding_type'])
        accuracies.append(result['test_accuracy'])
        f1_scores.append(result['test_f1_score'])
        train_times.append(result['training_time'])
        memory_usages.append(result['peak_memory_mb'])
    
    # 繪製準確率和F1分數比較
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    bars = plt.bar(encoding_types, accuracies, color='skyblue')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy by Encoding Method')
    plt.ylim(0, 1.0)
    # 添加值標籤
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.subplot(2, 2, 2)
    bars = plt.bar(encoding_types, f1_scores, color='lightgreen')
    plt.ylabel('F1 Score')
    plt.title('Test F1 Score by Encoding Method')
    plt.ylim(0, 1.0)
    # 添加值標籤
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.subplot(2, 2, 3)
    bars = plt.bar(encoding_types, train_times, color='salmon')
    plt.ylabel('Seconds')
    plt.title('Training Time by Encoding Method')
    # 添加值標籤
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s', ha='center', va='bottom')
    
    plt.subplot(2, 2, 4)
    bars = plt.bar(encoding_types, memory_usages, color='plum')
    plt.ylabel('MB')
    plt.title('Peak Memory Usage by Encoding Method')
    # 添加值標籤
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}MB', ha='center', va='bottom')
    
    plt.tight_layout()
    comparison_path = os.path.join(comparison_dir, f"encoding_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(comparison_path)
    plt.close()
    
    logger.info(f"比較視覺化已保存至: {comparison_path}")
    
    # 繪製學習曲線對比
    plt.figure(figsize=(15, 6))
    
    # 損失曲線
    plt.subplot(1, 2, 1)
    for result in all_results:
        encoding_type = result['encoding_type']
        val_loss = result['training_history']['val_loss']
        epochs = list(range(1, len(val_loss) + 1))
        plt.plot(epochs, val_loss, label=encoding_type, marker='o', markersize=4)
    
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 準確率曲線
    plt.subplot(1, 2, 2)
    for result in all_results:
        encoding_type = result['encoding_type']
        val_acc = result['training_history']['val_accuracy']
        epochs = list(range(1, len(val_acc) + 1))
        plt.plot(epochs, val_acc, label=encoding_type, marker='o', markersize=4)
    
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    curves_path = os.path.join(comparison_dir, f"learning_curves_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(curves_path)
    plt.close()
    
    logger.info(f"學習曲線比較已保存至: {curves_path}")
    
    # 創建比較表格
    comparison_df = pd.DataFrame({
        'Encoding Method': encoding_types,
        'Test Accuracy': accuracies,
        'Test F1 Score': f1_scores,
        'Training Time (s)': train_times,
        'Peak Memory (MB)': memory_usages
    })
    
    # 保存比較表
    table_path = os.path.join(comparison_dir, f"encoding_comparison_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    comparison_df.to_csv(table_path, index=False)
    logger.info(f"比較表已保存至: {table_path}")
    
    # 返回表現最好的編碼方法
    best_idx = np.argmax(accuracies)
    return encoding_types[best_idx]

@track_memory_usage
def main():
    """主函數"""
    # 解析命令行參數
    args = parse_args()
    
    # 載入配置
    config = load_config(args.config)
    
    # 設置輸出目錄
    base_output_dir, encoding_dirs, comparison_dir = setup_output_dirs(args)
    
    # 命令行參數覆蓋配置
    if args.data_path:
        config['data']['path'] = args.data_path
    if args.use_gpu is not None:
        config['model']['use_gpu'] = args.use_gpu
    
    # 檢查是否使用GPU
    use_gpu = config['model'].get('use_gpu', False) and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    logger.info(f"使用設備: {device}")
    
    # 獲取要比較的編碼類型
    encoding_types = args.encoding_types.split(',')
    
    # 驗證編碼類型
    valid_encodings = ['memory_efficient', 'learnable', 'time2vec', 'fourier']
    for enc_type in encoding_types:
        if enc_type not in valid_encodings:
            logger.warning(f"未知的編碼類型: {enc_type}，將被忽略")
            encoding_types.remove(enc_type)
    
    if not encoding_types:
        logger.error("沒有有效的編碼類型，退出")
        return
    
    # 比較不同的編碼方法
    all_results = []
    
    for enc_type in encoding_types:
        # 對每種編碼方法進行訓練和評估
        result = train_and_evaluate(
            config=config,
            encoding_type=enc_type,
            encoding_dim=args.encoding_dim,
            output_dir=encoding_dirs[enc_type],
            device=device,
            epochs=args.epochs
        )
        
        all_results.append(result)
        
        # 釋放記憶體
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 視覺化比較結果
    best_encoding = visualize_comparison(all_results, comparison_dir)
    
    # 輸出比較摘要
    logger.info("=" * 50)
    logger.info("時間編碼方法比較摘要:")
    logger.info("-" * 50)
    
    performance_summary = {}
    for result in all_results:
        enc_type = result['encoding_type']
        performance_summary[enc_type] = {
            'accuracy': result['test_accuracy'],
            'f1_score': result['test_f1_score'],
            'training_time': result['training_time'],
            'memory_usage': result['peak_memory_mb']
        }
        
        logger.info(f"編碼方法: {enc_type}")
        logger.info(f"  測試準確率: {result['test_accuracy']:.4f}")
        logger.info(f"  測試F1分數: {result['test_f1_score']:.4f}")
        logger.info(f"  訓練時間: {result['training_time']:.2f} 秒")
        logger.info(f"  峰值記憶體使用: {result['peak_memory_mb']} MB")
        logger.info("-" * 50)
    
    logger.info(f"表現最佳的編碼方法: {best_encoding}")
    logger.info("=" * 50)
    
    # 保存比較摘要
    summary_path = os.path.join(comparison_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'best_encoding': best_encoding,
            'performance_summary': performance_summary
        }, f, indent=2)
    
    # 如果指定了使用最佳編碼方法進行最終訓練
    if args.use_best_encoding:
        logger.info(f"使用最佳編碼方法 {best_encoding} 進行最終訓練...")
        
        # 最終訓練使用更多輪次
        final_epochs = args.epochs * 2
        
        final_output_dir = os.path.join(base_output_dir, 'final_model')
        os.makedirs(final_output_dir, exist_ok=True)
        os.makedirs(os.path.join(final_output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(final_output_dir, 'results'), exist_ok=True)
        os.makedirs(os.path.join(final_output_dir, 'plots'), exist_ok=True)
        
        # 使用最佳編碼方法進行最終訓練
        final_result = train_and_evaluate(
            config=config,
            encoding_type=best_encoding,
            encoding_dim=args.encoding_dim,
            output_dir=final_output_dir,
            device=device,
            epochs=final_epochs
        )
        
        logger.info("=" * 50)
        logger.info(f"最終模型 ({best_encoding} 編碼) 性能:")
        logger.info(f"  測試準確率: {final_result['test_accuracy']:.4f}")
        logger.info(f"  測試F1分數: {final_result['test_f1_score']:.4f}")
        logger.info("=" * 50)
    
    logger.info(f"所有結果已保存至: {base_output_dir}")
    
    return {
        'best_encoding': best_encoding,
        'performance_summary': performance_summary
    }

if __name__ == "__main__":
    results = main()
