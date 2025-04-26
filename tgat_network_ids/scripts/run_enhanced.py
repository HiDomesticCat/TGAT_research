#!/usr/bin/env python
# coding: utf-8 -*-

"""
增強版TGAT網路入侵檢測系統執行腳本

整合了所有記憶體優化、自適應時間窗口和進階圖採樣等功能的完整執行腳本。
支持以下增強功能：
1. 記憶體優化的資料載入和預處理
2. 自適應多尺度時間窗口選擇
3. 先進圖採樣策略(GraphSAINT, Cluster-GCN等)
4. IP地址結構特徵提取
5. 統計特徵選擇
"""

import os
import sys
import yaml
import logging
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
import json
import dgl

# 確保可以導入專案模組
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# 導入專案模組
from src.data.optimized_data_loader import EnhancedMemoryOptimizedDataLoader
from src.data.optimized_graph_builder import OptimizedGraphBuilder
from src.models.optimized_tgat_model import OptimizedTGATModel
from src.data.adaptive_window import AdaptiveWindowManager
from src.data.advanced_sampling import AdvancedGraphSampler
from src.utils.memory_utils import print_memory_usage, track_memory_usage

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"tgat_ids_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='增強版TGAT網路入侵檢測系統')
    
    # 基本參數
    parser.add_argument('--config', type=str, default='config/memory_optimized_config.yaml',
                        help='配置文件路徑')
    parser.add_argument('--data_path', type=str, default=None,
                        help='資料路徑，覆蓋配置文件中的設定')
    parser.add_argument('--use_gpu', action='store_true', default=None,
                        help='使用GPU，覆蓋配置文件中的設定')
    parser.add_argument('--epochs', type=int, default=None,
                        help='訓練輪數，覆蓋配置文件中的設定')
    
    # 增強功能相關參數
    parser.add_argument('--use_adaptive_window', action='store_true', default=True,
                        help='使用自適應時間窗口')
    parser.add_argument('--adaptive_window_config', type=str, default=None,
                        help='自適應窗口配置文件路徑')
    
    parser.add_argument('--use_advanced_sampling', action='store_true', default=True,
                        help='使用進階圖採樣策略')
    parser.add_argument('--sampling_method', type=str, default='graphsaint',
                        choices=['graphsaint', 'cluster-gcn', 'frontier', 'historical'],
                        help='圖採樣方法')
    parser.add_argument('--sample_size', type=int, default=5000,
                        help='採樣子圖大小')
    
    parser.add_argument('--use_memory', action='store_true', default=True,
                        help='啟用記憶機制')
    parser.add_argument('--memory_size', type=int, default=1000,
                        help='記憶緩衝區大小')
    
    parser.add_argument('--use_position_embedding', action='store_true', default=True,
                        help='使用位置嵌入')
    
    return parser.parse_args()

def load_config(config_path):
    """載入配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_output_dirs(config):
    """設置輸出目錄"""
    # 創建模型保存目錄
    model_dir = config.get('model', {}).get('save_dir', './models')
    os.makedirs(model_dir, exist_ok=True)
    
    # 創建結果保存目錄
    results_dir = config.get('evaluation', {}).get('results_dir', './results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 創建視覺化保存目錄
    vis_dir = config.get('visualization', {}).get('save_dir', './visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 創建輸出目錄
    output_dir = os.path.join(project_root, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    return model_dir, results_dir, vis_dir, output_dir

# @track_memory_usage  # 原裝飾器（已禁用）
def main():
    """主函數"""
    # 解析命令行參數
    args = parse_args()
    
    # 載入配置
    config = load_config(args.config)
    
    # 命令行參數覆蓋配置文件
    if args.data_path:
        config['data']['path'] = args.data_path
    if args.use_gpu is not None:
        config['model']['use_gpu'] = args.use_gpu
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    # 設置輸出目錄
    model_dir, results_dir, vis_dir, output_dir = setup_output_dirs(config)
    
    # 記錄配置
    logger.info(f"使用配置文件: {args.config}")
    logger.info(f"命令行參數: {args}")
    
    # 記錄增強功能狀態
    logger.info(f"自適應時間窗口: {args.use_adaptive_window}")
    logger.info(f"進階圖採樣策略: {args.use_advanced_sampling} (方法: {args.sampling_method})")
    logger.info(f"記憶機制: {args.use_memory} (緩衝區大小: {args.memory_size})")
    logger.info(f"位置嵌入: {args.use_position_embedding}")
    
    # 檢查是否使用GPU
    use_gpu = config['model'].get('use_gpu', False) and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    logger.info(f"使用裝置: {device}")
    
    try:
        # 記錄開始時間
        start_time = time.time()
        
        # 初始化記憶體優化版資料載入器
        logger.info("初始化資料載入器...")
        data_loader = EnhancedMemoryOptimizedDataLoader(config)
        
        # 載入資料
        logger.info("載入資料...")
        data_loader.load_data()
        
        # 預處理資料
        logger.info("預處理資料...")
        features, target = data_loader.preprocess()
        
        # 拆分資料
        logger.info("拆分訓練集和測試集...")
        X_train, X_test, y_train, y_test = data_loader.split_data()
        
        # 設置自適應窗口管理器
        if args.use_adaptive_window:
            logger.info("初始化自適應窗口管理器...")
            # 載入自適應窗口配置
            if args.adaptive_window_config:
                with open(args.adaptive_window_config, 'r') as f:
                    window_config = yaml.safe_load(f)
            else:
                # 使用默認配置
                window_config = {
                    'enabled_scales': ['micro', 'small', 'medium', 'large'],
                    'detect_bursts': True,
                    'detect_periodic': True,
                    'detect_low_slow': True
                }
                
            window_manager = AdaptiveWindowManager(window_config)
        else:
            window_manager = None
        
        # 設置進階圖採樣器
        if args.use_advanced_sampling:
            logger.info(f"初始化進階圖採樣器 (方法: {args.sampling_method})...")
            sampling_config = {
                'sampling_method': args.sampling_method,
                'sample_size': args.sample_size,
                'use_memory': args.use_memory,
                'memory_size': args.memory_size,
                'use_position_embedding': args.use_position_embedding
            }
            graph_sampler = AdvancedGraphSampler(sampling_config)
        else:
            graph_sampler = None
        
        # 建立圖
        logger.info("建立網路流量圖...")
        graph_builder = OptimizedGraphBuilder(config)
        
        # 提取時間戳和事件時間
        try:
            # 嘗試獲取時間特徵
            time_cols = [col for col in features.columns if any(kw in col.lower() for kw in ['time', 'timestamp', 'date'])]
            if time_cols:
                timestamps = features[time_cols[0]].values
                logger.info(f"使用時間列: {time_cols[0]}")
            else:
                # 如果沒有找到，使用索引作為時間
                timestamps = np.arange(len(features))
                logger.info("未找到時間列，使用索引作為時間")
        except Exception as e:
            logger.warning(f"提取時間戳時出錯: {str(e)}，使用索引作為時間")
            timestamps = np.arange(len(features))
        
        # 使用自適應時間窗口(如果啟用)
        if args.use_adaptive_window and window_manager:
            logger.info("選擇最佳時間窗口...")
            optimal_scales = window_manager.select_optimal_windows(timestamps, features)
            logger.info(f"選定的窗口尺度: {optimal_scales}")
            
            # 生成時間窗口
            windows = window_manager.generate_adaptive_windows(timestamps, optimal_scales)
            logger.info(f"生成 {len(windows)} 個時間窗口")
            
            # 按窗口類型統計數量
            window_stats = {}
            for _, _, scale, _ in windows:
                window_stats[scale] = window_stats.get(scale, 0) + 1
            
            for scale, count in window_stats.items():
                logger.info(f"  - {scale}: {count} 個窗口")
                
            # 保存窗口信息
            window_info = {
                'optimal_scales': optimal_scales,
                'window_count': len(windows),
                'window_stats': window_stats
            }
            
            with open(os.path.join(output_dir, 'window_info.json'), 'w') as f:
                json.dump(window_info, f, indent=2)
        else:
            # 不使用自適應窗口，使用整個時間範圍
            windows = [(min(timestamps), max(timestamps), 'full', max(timestamps) - min(timestamps))]
        
        # 處理每個時間窗口
        all_graphs = []
        all_labels = []
        all_features = []
        
        for win_idx, (start_time, end_time, scale, win_size) in enumerate(windows):
            logger.info(f"處理窗口 {win_idx+1}/{len(windows)} ({scale}, 大小: {win_size:.2f}s)...")
            
            # 獲取窗口內的數據索引
            in_window = (timestamps >= start_time) & (timestamps <= end_time)
            window_indices = np.where(in_window)[0]
            
            if len(window_indices) < 5:  # 跳過節點太少的窗口
                logger.warning(f"窗口 {win_idx+1} 數據點太少 ({len(window_indices)})，跳過")
                continue
                
            # 獲取窗口內的特徵和標籤
            window_features = features.iloc[window_indices].copy()
            window_labels = target.iloc[window_indices].copy()
            
            # 建立窗口內的圖結構
            window_graph = graph_builder.build_graph(window_features)
            
            # 使用進階圖採樣(如果啟用)
            if args.use_advanced_sampling and graph_sampler and len(window_indices) > args.sample_size:
                logger.info(f"對窗口 {win_idx+1} 執行圖採樣 (原始節點: {len(window_indices)})...")
                # 使用進階採樣策略
                window_time_range = (start_time, end_time)
                sampled_graph, sampled_features = graph_sampler.sample_subgraph(
                    window_graph, 
                    window_features, 
                    timestamps[window_indices], 
                    window_time_range
                )
                
                # 更新窗口圖和特徵
                window_graph = sampled_graph
                if isinstance(sampled_features, pd.DataFrame):
                    window_features = sampled_features
                elif isinstance(sampled_features, np.ndarray):
                    window_features = pd.DataFrame(
                        sampled_features,
                        columns=window_features.columns
                    )
                
                # 獲取採樣後的標籤
                if isinstance(window_graph, dgl.DGLGraph):
                    sampled_indices = window_graph.ndata.get('orig_id', None)
                    if sampled_indices is not None:
                        window_labels = window_labels.iloc[sampled_indices].copy()
                else:
                    # NetworkX圖採樣的情況較複雜，需要更多處理
                    logger.warning("無法準確獲取NetworkX採樣後的標籤")
                
                logger.info(f"採樣後節點數: {len(window_graph.nodes())}")
            
            # 添加到列表
            all_graphs.append(window_graph)
            all_labels.append(window_labels)
            all_features.append(window_features)
        
        # 初始化和訓練模型
        logger.info("初始化TGAT模型...")
        
        # 調整特徵維度 (如果使用了位置嵌入)
        if args.use_position_embedding:
            additional_dim = 32  # 位置嵌入的維度
            config['model']['node_features'] = features.shape[1] + additional_dim
        else:
            config['model']['node_features'] = features.shape[1]
            
        # 調整輸出類別數
        n_classes = len(np.unique(target))
        config['model']['num_classes'] = n_classes
        
        # 初始化模型
        model = OptimizedTGATModel(config)
        
        # 將模型移至適當裝置
        model.to(device)
        
        # 訓練模型
        logger.info("開始訓練模型...")
        training_stats = model.train_model(all_graphs, all_features, all_labels, device)
        
        # 評估模型
        logger.info("評估模型性能...")
        evaluation_results = model.evaluate(X_test, y_test)
        
        # 保存結果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存模型
        model_path = os.path.join(model_dir, f"model_{timestamp}.pt")
        model.save(model_path)
        logger.info(f"模型已保存至: {model_path}")
        
        # 保存評估結果
        results_path = os.path.join(results_dir, f"evaluation_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        logger.info(f"評估結果已保存至: {results_path}")
        
        # 保存訓練歷史
        history_path = os.path.join(results_dir, f"history_{timestamp}.json")
        with open(history_path, 'w') as f:
            json.dump(training_stats, f, indent=2)
        logger.info(f"訓練歷史已保存至: {history_path}")
        
        # 繪製學習曲線
        plt.figure(figsize=(12, 5))
        
        # 繪製訓練損失
        plt.subplot(1, 2, 1)
        plt.plot(training_stats['loss'], label='Training Loss')
        plt.plot(training_stats['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # 繪製準確率
        plt.subplot(1, 2, 2)
        plt.plot(training_stats['accuracy'], label='Training Accuracy')
        plt.plot(training_stats['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        
        # 保存圖形
        plt.tight_layout()
        vis_path = os.path.join(vis_dir, f"training_history_{timestamp}.png")
        plt.savefig(vis_path)
        logger.info(f"學習曲線已保存至: {vis_path}")
        
        # 繪製混淆矩陣
        if 'confusion_matrix' in evaluation_results:
            # 導入相關庫
            from sklearn.metrics import ConfusionMatrixDisplay
            
            plt.figure(figsize=(10, 8))
            cm = evaluation_results['confusion_matrix']
            
            # 獲取類別標籤
            attack_types = data_loader.get_attack_types()
            labels = [attack_types.get(i, f"Class {i}") for i in range(len(cm))]
            
            # 繪製混淆矩陣
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
            plt.title('Confusion Matrix')
            plt.tight_layout()
            
            # 保存圖形
            cm_path = os.path.join(vis_dir, f"confusion_matrix_{timestamp}.png")
            plt.savefig(cm_path)
            logger.info(f"混淆矩陣已保存至: {cm_path}")
        
        # 計算總運行時間
        elapsed_time = time.time() - start_time
        logger.info(f"總運行時間: {elapsed_time:.2f} 秒")
        
        # 輸出最終結果摘要
        logger.info("======= 結果摘要 =======")
        logger.info(f"資料集大小: {len(features)} 筆記錄")
        logger.info(f"處理的時間窗口數: {len(windows)}")
        logger.info(f"最終測試準確率: {evaluation_results.get('accuracy', 0):.4f}")
        logger.info(f"最終測試F1分數: {evaluation_results.get('f1_score', 0):.4f}")
        logger.info("========================")
        
        # 返回主要結果
                results = {
            result = results
    # 使用上下文管理器追蹤記憶體使用
    result = None
    with track_memory_usage('主函數執行'):
        result = result
    return result
            'model_path': model_path,
            'results_path': results_path,
            'accuracy': evaluation_results.get('accuracy', 0),
            'f1_score': evaluation_results.get('f1_score', 0)
        }
        
    except Exception as e:
        logger.error(f"執行過程中發生錯誤: {str(e)}", exc_info=True)
        return {'error': str(e)}
        
    finally:
        # 清理資源
        logger.info("清理資源...")
        # 確保釋放記憶體
        import gc
        gc.collect()
        if use_gpu:
            torch.cuda.empty_cache()

if __name__ == "__main__":
    results = main()
    if 'error' not in results:
        print(f"程序執行成功，模型已保存至: {results['model_path']}")
        print(f"測試準確率: {results['accuracy']:.4f}")
        print(f"測試F1分數: {results['f1_score']:.4f}")
    else:
        print(f"程序執行失敗: {results['error']}")
