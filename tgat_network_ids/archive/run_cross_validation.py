#!/usr/bin/env python
# coding: utf-8 -*-

"""
TGAT模型交叉驗證執行腳本

提供完整的交叉驗證流程，用於評估模型在不同數據分割上的性能並防止過擬合。
支持節點級、圖級和時間序列交叉驗證，同時提供豐富的視覺化和評估指標。
"""

import os
import sys
import torch
import dgl
import argparse
import logging
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

# 確保可以導入專案模組
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# 導入專案模組
from src.data.optimized_data_loader import EnhancedMemoryOptimizedDataLoader
from src.models.optimized_tgat_model import OptimizedTGATModel
from src.utils.cross_validation import GraphCrossValidator
from src.utils.memory_utils import print_memory_usage, track_memory_usage, clean_memory
from src.utils.enhanced_metrics import evaluate_nids_metrics, plot_nids_metrics

# 配置日誌
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='TGAT模型交叉驗證')
    
    # 基本參數
    parser.add_argument('--config', type=str, default='config/memory_optimized_config.yaml',
                        help='配置文件路徑')
    parser.add_argument('--data_path', type=str, default=None,
                        help='數據路徑，覆蓋配置文件中的設定')
    parser.add_argument('--use_gpu', action='store_true', default=None,
                        help='使用GPU，覆蓋配置文件設定')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='輸出目錄，若未指定則使用時間戳建立')
    
    # 交叉驗證參數
    parser.add_argument('--n_splits', type=int, default=5,
                        help='交叉驗證分割數')
    parser.add_argument('--split_type', type=str, choices=['node', 'time', 'graph'], default='node',
                        help='分割類型：節點級、時間序列或圖級')
    parser.add_argument('--shuffle', action='store_true', default=True,
                        help='是否打亂數據')
    parser.add_argument('--random_state', type=int, default=42,
                        help='隨機種子')
    
    # 訓練參數
    parser.add_argument('--epochs', type=int, default=50,
                        help='每個折的訓練輪數')
    parser.add_argument('--patience', type=int, default=10,
                        help='早停耐心值')
    parser.add_argument('--train_final_model', action='store_true', default=False,
                        help='是否使用全部數據訓練最終模型')
    parser.add_argument('--final_epochs', type=int, default=None,
                        help='最終模型訓練輪數，若未指定則使用--epochs的兩倍')
    
    # 正則化參數
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='權重衰減係數，用於L2正則化')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout率')
    
    # 其他參數
    parser.add_argument('--verbose', type=int, default=1,
                        help='輸出詳細程度')
    parser.add_argument('--save_models', action='store_true', default=True,
                        help='是否保存模型')
    parser.add_argument('--plot_results', action='store_true', default=True,
                        help='是否繪製結果圖表')
    
    return parser.parse_args()

def load_config(config_path):
    """載入配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_output_directory(args):
    """設置輸出目錄"""
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # 使用時間戳建立輸出目錄
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(project_root, f'cv_results_{timestamp}')
    
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 創建子目錄
    models_dir = os.path.join(output_dir, 'models')
    plots_dir = os.path.join(output_dir, 'plots')
    results_dir = os.path.join(output_dir, 'results')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    return {
        'base_dir': output_dir,
        'models_dir': models_dir,
        'plots_dir': plots_dir,
        'results_dir': results_dir
    }

def save_config_snapshot(config, output_dir):
    """保存配置快照"""
    config_path = os.path.join(output_dir, 'config_snapshot.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"配置快照已保存至: {config_path}")
    return config_path

@track_memory_usage
def main():
    """主函數"""
    # 解析命令行參數
    args = parse_arguments()
    
    # 載入配置
    config = load_config(args.config)
    
    # 設置輸出目錄
    output_dirs = setup_output_directory(args)
    
    # 命令行參數覆蓋配置文件
    if args.data_path:
        config['data']['path'] = args.data_path
    
    if args.use_gpu is not None:
        config['model']['use_gpu'] = args.use_gpu
    
    # 增加正則化參數 (解決過擬合)
    if args.weight_decay is not None:
        config['training']['weight_decay'] = args.weight_decay
        logger.info(f"使用自定義權重衰減係數: {args.weight_decay}")
    
    if args.dropout is not None:
        config['model']['dropout'] = args.dropout
        logger.info(f"使用自定義Dropout率: {args.dropout}")
    
    # 保存配置快照
    save_config_snapshot(config, output_dirs['base_dir'])
    
    # 檢查是否使用GPU
    use_gpu = config['model'].get('use_gpu', False) and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    logger.info(f"使用設備: {device}")
    
    # 載入資料
    logger.info("載入數據...")
    data_loader = EnhancedMemoryOptimizedDataLoader(config)
    data_loader.load_data()
    
    # 預處理資料
    logger.info("預處理數據...")
    features, target = data_loader.preprocess()
    
    # 構建圖
    logger.info("構建圖...")
    graph = data_loader.build_graph()
    
    # 獲取時間戳 (如果有)
    timestamps = getattr(graph, 'timestamps', None)
    
    # 確保標籤是張量
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target)
    
    # 準備交叉驗證配置
    cv_config = {
        'n_splits': args.n_splits,
        'split_type': args.split_type,
        'shuffle': args.shuffle,
        'random_state': args.random_state,
        'save_models': args.save_models,
        'save_dir': output_dirs['models_dir'],
        'verbose': args.verbose,
        'early_stopping': True,
        'patience': args.patience,
        'plot_results': args.plot_results
    }
    
    # 準備模型配置
    model_config = config.copy()
    model_config['model']['node_features'] = features.shape[1]
    model_config['model']['num_classes'] = len(np.unique(target))
    model_config['training']['epochs'] = args.epochs
    
    # 如果指定了最終輪數
    if args.final_epochs:
        model_config['training']['final_epochs'] = args.final_epochs
    else:
        model_config['training']['final_epochs'] = args.epochs * 2
    
    # 初始化交叉驗證器
    cv = GraphCrossValidator(
        model_class=OptimizedTGATModel,
        model_config=model_config,
        cv_config=cv_config,
        device=device
    )
    
    # 執行交叉驗證
    logger.info(f"開始{args.n_splits}折交叉驗證...")
    cv_results = cv.cross_validate(graph, target, timestamps)
    
    # 保存交叉驗證結果
    results_path = os.path.join(output_dirs['results_dir'], 'cv_results.json')
    
    # 將不可序列化的對象轉換為可序列化
    serializable_results = {}
    for k, v in cv_results.items():
        if k == 'fold_results':
            serializable_results[k] = []
            for fold_result in v:
                serializable_fold = {}
                for fold_k, fold_v in fold_result.items():
                    if fold_k == 'metrics' and isinstance(fold_v, dict):
                        serializable_metrics = {}
                        for metric_k, metric_v in fold_v.items():
                            # 轉換numpy數組和數據類型
                            if isinstance(metric_v, np.ndarray):
                                serializable_metrics[metric_k] = metric_v.tolist()
                            elif isinstance(metric_v, np.integer):
                                serializable_metrics[metric_k] = int(metric_v)
                            elif isinstance(metric_v, np.floating):
                                serializable_metrics[metric_k] = float(metric_v)
                            else:
                                serializable_metrics[metric_k] = metric_v
                        serializable_fold[fold_k] = serializable_metrics
                    elif fold_k == 'history':
                        # 轉換歷史記錄
                        if fold_v:
                            serializable_history = {}
                            for hist_k, hist_v in fold_v.items():
                                if isinstance(hist_v, list) and all(isinstance(x, np.number) for x in hist_v):
                                    serializable_history[hist_k] = [float(x) for x in hist_v]
                                else:
                                    serializable_history[hist_k] = hist_v
                            serializable_fold[fold_k] = serializable_history
                        else:
                            serializable_fold[fold_k] = None
                    else:
                        # 轉換其他類型
                        if isinstance(fold_v, np.ndarray):
                            serializable_fold[fold_k] = fold_v.tolist()
                        elif isinstance(fold_v, np.integer):
                            serializable_fold[fold_k] = int(fold_v)
                        elif isinstance(fold_v, np.floating):
                            serializable_fold[fold_k] = float(fold_v)
                        else:
                            serializable_fold[fold_k] = fold_v
                serializable_results[k].append(serializable_fold)
        else:
            # 轉換頂層鍵
            if isinstance(v, np.ndarray):
                serializable_results[k] = v.tolist()
            elif isinstance(v, np.integer):
                serializable_results[k] = int(v)
            elif isinstance(v, np.floating):
                serializable_results[k] = float(v)
            elif isinstance(v, dict):
                serializable_results[k] = {}
                for sub_k, sub_v in v.items():
                    if isinstance(sub_v, np.ndarray):
                        serializable_results[k][sub_k] = sub_v.tolist()
                    elif isinstance(sub_v, np.integer):
                        serializable_results[k][sub_k] = int(sub_v)
                    elif isinstance(sub_v, np.floating):
                        serializable_results[k][sub_k] = float(sub_v)
                    else:
                        serializable_results[k][sub_k] = sub_v
            else:
                serializable_results[k] = v
    
    # 保存序列化結果
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    logger.info(f"交叉驗證結果已保存至: {results_path}")
    
    # 訓練最終模型
    if args.train_final_model:
        logger.info("使用全部數據訓練最終模型...")
        final_model = cv.train_final_model(graph, target)
        
        # 評估最終模型
        loss, acc, metrics = final_model.evaluate(graph, target)
        
        logger.info(f"最終模型性能: 精度={acc:.4f}, 損失={loss:.4f}")
        
        # 使用增強的評估指標
        if hasattr(final_model, 'predict_proba'):
            y_proba = final_model.predict_proba(graph)
            enhanced_metrics = evaluate_nids_metrics(
                target.cpu().numpy(),
                y_proba.cpu().numpy(),
                target_tpr_levels=[0.90, 0.95, 0.99]
            )
            
            # 保存增強指標
            enhanced_metrics_path = os.path.join(output_dirs['results_dir'], 'enhanced_metrics.json')
            with open(enhanced_metrics_path, 'w') as f:
                serializable_enhanced = {}
                for k, v in enhanced_metrics.items():
                    if isinstance(v, dict):
                        serializable_enhanced[k] = {}
                        for sub_k, sub_v in v.items():
                            if isinstance(sub_v, np.ndarray):
                                serializable_enhanced[k][sub_k] = sub_v.tolist()
                            elif isinstance(sub_v, (float, int)):
                                serializable_enhanced[k][sub_k] = float(sub_v)
                            else:
                                serializable_enhanced[k][sub_k] = sub_v
                    else:
                        serializable_enhanced[k] = v
                json.dump(serializable_enhanced, f, indent=4)
                
            # 繪製NIDS特定指標
            if args.plot_results:
                plot_path = os.path.join(output_dirs['plots_dir'], 'nids_metrics.png')
                plot_nids_metrics(enhanced_metrics, plot_path)
                logger.info(f"NIDS指標圖表已保存至: {plot_path}")
                
            # 輸出FPR@TPR指標
            for tpr_level in [0.90, 0.95, 0.99]:
                key = f'fpr_at_{tpr_level}_tpr'
                if key in enhanced_metrics.get('fpr_at_tpr', {}):
                    fpr = enhanced_metrics['fpr_at_tpr'][key]['fpr_at_target_tpr']
                    logger.info(f"FPR@{tpr_level}TPR: {fpr:.6f}")
    
    logger.info("交叉驗證流程完成")
    logger.info(f"所有結果已保存至: {output_dirs['base_dir']}")
    
    # 返回交叉驗證結果概要
    return {
        'mean_accuracy': cv_results['mean_test_accuracy'],
        'std_accuracy': cv_results['std_test_accuracy'],
        'output_dir': output_dirs['base_dir']
    }

if __name__ == "__main__":
    results = main()
    print(f"\n交叉驗證結果摘要:")
    print(f"平均準確率: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    print(f"輸出目錄: {results['output_dir']}")
