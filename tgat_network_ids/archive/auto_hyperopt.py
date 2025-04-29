#!/usr/bin/env python
# coding: utf-8 -*-

"""
TGAT 網路入侵檢測系統超參數自動優化腳本

使用 Optuna 框架實現自動超參數調優，降低手動調整的工作量，
同時系統性地探索最佳參數組合，提升模型性能。
"""

import os
import sys
import yaml
import logging
import argparse
import numpy as np
import json
import torch
import time
import optuna
from datetime import datetime
from functools import partial
import matplotlib.pyplot as plt
import joblib

# 確保可以導入專案模組
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# 導入專案模組
from src.data.optimized_data_loader import EnhancedMemoryOptimizedDataLoader
from src.models.optimized_tgat_model import OptimizedTGATModel
from src.utils.memory_utils import print_memory_usage, track_memory_usage

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"tgat_hyperopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

# 為Optuna設定更簡潔的日誌級別
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='TGAT超參數自動優化')
    
    # 基本參數
    parser.add_argument('--config', type=str, default='config/memory_optimized_config.yaml',
                        help='基礎配置文件路徑')
    parser.add_argument('--data_path', type=str, default=None,
                        help='數據路徑，覆蓋配置文件中的設定')
    parser.add_argument('--use_gpu', action='store_true', default=None,
                        help='使用GPU，覆蓋配置文件中的設定')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Optuna優化試驗次數')
    parser.add_argument('--study_name', type=str, default=None,
                        help='Optuna研究名稱，若未指定則使用時間戳')
    parser.add_argument('--storage', type=str, default=None,
                        help='Optuna存儲URI，例如 sqlite:///optuna.db')
    parser.add_argument('--load_if_exists', action='store_true', default=False,
                        help='如果研究已存在，則載入繼續進行')
    parser.add_argument('--optimize_target', type=str, default='f1_score',
                        choices=['accuracy', 'f1_score', 'precision', 'recall', 'auc'],
                        help='優化目標指標')
    parser.add_argument('--timeout', type=int, default=None,
                        help='優化超時時間(秒)，若未指定則無限制')
    
    # 優化範圍參數
    parser.add_argument('--optimize_model', action='store_true', default=True,
                        help='優化模型架構參數')
    parser.add_argument('--optimize_training', action='store_true', default=True,
                        help='優化訓練參數')
    parser.add_argument('--optimize_time_encoding', action='store_true', default=True,
                        help='優化時間編碼參數')
    parser.add_argument('--optimize_attention', action='store_true', default=True,
                        help='優化注意力機制參數')
    parser.add_argument('--optimize_sampling', action='store_true', default=False,
                        help='優化採樣參數')
    
    return parser.parse_args()

def load_config(config_path):
    """載入配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_output_dirs():
    """設置輸出目錄"""
    # 創建超參數優化結果目錄
    hyperopt_dir = os.path.join(project_root, 'hyperopt_results')
    os.makedirs(hyperopt_dir, exist_ok=True)
    
    # 創建最佳模型目錄
    best_model_dir = os.path.join(hyperopt_dir, 'best_models')
    os.makedirs(best_model_dir, exist_ok=True)
    
    # 創建視覺化目錄
    plots_dir = os.path.join(hyperopt_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    return hyperopt_dir, best_model_dir, plots_dir

def suggest_model_params(trial, config):
    """為模型架構建議超參數
    
    參數:
        trial: Optuna trial物件
        config: 基礎配置字典
    
    返回:
        dict: 更新後的配置字典
    """
    if 'model' not in config:
        config['model'] = {}
    
    # 模型隱藏層維度
    config['model']['hidden_dim'] = trial.suggest_categorical(
        'hidden_dim', [32, 64, 128, 256]
    )
    
    # 模型層數
    config['model']['num_layers'] = trial.suggest_int(
        'num_layers', 1, 3
    )
    
    # Dropout率
    config['model']['dropout'] = trial.suggest_float(
        'dropout', 0.1, 0.5
    )
    
    # 注意力頭數
    if config.get('optimize_attention', True):
        config['model']['num_heads'] = trial.suggest_int(
            'num_heads', 1, 8
        )
    
    return config

def suggest_training_params(trial, config):
    """為訓練過程建議超參數
    
    參數:
        trial: Optuna trial物件
        config: 基礎配置字典
    
    返回:
        dict: 更新後的配置字典
    """
    if 'training' not in config:
        config['training'] = {}
    
    # 學習率
    config['training']['learning_rate'] = trial.suggest_float(
        'learning_rate', 1e-5, 1e-2, log=True
    )
    
    # 批次大小
    config['training']['batch_size'] = trial.suggest_categorical(
        'batch_size', [16, 32, 64, 128, 256]
    )
    
    # 權重衰減(L2正則化)
    config['training']['weight_decay'] = trial.suggest_float(
        'weight_decay', 1e-6, 1e-3, log=True
    )
    
    # 早停耐心值
    config['training']['patience'] = trial.suggest_int(
        'patience', 5, 15
    )
    
    # 學習率調度器
    config['training']['lr_scheduler'] = trial.suggest_categorical(
        'lr_scheduler', ['step', 'cosine', 'plateau', 'none']
    )
    
    # 根據學習率調度器類型建議額外參數
    if config['training']['lr_scheduler'] == 'step':
        config['training']['lr_step_size'] = trial.suggest_int(
            'lr_step_size', 3, 10
        )
        config['training']['lr_gamma'] = trial.suggest_float(
            'lr_gamma', 0.1, 0.9
        )
    elif config['training']['lr_scheduler'] == 'plateau':
        config['training']['lr_factor'] = trial.suggest_float(
            'lr_factor', 0.1, 0.9
        )
        config['training']['lr_patience'] = trial.suggest_int(
            'lr_patience', 2, 7
        )
    
    return config

def suggest_time_encoding_params(trial, config):
    """為時間編碼建議超參數
    
    參數:
        trial: Optuna trial物件
        config: 基礎配置字典
    
    返回:
        dict: 更新後的配置字典
    """
    if 'time_encoding' not in config:
        config['time_encoding'] = {}
    
    # 時間編碼方法
    config['time_encoding']['method'] = trial.suggest_categorical(
        'time_encoding_method', ['timefeature', 'fourier', 'learnable']
    )
    
    # 時間編碼維度
    config['time_encoding']['dimension'] = trial.suggest_categorical(
        'time_encoding_dimension', [16, 32, 64, 128]
    )
    
    # 針對傅立葉編碼的特殊參數
    if config['time_encoding']['method'] == 'fourier':
        config['time_encoding']['fourier_freqs'] = trial.suggest_int(
            'fourier_frequencies', 8, 32
        )
    
    return config

def suggest_attention_params(trial, config):
    """為注意力機制建議超參數
    
    參數:
        trial: Optuna trial物件
        config: 基礎配置字典
    
    返回:
        dict: 更新後的配置字典
    """
    if 'attention' not in config:
        config['attention'] = {}
    
    # 注意力類型
    config['attention']['type'] = trial.suggest_categorical(
        'attention_type', ['dot_product', 'general', 'additive']
    )
    
    # 注意力溫度係數(用於調整softmax的銳利度)
    config['attention']['temperature'] = trial.suggest_float(
        'attention_temperature', 0.5, 2.0
    )
    
    # 是否使用殘差連接
    config['attention']['use_residual'] = trial.suggest_categorical(
        'use_residual', [True, False]
    )
    
    # 位置編碼選擇
    config['attention']['position_encoding'] = trial.suggest_categorical(
        'position_encoding', ['none', 'fixed', 'learned']
    )
    
    return config

def suggest_sampling_params(trial, config):
    """為圖採樣策略建議超參數
    
    參數:
        trial: Optuna trial物件
        config: 基礎配置字典
    
    返回:
        dict: 更新後的配置字典
    """
    if 'sampling' not in config:
        config['sampling'] = {}
    
    # 採樣方法
    config['sampling']['method'] = trial.suggest_categorical(
        'sampling_method', ['graphsaint', 'cluster-gcn', 'frontier', 'historical']
    )
    
    # 基於方法選擇不同參數
    if config['sampling']['method'] == 'graphsaint':
        config['sampling']['walk_length'] = trial.suggest_int(
            'walk_length', 5, 20
        )
        config['sampling']['num_walks'] = trial.suggest_int(
            'num_walks', 10, 50
        )
    elif config['sampling']['method'] == 'cluster-gcn':
        config['sampling']['num_clusters'] = trial.suggest_int(
            'num_clusters', 10, 200
        )
    
    # 通用採樣參數
    config['sampling']['sample_size'] = trial.suggest_categorical(
        'sample_size', [1000, 2000, 5000, 10000]
    )
    
    # 記憶機制設定
    config['sampling']['use_memory'] = trial.suggest_categorical(
        'use_memory', [True, False]
    )
    
    if config['sampling']['use_memory']:
        config['sampling']['memory_size'] = trial.suggest_categorical(
            'memory_size', [500, 1000, 2000, 5000]
        )
    
    return config

def objective(trial, base_config, data_path=None, use_gpu=None, optimize_target='f1_score',
             optimize_model=True, optimize_training=True, optimize_time_encoding=True,
             optimize_attention=True, optimize_sampling=False):
    """Optuna優化目標函數
    
    參數:
        trial: Optuna trial物件
        base_config: 基礎配置字典
        data_path: 數據路徑
        use_gpu: 是否使用GPU
        optimize_target: 優化目標指標
        optimize_*: 是否優化對應部分的超參數
    
    返回:
        float: 優化目標的評估分數
    """
    try:
        logger.info(f"開始Trial #{trial.number}")
        start_time = time.time()
        
        # 複製基礎配置
        config = base_config.copy()
        
        # 命令行參數覆蓋配置文件
        if data_path:
            config['data']['path'] = data_path
        if use_gpu is not None:
            config['model']['use_gpu'] = use_gpu
        
        # 應用超參數建議
        if optimize_model:
            config = suggest_model_params(trial, config)
        if optimize_training:
            config = suggest_training_params(trial, config)
        if optimize_time_encoding:
            config = suggest_time_encoding_params(trial, config)
        if optimize_attention:
            config = suggest_attention_params(trial, config)
        if optimize_sampling:
            config = suggest_sampling_params(trial, config)
        
        # 記錄當前超參數組合
        logger.info(f"Trial #{trial.number} 超參數: {trial.params}")
        
        # 檢查是否使用GPU
        use_gpu = config['model'].get('use_gpu', False) and torch.cuda.is_available()
        device = torch.device('cuda:0' if use_gpu else 'cpu')
        logger.info(f"使用裝置: {device}")
        
        # 初始化資料載入器
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
        
        # 配置模型輸入參數
        config['model']['node_features'] = features.shape[1]
        n_classes = len(np.unique(target))
        config['model']['num_classes'] = n_classes
        
        # 初始化模型
        logger.info(f"初始化TGAT模型...")
        model = OptimizedTGATModel(config)
        model.to(device)
        
        # 訓練模型
        logger.info("開始訓練模型...")
        training_stats = model.train_model(X_train, y_train, X_test, y_test, device)
        
        # 評估模型
        logger.info("評估模型性能...")
        evaluation_results = model.evaluate(X_test, y_test)
        
        # 獲取目標指標
        score = evaluation_results.get(optimize_target, 0.0)
        logger.info(f"Trial #{trial.number} 完成. {optimize_target} = {score:.4f}")
        logger.info(f"Trial執行時間: {time.time() - start_time:.2f}秒")
        
        # 記錄中間結果
        trial.set_user_attr('training_history', training_stats)
        trial.set_user_attr('evaluation_results', evaluation_results)
        
        return score
    
    except Exception as e:
        logger.error(f"Trial #{trial.number} 執行失敗: {str(e)}", exc_info=True)
        raise optuna.exceptions.TrialPruned()

def save_study_results(study, hyperopt_dir, best_model_dir, plots_dir):
    """保存優化研究結果
    
    參數:
        study: Optuna Study物件
        hyperopt_dir: 超參數優化結果目錄
        best_model_dir: 最佳模型目錄
        plots_dir: 視覺化目錄
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存最佳超參數
    best_params = study.best_params
    best_params_path = os.path.join(hyperopt_dir, f"best_params_{timestamp}.json")
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"最佳超參數已保存至: {best_params_path}")
    
    # 保存研究物件
    study_path = os.path.join(hyperopt_dir, f"study_{timestamp}.pkl")
    joblib.dump(study, study_path)
    logger.info(f"研究物件已保存至: {study_path}")
    
    # 生成參數重要性圖
    try:
        importance = optuna.importance.get_param_importances(study)
        importance_df = pd.DataFrame(
            importance.items(), columns=['Parameter', 'Importance']
        ).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.bar(importance_df['Parameter'], importance_df['Importance'])
        plt.xticks(rotation=45, ha='right')
        plt.title('超參數重要性')
        plt.tight_layout()
        
        importance_fig_path = os.path.join(plots_dir, f"param_importance_{timestamp}.png")
        plt.savefig(importance_fig_path)
        plt.close()
        logger.info(f"參數重要性圖已保存至: {importance_fig_path}")
    except Exception as e:
        logger.warning(f"生成參數重要性圖失敗: {str(e)}")
    
    # 生成優化歷史圖
    plt.figure(figsize=(10, 6))
    
    # 獲取所有已完成的trial
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    # 按順序繪製目標值
    values = [t.value for t in trials]
    plt.plot(range(1, len(values) + 1), values, 'b-', alpha=0.3)
    plt.plot(range(1, len(values) + 1), values, 'bo')
    
    # 繪製最佳值
    best_values = [study.best_value if i >= study.best_trial.number else float('nan') 
                 for i in range(len(values))]
    plt.plot(range(1, len(values) + 1), best_values, 'r-')
    
    plt.xlabel('Trial數')
    plt.ylabel(f'目標指標 ({study.direction.name})')
    plt.title('超參數優化進程')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    history_fig_path = os.path.join(plots_dir, f"optimization_history_{timestamp}.png")
    plt.savefig(history_fig_path)
    plt.close()
    logger.info(f"優化歷史圖已保存至: {history_fig_path}")
    
    # 生成超參數分布圖
    param_names = list(study.best_params.keys())
    n_params = len(param_names)
    
    if n_params > 0:
        # 為每個參數創建一個子圖
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
        if n_params == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, param_name in enumerate(param_names):
            if i < len(axes):
                ax = axes[i]
                param_values = [t.params[param_name] for t in trials if param_name in t.params]
                ax.hist(param_values, bins=20)
                ax.set_title(param_name)
                ax.set_xlabel('參數值')
                ax.set_ylabel('頻率')
        
        # 隱藏未使用的子圖
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        param_dist_path = os.path.join(plots_dir, f"param_distribution_{timestamp}.png")
        plt.savefig(param_dist_path)
        plt.close()
        logger.info(f"參數分布圖已保存至: {param_dist_path}")
    
    # 創建最終報告
    report = {
        "best_params": best_params,
        "best_value": study.best_value,
        "direction": study.direction.name,
        "n_trials": len(study.trials),
        "completed_trials": len(trials),
        "timestamp": timestamp,
        "study_name": study.study_name,
    }
    
    report_path = os.path.join(hyperopt_dir, f"optimization_report_{timestamp}.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info(f"優化報告已保存至: {report_path}")

@track_memory_usage
def main():
    """主函數"""
    # 解析命令行參數
    args = parse_args()
    
    # 載入配置
    config = load_config(args.config)
    
    # 設置輸出目錄
    hyperopt_dir, best_model_dir, plots_dir = setup_output_dirs()
    
    # 設置研究名稱
    study_name = args.study_name or f"tgat_hyperopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 設置存儲
    if args.storage:
        storage = args.storage
    else:
        # 使用本地SQLite數據庫
        storage = f"sqlite:///{os.path.join(hyperopt_dir, study_name + '.db')}"
    
    logger.info(f"創建Optuna研究: {study_name}")
    logger.info(f"存儲位置: {storage}")
    logger.info(f"優化目標: {args.optimize_target} (方向: 最大化)")
    
    try:
        # 創建Optuna研究
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",  # 最大化目標指標
            load_if_exists=args.load_if_exists
        )
        
        # 設置優化目標函數
        objective_func = partial(
            objective,
            base_config=config,
            data_path=args.data_path,
            use_gpu=args.use_gpu,
            optimize_target=args.optimize_target,
            optimize_model=args.optimize_model,
            optimize_training=args.optimize_training,
            optimize_time_encoding=args.optimize_time_encoding,
            optimize_attention=args.optimize_attention,
            optimize_sampling=args.optimize_sampling
        )
        
        # 顯示優化範圍
        optimization_targets = []
        if args.optimize_model:
            optimization_targets.append("模型架構")
        if args.optimize_training:
            optimization_targets.append("訓練參數")
        if args.optimize_time_encoding:
            optimization_targets.append("時間編碼")
        if args.optimize_attention:
            optimization_targets.append("注意力機制")
        if args.optimize_sampling:
            optimization_targets.append("採樣策略")
        
        logger.info(f"優化範圍: {', '.join(optimization_targets)}")
        
        # 開始優化
        logger.info(f"開始執行 {args.n_trials} 次試驗...")
        study.optimize(objective_func, n_trials=args.n_trials, timeout=args.timeout)
        
        # 顯示最佳結果
        logger.info("優化完成!")
        logger.info(f"最佳試驗 #{study.best_trial.number}")
        logger.info(f"最佳 {args.optimize_target}: {study.best_value:.4f}")
        logger.info(f"最佳超參數: {study.best_params}")
        
        # 保存研究結果
        save_study_results(study, hyperopt_dir, best_model_dir, plots_dir)
        
        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "best_trial": study.best_trial.number
        }
        
    except Exception as e:
        logger.error(f"超參數優化過程中發生錯誤: {str(e)}", exc_info=True)
        return {"error": str(e)}
    
    finally:
        # 清理資源
        logger.info("清理資源...")
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # 檢查是否安裝了Optuna
    try:
        import optuna
        import pandas as pd
    except ImportError:
        print("錯誤: 請先安裝Optuna和pandas。執行: pip install optuna pandas")
        sys.exit(1)
        
    results = main()
    if "error" not in results:
        print(f"超參數優化完成，最佳{args.optimize_target}: {results['best_value']:.4f}")
    else:
        print(f"超參數優化失敗: {results['error']}")
