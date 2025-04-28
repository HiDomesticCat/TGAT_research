#!/usr/bin/env python
# coding: utf-8 -*-

"""
增強版 TGAT 網路入侵檢測系統運行腳本

提供完整的訓練、評估和預測功能，整合了：
1. 進階圖採樣策略
2. 自適應時間窗口
3. 高效記憶體管理
4. 混合精度訓練
5. 完整的訓練引擎
"""

import os
import sys
import yaml
import argparse
import logging
import time
import torch
import numpy as np
from datetime import datetime

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 導入項目模塊
from src.data.optimized_data_loader import EnhancedMemoryOptimizedDataLoader as OptimizedDataLoader
from src.data.optimized_graph_builder import OptimizedGraphBuilder
from src.data.advanced_sampling import AdvancedGraphSampler
from src.data.adaptive_window import AdaptiveTimeWindow
from src.models.optimized_tgat_model import OptimizedTGATModel
from src.models.training_engine import TrainingEngine
from src.utils.memory_utils import track_memory_usage, log_memory_usage
from src.utils.enhanced_metrics import compute_advanced_metrics

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def safe_float_convert(value, default=0.0):
    """安全地將值轉換為浮點數"""
    if value is None:
        return default
    
    try:
        # 首先嘗試直接轉換
        return float(value)
    except (ValueError, TypeError):
        # 如果是字符串，可能是科學記數法或格式問題
        if isinstance(value, str):
            try:
                # 處理可能的科學記數法
                value = value.strip().lower()
                if 'e' in value:
                    base, exp = value.split('e')
                    return float(base) * (10 ** int(exp))
                return default
            except:
                return default
        return default

def safe_int_convert(value, default=0):
    """安全地將值轉換為整數"""
    try:
        # 先轉為浮點數然後取整
        return int(float(value))
    except (ValueError, TypeError):
        return default

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='Enhanced TGAT Network Intrusion Detection System')
    
    # 基本參數
    parser.add_argument('--config', type=str, default='config/memory_optimized_config.yaml',
                        help='Configuration file path')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to the data directory')
    parser.add_argument('--use_gpu', type=bool, default=None,
                        help='Whether to use GPU (True/False)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'predict'],
                        help='Operation mode')
    
    # 圖優化和記憶體管理參數
    parser.add_argument('--use_adaptive_window', action='store_true',
                        help='Use adaptive time window')
    parser.add_argument('--adaptive_window_config', type=str, default=None,
                        help='Adaptive window configuration')
    parser.add_argument('--use_advanced_sampling', action='store_true',
                        help='Use advanced graph sampling strategy')
    parser.add_argument('--sampling_method', type=str, default='graphsaint',
                        choices=['graphsaint', 'cluster-gcn', 'frontier', 'historical'],
                        help='Graph sampling method')
    parser.add_argument('--sample_size', type=int, default=5000,
                        help='Subgraph sample size')
    
    # 模型參數
    parser.add_argument('--use_memory', action='store_true',
                        help='Use memory mechanism')
    parser.add_argument('--memory_size', type=int, default=1000,
                        help='Memory size')
    parser.add_argument('--use_position_embedding', action='store_true',
                        help='Use position embedding')
    
    # 輸出參數
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--save_model', action='store_true',
                        help='Save model checkpoint')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results')
    
    # 高級優化參數
    parser.add_argument('--use_sparse_representation', action='store_true',
                        help='Use sparse graph representation')
    parser.add_argument('--use_mixed_precision', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Use gradient checkpointing to save memory')
    
    # 日誌級別
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    
    return parser.parse_args()

def load_config(config_path):
    """加載配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """主函數"""
    # 開始追蹤記憶體使用
    start_time = time.time()
    mem_tracker = track_memory_usage("Main function execution")
    
    # 解析參數
    args = parse_args()
    
    # 設置日誌級別
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 加載配置
    logger.info(f"Using configuration file: {args.config}")
    config = load_config(args.config)
    
    # 命令行參數優先於配置文件
    logger.info(f"Command line arguments: {args}")
    
    # 設置數據路徑
    if args.data_path:
        config['data']['path'] = args.data_path
    
    # 設置模式
    mode = args.mode
    
    # 自適應時間窗口設置
    logger.info(f"Adaptive time window: {args.use_adaptive_window}")
    
    # 進階圖採樣設置
    logger.info(f"Advanced graph sampling strategy: {args.use_advanced_sampling} (method: {args.sampling_method})")
    logger.info(f"Memory mechanism: {args.use_memory}")
    logger.info(f"Position embedding: {args.use_position_embedding}")
    
    # 檢查 GPU 可用性
    use_gpu = config['model'].get('use_gpu', False) and torch.cuda.is_available()
    if args.use_gpu is not None:
        use_gpu = args.use_gpu and torch.cuda.is_available()
    
    device = torch.device('cuda' if use_gpu else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 載入數據
    logger.info("Starting data loading...")
    data_loader = OptimizedDataLoader(config)
    
    # 構建圖
    logger.info("Starting graph building...")
    graph_builder = OptimizedGraphBuilder(config, device=device)
    
    # 如果啟用進階採樣，創建採樣器
    if args.use_advanced_sampling:
        logger.info(f"Initializing advanced sampler with method: {args.sampling_method}")
        sampler = AdvancedGraphSampler(
            method=args.sampling_method,
            sample_size=args.sample_size,
            seed=config.get('random_seed', 42)
        )
        graph_builder.set_sampler(sampler)
    
    # 如果啟用自適應時間窗口，設置窗口
    if args.use_adaptive_window:
        adaptive_config = {}
        if args.adaptive_window_config:
            with open(args.adaptive_window_config, 'r') as f:
                adaptive_config = yaml.safe_load(f)
        
        logger.info(f"Initializing adaptive time window with config: {adaptive_config}")
        adaptive_window = AdaptiveTimeWindow(**adaptive_config)
        graph_builder.set_adaptive_window(adaptive_window)
    
    # 創建模型
    logger.info("Creating model...")
    
    # 獲取模型參數
    model_config = config['model']
    input_dim = model_config.get('input_dim', 128)
    hidden_dim = model_config.get('hidden_dim', 64)
    out_dim = model_config.get('out_dim', 64)
    num_classes = model_config.get('num_classes', 2)
    num_layers = model_config.get('num_layers', 2)
    num_heads = model_config.get('num_heads', 4)
    
    logger.info(f"Model parameters: input_dim={input_dim}, hidden_dim={hidden_dim}, "
               f"out_dim={out_dim}, num_classes={num_classes}, "
               f"num_layers={num_layers}, num_heads={num_heads}")
    
    # 創建模型
    model = OptimizedTGATModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        num_heads=num_heads,
        use_memory=args.use_memory,
        memory_size=args.memory_size,
        use_position_embedding=args.use_position_embedding
    )
    model.to(device)
    
    # 設置優化器
    train_config = config.get('train', {})
    learning_rate_raw = train_config.get('learning_rate', 0.001)
    weight_decay_raw = train_config.get('weight_decay', 5e-4)
    
    # 輸出原始值和類型
    logger.info(f"Raw values from config - learning_rate: {learning_rate_raw} (type: {type(learning_rate_raw).__name__}), "
               f"weight_decay: {weight_decay_raw} (type: {type(weight_decay_raw).__name__})")
    
    # 安全轉換
    learning_rate = safe_float_convert(learning_rate_raw, 0.001)
    weight_decay = safe_float_convert(weight_decay_raw, 5e-4)
    
    # 輸出轉換後的值
    logger.info(f"Converted values - learning_rate: {learning_rate}, weight_decay: {weight_decay}")
    
    # 創建優化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 創建損失函數
    criterion = torch.nn.CrossEntropyLoss()
    
    logger.info(f"Optimizer configured with lr={learning_rate}, weight_decay={weight_decay}")
    
    # 根據模式執行不同操作
    if mode == 'train':
        logger.info("Starting model training...")
        
        # 創建訓練引擎
        training_engine = TrainingEngine(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            config=config
        )
        
        # 執行訓練
        training_results = training_engine.train(data_loader, graph_builder)
        
        # 輸出訓練結果
        logger.info(f"Training completed with best validation accuracy: {training_results['best_val_metrics']['accuracy']:.4f}")
        logger.info(f"Total training time: {training_results['total_time']:.2f}s")
        
        # 如果需要視覺化，創建訓練歷史視覺化
        if args.visualize:
            from src.visualization.visualization import plot_training_history
            
            # 創建輸出目錄
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = args.output_dir or f"./visualizations/{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
            
            # 繪製訓練歷史
            plot_training_history(
                training_results['training_history'], 
                os.path.join(output_dir, f"training_history_{timestamp}.png")
            )
            logger.info(f"Training history visualization saved to {output_dir}")

    elif mode == 'eval':
        logger.info("Evaluating model performance...")
        
        # 加載最佳模型
        model_path = os.path.join(config.get('output', {}).get('model_dir', './models'), 'best_model.pt')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from {model_path}")
        else:
            logger.warning(f"Model file not found at {model_path}, using untrained model")
        
        # 創建訓練引擎
        training_engine = TrainingEngine(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            config=config
        )
        
        # 執行評估
        val_loss, val_metrics = training_engine.validate(data_loader, graph_builder)
        
        # 輸出評估結果
        logger.info(f"Evaluation results - Loss: {val_loss:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
        
        # 計算進階指標
        advanced_metrics = compute_advanced_metrics(model, data_loader, graph_builder, device)
        for metric_name, metric_value in advanced_metrics.items():
            logger.info(f"{metric_name}: {metric_value:.4f}")
        
        # 保存評估結果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_dir or f"./results/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        import json
        with open(os.path.join(output_dir, f"evaluation_{timestamp}.json"), 'w') as f:
            json.dump({
                'loss': val_loss,
                **val_metrics,
                **advanced_metrics
            }, f, indent=2)
        logger.info(f"Evaluation results saved to {output_dir}")
        
        # 如果需要視覺化，創建混淆矩陣等
        if args.visualize:
            from src.visualization.visualization import plot_confusion_matrix
            
            plot_confusion_matrix(
                advanced_metrics['confusion_matrix'],
                os.path.join(output_dir, f"confusion_matrix_{timestamp}.png")
            )
            logger.info(f"Confusion matrix visualization saved to {output_dir}")

    elif mode == 'predict':
        logger.info("Running prediction...")
        
        # 加載最佳模型
        model_path = os.path.join(config.get('output', {}).get('model_dir', './models'), 'best_model.pt')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from {model_path}")
        else:
            logger.warning(f"Model file not found at {model_path}, using untrained model")
        
        # 獲取預測數據
        predict_data = data_loader.get_test_data()
        
        # 執行預測
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in predict_data:
                batch_graph = graph_builder.build_batch_graph(batch)
                batch_features = batch['features'].to(device)
                
                outputs = model(batch_graph, batch_features)
                _, predicted = torch.max(outputs, 1)
                
                predictions.extend(predicted.cpu().numpy())
        
        # 保存預測結果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = args.output_dir or f"./predictions/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        import pandas as pd
        pd.DataFrame({
            'prediction': predictions
        }).to_csv(os.path.join(output_dir, f"predictions_{timestamp}.csv"), index=False)
        
        logger.info(f"Predictions saved to {output_dir}")
    
    # 結束記憶體追蹤並輸出統計信息
    mem_tracker.stop()
    log_memory_usage(mem_tracker)
    
    logger.info("Execution completed: success")

if __name__ == '__main__':
    main()
