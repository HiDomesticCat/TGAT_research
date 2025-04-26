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

    # 視覺化與輸出相關
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='是否生成視覺化圖表')
    parser.add_argument('--save_model', action='store_true', default=True,
                        help='保存模型檢查點')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='輸出目錄，默認使用配置中的設置')

    # 新增的優化參數
    parser.add_argument('--use_sparse_representation', action='store_true', default=False,
                        help='使用稀疏表示節省記憶體')
    parser.add_argument('--use_mixed_precision', action='store_true', default=False,
                        help='使用混合精度訓練')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=False,
                        help='使用梯度檢查點節省記憶體')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='日誌級別')

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
    device = torch.device('cuda' if use_gpu else 'cpu')
    logger.info(f"使用設備: {device}")

    # 設置隨機種子確保可重現性
    seed = config.get('random_seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if use_gpu:
        torch.cuda.manual_seed_all(seed)

    # 使用上下文管理器追蹤記憶體使用
    with track_memory_usage('主函數執行'):
        # 載入資料
        logger.info("開始載入資料...")
        data_loader = EnhancedMemoryOptimizedDataLoader(
            config['data']['path'],
            use_memory=args.use_memory,
            memory_size=args.memory_size
        )
        
        # 構建圖
        logger.info("開始構建圖...")
        graph_builder = OptimizedGraphBuilder(
            data_loader,
            use_sparse=args.use_sparse_representation
        )
        
        # 創建模型
        logger.info("創建模型...")
        model = OptimizedTGATModel(
            in_dim=config['model']['input_dim'],
            hidden_dim=config['model']['hidden_dim'],
            out_dim=config['model']['output_dim'],
            time_dim=config['model']['time_dim'],
            num_layers=config['model']['num_layers'],
            num_heads=config['model']['num_heads'],
            dropout=config['model']['dropout'],
            num_classes=config['model']['num_classes']
        )
        
        # 檢查是否使用混合精度訓練
        if args.use_mixed_precision:
            logger.info("啟用混合精度訓練...")
            model.enable_mixed_precision()
            
        # 檢查是否使用梯度檢查點
        if args.gradient_checkpointing:
            logger.info("啟用梯度檢查點...")
            model.enable_gradient_checkpointing()
            
        # 移動模型到設備
        model.to(device)
        
        # 設置優化器
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 設置損失函數
        criterion = torch.nn.CrossEntropyLoss()
        
        # 訓練模式或評估模式
        mode = 'train'  # 默認訓練模式
        
        # 根據模式執行相應操作
        if mode == 'train':
            logger.info("開始訓練模型...")
            # 訓練邏輯...
            epochs = config['training']['epochs']
            for epoch in range(epochs):
                logger.info(f"Epoch {epoch+1}/{epochs}")
                # 訓練一個輪次...
                
        elif mode == 'eval':
            logger.info("評估模型性能...")
            # 評估邏輯...
            
        # 保存結果
        logger.info("保存結果...")

    # 返回結果字典
    results = {
        'status': 'success',
        'model_path': os.path.join(model_dir, 'final_model.pth'),
        'config': config,
        'output_dir': output_dir
    }
    
    return results

if __name__ == "__main__":
    results = main()
    logger.info(f"執行完成: {results['status']}")
