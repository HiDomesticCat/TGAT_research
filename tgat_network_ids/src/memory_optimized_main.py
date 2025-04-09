#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
記憶體優化版 TGAT 網路攻擊檢測系統主程式

使用記憶體優化版的組件實現 TGAT (Temporal Graph Attention Network) 模型檢測網路封包中的攻擊行為。
主要優化包括:
1. 增量式資料加載
2. 記憶體映射大型數據集
3. 子圖採樣
4. 混合精度訓練
5. 梯度累積
6. 動態批次大小
7. 主動記憶體管理
"""

import os
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import time
from tqdm import tqdm
import gc

import sys
import os

# 添加當前目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

# 導入記憶體優化工具
from tgat_network_ids.src.utils.memory_utils import (
    clean_memory, memory_usage_decorator, print_memory_usage,
    get_memory_usage, print_optimization_suggestions, MemoryMonitor
)

try:
    # 嘗試使用絕對導入
    # 導入記憶體優化工具
    from tgat_network_ids.src.utils.memory_utils import (
        clean_memory, memory_usage_decorator, print_memory_usage,
        get_memory_usage, print_optimization_suggestions, MemoryMonitor
    )

    # 導入工具函數
    from tgat_network_ids.src.utils.utils import (
        set_seed, get_device, load_config, save_config, 
        evaluate_predictions, format_metrics, create_dir, 
        save_results, get_timestamp, time_execution
    )

    # 導入記憶體優化版模組
    from tgat_network_ids.src.data.memory_optimized_data_loader import MemoryOptimizedDataLoader
    from tgat_network_ids.src.data.memory_optimized_graph_builder import MemoryOptimizedDynamicNetworkGraph
    from tgat_network_ids.src.models.tgat_model import TGAT
    from tgat_network_ids.src.models.memory_optimized_train import MemoryOptimizedTGATTrainer
    from tgat_network_ids.src.visualization.visualization import NetworkVisualizer
except ModuleNotFoundError:
    # 如果絕對導入失敗，嘗試使用相對路徑導入
    print("使用相對路徑導入模組...")
    
    # 導入記憶體優化工具
    sys.path.insert(0, os.path.join(current_dir, '..'))
    
    from utils.memory_utils import (
        clean_memory, memory_usage_decorator, print_memory_usage,
        get_memory_usage, print_optimization_suggestions, MemoryMonitor
    )

    # 導入工具函數
    from utils.utils import (
        set_seed, get_device, load_config, save_config, 
        evaluate_predictions, format_metrics, create_dir, 
        save_results, get_timestamp, time_execution
    )

    # 導入記憶體優化版模組
    from data.memory_optimized_data_loader import MemoryOptimizedDataLoader
    from data.memory_optimized_graph_builder import MemoryOptimizedDynamicNetworkGraph
    from models.tgat_model import TGAT
    from models.memory_optimized_train import MemoryOptimizedTGATTrainer
    from visualization.visualization import NetworkVisualizer

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("memory_optimized_tgat_ids.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='記憶體優化版 TGAT 網路攻擊檢測系統')
    
    parser.add_argument('--config', type=str, default='./memory_optimized_config.yaml',
                        help='配置文件路徑')
    parser.add_argument('--data_path', type=str, default=None,
                        help='資料集路徑')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'detect'], default='train',
                        help='運行模式：訓練、測試或即時檢測')
    parser.add_argument('--model_path', type=str, default=None,
                        help='預訓練模型路徑 (用於測試和檢測模式)')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='啟用視覺化')
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機種子')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID (-1 表示使用 CPU)')
    parser.add_argument('--monitor_memory', action='store_true', default=False,
                        help='啟用記憶體監控')
    
    return parser.parse_args()

def get_config(args):
    """獲取配置"""
    # 嘗試從配置文件加載
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
        if not config:
            logger.error(f"無法加載配置文件: {args.config}")
            config = {}
    else:
        logger.warning(f"配置文件不存在: {args.config}，使用預設配置")
        config = {}
    
    # 命令行參數覆蓋配置文件
    if args.data_path:
        if 'data' not in config:
            config['data'] = {}
        config['data']['path'] = args.data_path
    
    if args.model_path:
        config['model_path'] = args.model_path
    
    if args.seed:
        config['seed'] = args.seed
    
    # 確保必要的配置項存在
    if 'data' not in config:
        config['data'] = {'path': './data', 'test_size': 0.2, 'random_state': 42, 'batch_size': 128}
    
    if 'output' not in config:
        config['output'] = {
            'model_dir': './models',
            'result_dir': './results',
            'visualization_dir': './visualizations',
            'memory_report_dir': './memory_reports',
            'checkpoint_dir': './checkpoints'
        }
    
    # 創建輸出目錄
    for dir_path in config['output'].values():
        create_dir(dir_path)
    
    return config

@memory_usage_decorator
def load_and_preprocess_data(config):
    """加載和預處理資料"""
    logger.info(f"加載資料集: {config['data']['path']}")
    
    # 初始化記憶體優化版資料加載器
    data_loader = MemoryOptimizedDataLoader(config)
    
    # 預處理資料
    features, target = data_loader.preprocess()
    
    # 拆分訓練集和測試集
    X_train, X_test, y_train, y_test = data_loader.split_data()
    
    logger.info(f"資料集處理完成:")
    logger.info(f"  特徵維度: {features.shape[1]}")
    logger.info(f"  訓練集樣本數: {len(y_train)}")
    logger.info(f"  測試集樣本數: {len(y_test)}")
    logger.info(f"  類別數量: {len(set(target))}")
    
    # 清理記憶體
    clean_memory()
    
    return data_loader, X_train, X_test, y_train, y_test

@memory_usage_decorator
def build_graph(config, data_loader, features, labels, indices=None):
    """建立圖結構"""
    # 初始化記憶體優化版圖構建器
    graph_builder = MemoryOptimizedDynamicNetworkGraph(config)
    
    # 獲取時間性邊
    edges = data_loader.get_temporal_edges(max_edges=500000)
    
    # 如果提供了索引，只使用指定的節點
    if indices is not None:
        node_ids = indices
        node_features = features[indices]
        node_labels = labels[indices] if isinstance(labels, np.ndarray) else labels.iloc[indices].values
    else:
        node_ids = list(range(len(features)))
        node_features = features
        node_labels = labels
    
    # 過濾邊並獲取時間戳記
    if edges is not None and len(edges) > 0:
        # 如果提供了索引，過濾只包含這些節點的邊
        if indices is not None:
            filtered_edges = []
            for src, dst, time, feat in edges:
                if src in node_ids and dst in node_ids:
                    filtered_edges.append((src, dst, time, feat))
            edges = filtered_edges
        
        # 使用邊的時間作為節點時間
        timestamps = [edge[2] for edge in edges]
        
        # 確保每個節點都有時間戳記
        if not timestamps:
            timestamps = [0.0] * len(node_ids)
        else:
            while len(timestamps) < len(node_ids):
                timestamps.append(timestamps[-1])
        
        # 添加節點
        graph_builder.add_nodes(node_ids, node_features, timestamps, node_labels)
        
        # 添加邊
        src_nodes = [edge[0] for edge in edges]
        dst_nodes = [edge[1] for edge in edges]
        edge_timestamps = [edge[2] for edge in edges]
        edge_feats = [edge[3] for edge in edges]
        
        graph_builder.add_edges_in_batches(src_nodes, dst_nodes, edge_timestamps, edge_feats)
        
        # 更新時間圖
        graph = graph_builder.update_temporal_graph()
    else:
        # 如果沒有邊或邊數量為0，使用 simulate_stream 方法創建邊
        logger.warning("未獲取到時間性邊，使用 simulate_stream 方法創建邊")
        
        # 創建時間戳記 (使用索引作為時間戳記)
        timestamps = [float(i) for i in range(len(node_ids))]
        
        # 使用 simulate_stream 方法創建邊
        graph = graph_builder.simulate_stream(node_ids, node_features, timestamps, node_labels)
    
    logger.info(f"構建圖完成:")
    logger.info(f"  節點數量: {graph.num_nodes()}")
    logger.info(f"  邊數量: {graph.num_edges()}")
    
    # 清理記憶體
    clean_memory()
    
    return graph_builder, graph

@memory_usage_decorator
def train_model(config, graph, labels, val_graph=None, val_labels=None, class_names=None):
    """訓練模型"""
    model_config = config['model']
    train_config = config['train']
    
    # 獲取設備
    device = get_device()
    
    # 獲取輸入維度
    in_dim = graph.ndata['h'].shape[1]
    
    # 獲取分類數
    num_classes = len(torch.unique(labels))
    
    logger.info(f"初始化 TGAT 模型:")
    logger.info(f"  輸入維度: {in_dim}")
    logger.info(f"  隱藏維度: {model_config['hidden_dim']}")
    logger.info(f"  輸出維度: {model_config['out_dim']}")
    logger.info(f"  時間維度: {model_config['time_dim']}")
    logger.info(f"  層數: {model_config['num_layers']}")
    logger.info(f"  注意力頭數: {model_config['num_heads']}")
    logger.info(f"  丟棄率: {model_config['dropout']}")
    logger.info(f"  分類數: {num_classes}")
    
    # 初始化模型
    model = TGAT(
        in_dim=in_dim,
        hidden_dim=model_config['hidden_dim'],
        out_dim=model_config['out_dim'],
        time_dim=model_config['time_dim'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        dropout=model_config['dropout'],
        num_classes=num_classes
    )
    
    # 確保模型目錄存在
    create_dir(config['output']['model_dir'])
    
    # 初始化記憶體優化版訓練器
    trainer = MemoryOptimizedTGATTrainer(
        model=model,
        config=config,
        device=device
    )
    
    # 訓練模型
    logger.info(f"開始訓練模型:")
    logger.info(f"  學習率: {train_config['lr']}")
    logger.info(f"  權重衰減: {train_config['weight_decay']}")
    logger.info(f"  訓練輪數: {train_config['epochs']}")
    logger.info(f"  早停耐心: {train_config['patience']}")
    logger.info(f"  批次大小: {train_config['batch_size']}")
    logger.info(f"  混合精度訓練: {model_config.get('use_mixed_precision', True)}")
    logger.info(f"  梯度累積: {model_config.get('use_gradient_accumulation', True)}")
    logger.info(f"  梯度累積步數: {model_config.get('gradient_accumulation_steps', 4)}")
    
    history = trainer.train(
        train_graph=graph,
        train_labels=labels,
        val_graph=val_graph,
        val_labels=val_labels,
        epochs=train_config['epochs'],
        patience=train_config['patience'],
        class_names=class_names
    )
    
    # 儲存訓練歷史
    create_dir(config['output']['result_dir'])
    timestamp = get_timestamp()
    history_path = os.path.join(config['output']['result_dir'], f'history_{timestamp}.json')
    save_results(history, history_path)
    
    # 儲存最終模型
    final_model_path = os.path.join(config['output']['model_dir'], f'model_{timestamp}.pt')
    trainer.save_model(final_model_path)
    logger.info(f"模型已儲存至: {final_model_path}")
    
    # 清理記憶體
    clean_memory()
    
    return trainer, history, final_model_path

@memory_usage_decorator
def evaluate_model(trainer, graph, labels, class_names=None, visualize=False, config=None):
    """評估模型"""
    logger.info("開始評估模型")
    
    # 評估模型
    loss, acc, metrics = trainer.evaluate(graph, labels, class_names)
    
    logger.info(f"模型評估結果:")
    logger.info(f"  損失: {loss:.4f}")
    logger.info(f"  準確率: {acc:.4f}")
    logger.info(f"  F1 分數 (加權): {metrics['f1_weighted']:.4f}")
    
    # 格式化詳細指標輸出
    detailed_metrics = format_metrics(metrics)
    logger.info(f"\n{detailed_metrics}")
    
    # 視覺化混淆矩陣
    if visualize and config:
        create_dir(config['output']['visualization_dir'])
        timestamp = get_timestamp()
        cm_path = os.path.join(config['output']['visualization_dir'], f'confusion_matrix_{timestamp}.png')
        
        trainer.plot_confusion_matrix(
            graph, labels, 
            class_names=class_names, 
            save_path=cm_path
        )
        
        # 視覺化訓練歷史
        history_path = os.path.join(config['output']['visualization_dir'], f'training_history_{timestamp}.png')
        trainer.plot_training_history(save_path=history_path)
    
    # 清理記憶體
    clean_memory()
    
    return metrics

@memory_usage_decorator
def simulate_real_time_detection(config, data_loader, model, device='cpu', visualize=False):
    """模擬即時檢測"""
    logger.info("開始模擬即時檢測")
    
    # 初始化記憶體優化版圖構建器
    graph_builder = MemoryOptimizedDynamicNetworkGraph(config, device=device)
    
    # 初始化視覺化工具
    visualizer = NetworkVisualizer()
    
    # 檢測配置
    detection_config = config.get('detection', {})
    threshold = detection_config.get('threshold', 0.7)
    window_size = detection_config.get('window_size', 50)
    use_sliding_window = detection_config.get('use_sliding_window', True)
    sliding_window_size = detection_config.get('sliding_window_size', 100)
    sliding_window_step = detection_config.get('sliding_window_step', 50)
    
    # 準備儲存檢測結果
    timestamps = []
    attack_scores = []
    detected_attacks = []
    
    # 模擬批次數量
    num_batches = 20
    
    for batch_idx in tqdm(range(num_batches), desc="即時檢測模擬"):
        # 獲取批次資料
        batch_features, batch_labels, batch_indices = data_loader.get_sample_batch(
            batch_size=config['data']['batch_size']
        )
        
        # 模擬流式資料，建立動態圖
        graph = graph_builder.simulate_stream(
            node_ids=batch_indices,
            features=batch_features,
            timestamps=[float(i + batch_idx * 100) for i in range(len(batch_indices))],
            labels=batch_labels
        )
        
        # 使用模型進行預測
        model.eval()
        with torch.no_grad():
            logits = model(graph)
            probs = torch.softmax(logits, dim=1)
            
            # 獲取攻擊類別的概率
            # 假設索引 0 是正常流量，其他都是攻擊
            attack_prob = 1.0 - probs[:, 0] if probs.shape[1] > 1 else probs[:, 0]
            
            # 檢測攻擊
            detected = (attack_prob > threshold).cpu().numpy()
            
            # 記錄結果
            batch_time = time.time()
            timestamps.append(batch_time)
            
            # 使用最大攻擊概率作為批次的異常分數
            max_attack_prob = attack_prob.max().item()
            attack_scores.append(max_attack_prob)
            
            if detected.any():
                detected_attacks.append(batch_idx)
                logger.warning(f"批次 {batch_idx}: 檢測到攻擊！最高攻擊概率: {max_attack_prob:.4f}")
            else:
                logger.info(f"批次 {batch_idx}: 正常流量。最高攻擊概率: {max_attack_prob:.4f}")
        
        # 視覺化檢測結果
        if visualize and (batch_idx + 1) % 5 == 0:
            create_dir(config['output']['visualization_dir'])
            timestamp = get_timestamp()
            
            # 視覺化檢測結果
            detection_path = os.path.join(
                config['output']['visualization_dir'], 
                f'detection_{timestamp}.png'
            )
            
            visualizer.visualize_attack_detection(
                timestamps=list(range(len(attack_scores))),
                scores=attack_scores,
                threshold=threshold,
                attack_indices=detected_attacks,
                title="即時網路攻擊檢測結果",
                save_path=detection_path
            )
            
            # 視覺化當前圖結構
            graph_path = os.path.join(
                config['output']['visualization_dir'], 
                f'graph_{timestamp}.png'
            )
            
            node_labels = graph.ndata['label'].cpu().numpy()
            # 確保圖在 CPU 上再進行視覺化
            graph_cpu = graph.cpu() if graph.device.type != 'cpu' else graph
            visualizer.visualize_graph(
                g=graph_cpu,
                node_labels=node_labels,
                title=f"網路流量圖結構 (批次 {batch_idx})",
                save_path=graph_path
            )
        
        # 清理記憶體
        clean_memory()
    
    # 視覺化最終檢測結果
    if visualize:
        create_dir(config['output']['visualization_dir'])
        timestamp = get_timestamp()
        
        final_detection_path = os.path.join(
            config['output']['visualization_dir'], 
            f'final_detection_{timestamp}.png'
        )
        
        visualizer.visualize_attack_detection(
            timestamps=list(range(len(attack_scores))),
            scores=attack_scores,
            threshold=threshold,
            attack_indices=detected_attacks,
            title="即時網路攻擊檢測最終結果",
            save_path=final_detection_path
        )
    
    return timestamps, attack_scores, detected_attacks

def main():
    """主函數"""
    # 解析命令行參數
    args = parse_args()
    
    # 獲取配置
    config = get_config(args)
    
    # 設置隨機種子
    seed = config.get('seed', args.seed)
    set_seed(seed)
    
    # 設置設備
    if args.gpu >= 0 and torch.cuda.is_available():
        device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'
    
    logger.info(f"使用裝置: {device}")
    
    # 啟用記憶體監控
    memory_monitor = None
    if args.monitor_memory:
        memory_report_dir = config['output'].get('memory_report_dir', './memory_reports')
        create_dir(memory_report_dir)
        memory_monitor = MemoryMonitor(
            interval=10,
            report_dir=memory_report_dir,
            enable_gpu=(device != 'cpu')
        )
        memory_monitor.start()
        logger.info("已啟用記憶體監控")
    
    try:
        # 加載和預處理資料
        data_loader, X_train, X_test, y_train, y_test = load_and_preprocess_data(config)
        
        # 獲取類別名稱
        attack_types = data_loader.get_attack_types()
        class_names = [attack_types[i] for i in sorted(attack_types.keys())]
        
        logger.info(f"攻擊類型: {attack_types}")
        
        # 根據模式執行不同操作
        if args.mode == 'train':
            # 建立訓練圖
            logger.info("建立訓練圖...")
            train_graph_builder, train_graph = build_graph(config, data_loader, X_train, y_train)
            
            # 建立測試圖
            logger.info("建立測試圖...")
            test_graph_builder, test_graph = build_graph(config, data_loader, X_test, y_test)
            
            # 訓練模型
            trainer, history, model_path = train_model(
                config, train_graph, 
                labels=torch.tensor(y_train.values, dtype=torch.long),
                val_graph=test_graph, 
                val_labels=torch.tensor(y_test.values, dtype=torch.long),
                class_names=class_names
            )
            
            # 評估模型
            metrics = evaluate_model(
                trainer, test_graph, 
                labels=torch.tensor(y_test.values, dtype=torch.long),
                class_names=class_names,
                visualize=args.visualize,
                config=config
            )
            
            # 儲存評估結果
            result_path = os.path.join(
                config['output']['result_dir'], 
                f'evaluation_{get_timestamp()}.json'
            )
            save_results(metrics, result_path)
            
            # 即時檢測模擬
            logger.info("模擬即時檢測...")
            timestamps, attack_scores, detected_attacks = simulate_real_time_detection(
                config, data_loader, trainer.model, 
                device=device, visualize=args.visualize
            )
            
            logger.info(f"檢測到 {len(detected_attacks)} 個可能的攻擊批次")
            
        elif args.mode == 'test':
            # 確保模型路徑存在
            model_path = args.model_path or config.get('model_path')
            if not model_path or not os.path.exists(model_path):
                logger.error(f"測試模式需要有效的模型路徑，請提供 --model_path 參數")
                return
            
            # 建立測試圖
            logger.info("建立測試圖...")
            test_graph_builder, test_graph = build_graph(config, data_loader, X_test, y_test)
            
            # 載入模型
            in_dim = test_graph.ndata['h'].shape[1]
            num_classes = len(torch.unique(torch.tensor(y_test.values)))
            
            model = TGAT(
                in_dim=in_dim,
                hidden_dim=config['model']['hidden_dim'],
                out_dim=config['model']['out_dim'],
                time_dim=config['model']['time_dim'],
                num_layers=config['model']['num_layers'],
                num_heads=config['model']['num_heads'],
                dropout=config['model']['dropout'],
                num_classes=num_classes
            )
            
            trainer = MemoryOptimizedTGATTrainer(
                model=model,
                config=config,
                device=device
            )
            
            trainer.load_model(model_path)
            logger.info(f"已載入模型: {model_path}")
            
            # 評估模型
            metrics = evaluate_model(
                trainer, test_graph, 
                labels=torch.tensor(y_test.values, dtype=torch.long),
                class_names=class_names,
                visualize=args.visualize,
                config=config
            )
            
            # 儲存評估結果
            result_path = os.path.join(
                config['output']['result_dir'], 
                f'evaluation_{get_timestamp()}.json'
            )
            save_results(metrics, result_path)
            
        elif args.mode == 'detect':
            # 確保模型路徑存在
            model_path = args.model_path or config.get('model_path')
            if not model_path or not os.path.exists(model_path):
                logger.error(f"檢測模式需要有效的模型路徑，請提供 --model_path 參數")
                return
            
            # 載入模型
            # 由於我們不知道確切的輸入維度，先用第一個樣本估計
            sample_features, sample_labels, _ = data_loader.get_sample_batch(batch_size=1)
            in_dim = sample_features.shape[1]
            num_classes = len(torch.unique(torch.tensor(sample_labels)))
            
            model = TGAT(
                in_dim=in_dim,
                hidden_dim=config['model']['hidden_dim'],
                out_dim=config['model']['out_dim'],
                time_dim=config['model']['time_dim'],
                num_layers=config['model']['num_layers'],
                num_heads=config['model']['num_heads'],
                dropout=config['model']['dropout'],
                num_classes=num_classes
            )
            
            trainer = MemoryOptimizedTGATTrainer(
                model=model,
                config=config,
                device=device
            )
            
            trainer.load_model(model_path)
            logger.info(f"已載入模型: {model_path}")
            
            # 即時檢測模擬
            logger.info("模擬即時檢測...")
            timestamps, attack_scores, detected_attacks = simulate_real_time_detection(
                config, data_loader, trainer.model, 
                device=device, visualize=args.visualize
            )
            
            logger.info(f"檢測到 {len(detected_attacks)} 個可能的攻擊批次")
        
        logger.info("記憶體優化版 TGAT 網路攻擊檢測系統已完成運行")
        
        # 打印記憶體使用情況
        print_memory_usage()
        
        # 打印記憶體優化建議
        print_optimization_suggestions()
        
    finally:
        # 停止記憶體監控
        if memory_monitor:
            memory_monitor.stop()
            logger.info("已停止記憶體監控")

if __name__ == "__main__":
    main()
