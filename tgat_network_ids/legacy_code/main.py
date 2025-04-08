#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TGAT 網路攻擊檢測系統主程式

使用 TGAT (Temporal Graph Attention Network) 模型檢測網路封包中的攻擊行為。
主程序包含:
1. 參數解析
2. 資料加載與預處理
3. 圖結構建立
4. 模型訓練與評估
5. 即時檢測模擬
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

# 導入自定義模組
from data_loader import CICDDoSDataLoader
from graph_builder import DynamicNetworkGraph
from tgat_model import TGAT
from train import TGATTrainer
from visualization import NetworkVisualizer
from utils import (
    set_seed, get_device, load_config, save_config, 
    evaluate_predictions, format_metrics, create_dir, 
    save_results, get_timestamp, time_execution
)

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tgat_ids.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 預設配置
DEFAULT_CONFIG = {
    'data': {
        'path': './data',
        'test_size': 0.2,
        'random_state': 42,
        'batch_size': 1000
    },
    'graph': {
        'temporal_window': 600,  # 10分鐘
    },
    'model': {
        'hidden_dim': 64,
        'out_dim': 64,
        'time_dim': 16,
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.1
    },
    'train': {
        'lr': 0.001,
        'weight_decay': 0.00001,
        'epochs': 50,
        'patience': 10,
        'batch_size': 1000
    },
    'detection': {
        'threshold': 0.7,
        'window_size': 100,
        'update_interval': 1
    },
    'output': {
        'model_dir': './models',
        'result_dir': './results',
        'visualization_dir': './visualizations'
    }
}

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='TGAT 網路攻擊檢測系統')
    
    parser.add_argument('--config', type=str, default='./config.yaml',
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
    
    return parser.parse_args()

def get_config(args):
    """獲取配置"""
    config = DEFAULT_CONFIG.copy()
    
    # 嘗試從配置文件加載
    if args.config and os.path.exists(args.config):
        file_config = load_config(args.config)
        if file_config:
            # 更新配置
            for section, values in file_config.items():
                if section in config:
                    config[section].update(values)
                else:
                    config[section] = values
    
    # 命令行參數覆蓋配置文件
    if args.data_path:
        config['data']['path'] = args.data_path
    if args.model_path:
        config['model_path'] = args.model_path
    if args.seed:
        config['seed'] = args.seed
    
    return config

@time_execution
def load_and_preprocess_data(config):
    """加載和預處理資料"""
    data_config = config['data']
    
    logger.info(f"加載資料集: {data_config['path']}")
    
    # 初始化資料加載器
    data_loader = CICDDoSDataLoader(
        data_path=data_config['path'],
        test_size=data_config['test_size'],
        random_state=data_config['random_state']
    )
    
    # 預處理資料
    features, target = data_loader.preprocess()
    
    # 拆分訓練集和測試集
    X_train, X_test, y_train, y_test = data_loader.split_data()
    
    logger.info(f"資料集處理完成:")
    logger.info(f"  特徵維度: {features.shape[1]}")
    logger.info(f"  訓練集樣本數: {len(y_train)}")
    logger.info(f"  測試集樣本數: {len(y_test)}")
    logger.info(f"  類別數量: {len(set(target))}")
    
    return data_loader, X_train, X_test, y_train, y_test

@time_execution
def build_graph(data_loader, features, labels, indices=None):
    """建立圖結構"""
    # 初始化圖構建器
    graph_builder = DynamicNetworkGraph()
    
    # 獲取時間性邊
    edges = data_loader.get_temporal_edges()
    
    # 如果提供了索引，只使用指定的節點
    if indices is not None:
        node_ids = indices
        node_features = features[indices]
        node_labels = labels[indices] if isinstance(labels, np.ndarray) else labels.iloc[indices].values
        
        # 過濾只包含這些節點的邊
        filtered_edges = []
        for src, dst, time, feat in edges:
            if src in node_ids and dst in node_ids:
                filtered_edges.append((src, dst, time, feat))
        edges = filtered_edges
    else:
        node_ids = list(range(len(features)))
        node_features = features
        node_labels = labels
    
    # 添加節點
    timestamps = [edge[2] for edge in edges]  # 使用邊的時間作為節點時間
    if not timestamps:
        timestamps = [0.0] * len(node_ids)
    else:
        # 確保每個節點都有時間戳記
        while len(timestamps) < len(node_ids):
            timestamps.append(timestamps[-1])
    
    # 添加節點
    graph_builder.add_nodes(node_ids, node_features, timestamps, node_labels)
    
    # 添加邊
    if edges:
        src_nodes = [edge[0] for edge in edges]
        dst_nodes = [edge[1] for edge in edges]
        edge_timestamps = [edge[2] for edge in edges]
        edge_feats = [edge[3] for edge in edges]
        
        graph_builder.add_edges(src_nodes, dst_nodes, edge_timestamps, edge_feats)
    
    # 更新時間圖
    graph = graph_builder.update_temporal_graph()
    
    logger.info(f"構建圖完成:")
    logger.info(f"  節點數量: {graph.num_nodes()}")
    logger.info(f"  邊數量: {graph.num_edges()}")
    
    return graph_builder, graph

@time_execution
def train_model(config, graph, labels, val_graph=None, val_labels=None):
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
    
    # 初始化訓練器
    trainer = TGATTrainer(
        model=model,
        device=device,
        lr=train_config['lr'],
        weight_decay=train_config['weight_decay'],
        save_dir=config['output']['model_dir']
    )
    
    # 訓練模型
    logger.info(f"開始訓練模型:")
    logger.info(f"  學習率: {train_config['lr']}")
    logger.info(f"  權重衰減: {train_config['weight_decay']}")
    logger.info(f"  訓練輪數: {train_config['epochs']}")
    logger.info(f"  早停耐心: {train_config['patience']}")
    
    history = trainer.train(
        train_graph=graph,
        train_labels=labels,
        val_graph=val_graph,
        val_labels=val_labels,
        epochs=train_config['epochs'],
        patience=train_config['patience']
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
    
    return trainer, history, final_model_path

@time_execution
def evaluate_model(trainer, graph, labels, class_names=None, visualize=False, config=None):
    """評估模型"""
    logger.info("開始評估模型")
    
    # 評估模型
    loss, acc, metrics = trainer.evaluate(graph, labels)
    
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
    
    return metrics

@time_execution
def simulate_real_time_detection(config, data_loader, model, device='cpu', visualize=False):
    """模擬即時檢測"""
    logger.info("開始模擬即時檢測")
    
    # 初始化圖構建器
    graph_builder = DynamicNetworkGraph(device=device)
    
    # 初始化視覺化工具
    visualizer = NetworkVisualizer()
    
    # 檢測配置
    detection_config = config['detection']
    threshold = detection_config['threshold']
    window_size = detection_config['window_size']
    
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
            visualizer.visualize_graph(
                g=graph,
                node_labels=node_labels,
                title=f"網路流量圖結構 (批次 {batch_idx})",
                save_path=graph_path
            )
    
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
        train_graph_builder, train_graph = build_graph(data_loader, X_train, y_train)
        
        # 建立測試圖
        logger.info("建立測試圖...")
        test_graph_builder, test_graph = build_graph(data_loader, X_test, y_test)
        
        # 訓練模型
        trainer, history, model_path = train_model(
            config, train_graph, 
            labels=torch.tensor(y_train.values, dtype=torch.long),
            val_graph=test_graph, 
            val_labels=torch.tensor(y_test.values, dtype=torch.long)
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
        test_graph_builder, test_graph = build_graph(data_loader, X_test, y_test)
        
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
        
        trainer = TGATTrainer(
            model=model,
            device=device,
            save_dir=config['output']['model_dir']
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
        
        trainer = TGATTrainer(
            model=model,
            device=device,
            save_dir=config['output']['model_dir']
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
    
    logger.info("TGAT 網路攻擊檢測系統已完成運行")

if __name__ == "__main__":
    main()