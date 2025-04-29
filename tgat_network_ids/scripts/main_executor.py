#!/usr/bin/env python
# coding: utf-8 -*-

"""
統一版 TGAT 網路入侵偵測系統執行腳本

整合所有核心功能與增強特性：
- 處理 'train'、'eval'、'predict' 模式。
- 支援記憶體優化（稀疏表示、混合精度、梯度檢查點）。
- 整合自適應時間窗口與進階圖採樣（可選）。
- 使用專用的 TrainingEngine 進行訓練與評估。
- 提供清晰的參數解析與配置處理。
- 包含記憶體監控與工具函數。
"""

import os
import sys
import yaml
import argparse
import logging
import time
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import json
import gc

# --- 專案設定 ---
# 確保可以導入專案模組
# 假設此腳本位於 'scripts' 目錄下
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- 模組導入 ---
try:
    from src.data.optimized_data_loader import EnhancedMemoryOptimizedDataLoader as OptimizedDataLoader
    from src.data.optimized_graph_builder import OptimizedGraphBuilder
    from src.data.advanced_sampling import AdvancedGraphSampler
    from src.data.adaptive_window import AdaptiveWindowManager as AdaptiveTimeWindow
    from src.models.optimized_tgat_model import OptimizedTGATModel
    from src.models.training_engine import TrainingEngine
    from src.utils.memory_utils import track_memory_usage, MemoryMonitor, limit_gpu_memory, clean_memory, print_memory_usage
    from src.utils.enhanced_metrics import evaluate_nids_metrics, plot_nids_metrics
    from src.utils.utils import set_seed, load_config, save_config, create_dir, get_timestamp, save_results, format_metrics
    from src.visualization.visualization import NetworkVisualizer # 為潛在的視覺化需求導入
except ModuleNotFoundError as e:
    print(f"錯誤：導入模組失敗: {e}")
    print("請確認此腳本是從 'scripts' 目錄執行，或者專案根目錄已在 PYTHONPATH 中。")
    sys.exit(1)

# --- 日誌設定 ---
# 將日誌檔名設定移至 main 函數中，以便使用輸出目錄
logger = logging.getLogger(__name__)

# --- 工具函數 ---
def safe_float_convert(value, default=0.0):
    """安全地將值轉換為浮點數，處理字串和潛在錯誤。"""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            # 處理科學記數法，例如 '5e-5'
            return float(value)
        except ValueError:
            logger.warning(f"無法將字串 '{value}' 轉換為浮點數，使用預設值: {default}")
            return default
    logger.warning(f"無法將類型 {type(value)} 轉換為浮點數，使用預設值: {default}")
    return default

def safe_int_convert(value, default=0):
    """安全地將值轉換為整數，處理字串和潛在錯誤。"""
    if isinstance(value, int):
        return value
    if isinstance(value, (float, str)):
        try:
            # 先轉換為浮點數以處理 "1.0" 這樣的字串，再轉換為整數
            return int(float(value))
        except (ValueError, TypeError):
            logger.warning(f"無法將 '{value}' 轉換為整數，使用預設值: {default}")
            return default
    logger.warning(f"無法將類型 {type(value)} 轉換為整數，使用預設值: {default}")
    return default

def get_nested_config(cfg, key_path, default=None):
    """安全地讀取巢狀配置值。"""
    keys = key_path.split('.')
    val = cfg
    try:
        for key in keys:
            val = val[key]
        return val
    except (KeyError, TypeError):
        return default

# --- 參數解析 ---
def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='統一版 TGAT 網路入侵偵測系統執行器')

    # --- 核心參數 ---
    parser.add_argument('--config', type=str, default='config/memory_optimized_config.yaml',
                        help='配置文件路徑（相對於專案根目錄）')
    parser.add_argument('--data_path', type=str, default=None,
                        help='資料路徑，覆蓋配置')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'predict'],
                        help='執行模式')
    parser.add_argument('--model_path', type=str, default=None,
                        help='預訓練模型路徑（用於 eval/predict 模式）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='基礎輸出目錄，覆蓋配置。若為 None，則使用時間戳。')
    parser.add_argument('--seed', type=int, default=None, help='隨機種子，覆蓋配置')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='日誌級別')

    # --- 硬體與效能 ---
    # 使用 BooleanOptionalAction 提供更清晰的標誌用法 (--flag / --no-flag)
    parser.add_argument('--use_gpu', action=argparse.BooleanOptionalAction, default=None,
                        help='強制啟用/禁用 GPU 使用')
    parser.add_argument('--gpu_id', type=int, default=None, help='GPU ID，覆蓋配置')
    parser.add_argument('--num_workers', type=int, default=None, help='資料載入器工作線程數')
    parser.add_argument('--pin_memory', action=argparse.BooleanOptionalAction, default=None,
                         help='為資料載入使用固定記憶體')

    # --- 訓練特定參數 ---
    parser.add_argument('--epochs', type=int, default=None, help='訓練輪數')
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=None, help='學習率')
    parser.add_argument('--weight_decay', type=float, default=None, help='權重衰減 (L2 懲罰)')
    parser.add_argument('--patience', type=int, default=None, help='早停耐心值')

    # --- 記憶體優化 ---
    parser.add_argument('--use_sparse_representation', action=argparse.BooleanOptionalAction, default=None,
                        help='使用稀疏圖表示')
    parser.add_argument('--use_mixed_precision', action=argparse.BooleanOptionalAction, default=None,
                        help='使用混合精度訓練（需要 GPU）')
    parser.add_argument('--use_gradient_checkpointing', action=argparse.BooleanOptionalAction, default=None,
                         help='使用梯度檢查點')
    parser.add_argument('--use_gradient_accumulation', action=argparse.BooleanOptionalAction, default=None,
                         help='使用梯度累積')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None,
                         help='梯度累積步數')
    parser.add_argument('--use_dynamic_batch_size', action=argparse.BooleanOptionalAction, default=None,
                         help='根據記憶體動態調整批次大小')
    parser.add_argument('--limit_gpu_memory', type=int, default=0,
                        help='限制 GPU 記憶體使用量 (MB)，0 表示不限制')
    parser.add_argument('--monitor_memory', action=argparse.BooleanOptionalAction, default=None,
                         help='在執行期間啟用記憶體監控')

    # --- 進階功能 ---
    # parser.add_argument('--use_adaptive_window', action=argparse.BooleanOptionalAction, default=None,
    #                     help='使用自適應時間窗口（功能可能需要與 TrainingEngine 整合）')
    # parser.add_argument('--use_advanced_sampling', action=argparse.BooleanOptionalAction, default=None,
    #                     help='使用進階圖採樣策略（功能可能需要與 TrainingEngine 整合）')
    # parser.add_argument('--sampling_method', type=str, default=None,
    #                     choices=['graphsaint', 'cluster-gcn', 'frontier', 'historical'],
    #                     help='圖採樣方法')
    # parser.add_argument('--sample_size', type=int, default=None, help='子圖樣本大小')
    # parser.add_argument('--use_memory_mechanism', action=argparse.BooleanOptionalAction, default=None, # 從 --use_memory 更名
    #                     help='在模型中啟用記憶體機制')
    # parser.add_argument('--memory_size', type=int, default=None, help='模型的記憶體大小')
    # parser.add_argument('--use_position_embedding', action=argparse.BooleanOptionalAction, default=None,
    #                      help='在模型中使用位置嵌入')

    # --- 輸出與視覺化 ---
    parser.add_argument('--visualize', action=argparse.BooleanOptionalAction, default=None,
                        help='生成視覺化圖表')
    parser.add_argument('--save_model', action=argparse.BooleanOptionalAction, default=None,
                         help='保存模型檢查點與最佳模型')

    return parser.parse_args()

def update_config(config, args):
    """使用 argparse 的值更新配置字典。"""
    logger.debug("根據命令行參數更新配置...")
    # --- 基本 ---
    if args.data_path: config.setdefault('data', {})['path'] = args.data_path
    if args.seed is not None: config.setdefault('system', {})['seed'] = args.seed
    if args.output_dir: config.setdefault('output', {})['base_dir'] = args.output_dir # 使用基礎目錄

    # --- 硬體與效能 ---
    if args.use_gpu is not None: config.setdefault('system', {})['device'] = 'cuda' if args.use_gpu else 'cpu'
    if args.gpu_id is not None: config.setdefault('system', {})['gpu_id'] = args.gpu_id
    if args.num_workers is not None: config.setdefault('system', {})['num_workers'] = args.num_workers
    if args.pin_memory is not None: config.setdefault('system', {})['pin_memory'] = args.pin_memory

    # --- 訓練 ---
    if args.epochs is not None: config.setdefault('train', {})['epochs'] = args.epochs
    if args.batch_size is not None: config.setdefault('train', {})['batch_size'] = args.batch_size
    if args.learning_rate is not None: config.setdefault('train', {})['learning_rate'] = args.learning_rate
    if args.weight_decay is not None: config.setdefault('train', {})['weight_decay'] = args.weight_decay
    if args.patience is not None: config.setdefault('train', {})['patience'] = args.patience

    # --- 記憶體優化 ---
    if args.use_sparse_representation is not None: config.setdefault('graph', {})['use_sparse_representation'] = args.use_sparse_representation
    if args.use_mixed_precision is not None: config.setdefault('model', {})['use_mixed_precision'] = args.use_mixed_precision
    if args.use_gradient_checkpointing is not None: config.setdefault('model', {})['use_gradient_checkpointing'] = args.use_gradient_checkpointing
    if args.use_gradient_accumulation is not None: config.setdefault('model', {})['use_gradient_accumulation'] = args.use_gradient_accumulation
    if args.gradient_accumulation_steps is not None: config.setdefault('model', {})['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    if args.use_dynamic_batch_size is not None: config.setdefault('train', {})['use_dynamic_batch_size'] = args.use_dynamic_batch_size
    if args.limit_gpu_memory > 0: config.setdefault('system', {})['limit_gpu_memory'] = args.limit_gpu_memory

    # --- 進階功能 (目前註解掉，可根據需要取消註解並整合) ---
    # if args.use_adaptive_window is not None: config.setdefault('adaptive_window', {})['enabled'] = args.use_adaptive_window
    # if args.use_advanced_sampling is not None: config.setdefault('sampling', {})['enabled'] = args.use_advanced_sampling
    # if args.sampling_method: config.setdefault('sampling', {})['method'] = args.sampling_method
    # if args.sample_size: config.setdefault('sampling', {})['sample_size'] = args.sample_size
    # if args.use_memory_mechanism is not None: config.setdefault('model', {})['use_memory'] = args.use_memory_mechanism
    # if args.memory_size: config.setdefault('model', {})['memory_size'] = args.memory_size
    # if args.use_position_embedding is not None: config.setdefault('model', {})['use_position_embedding'] = args.use_position_embedding

    # --- 輸出與視覺化 ---
    if args.visualize is not None: config.setdefault('output', {})['visualize'] = args.visualize
    if args.save_model is not None: config.setdefault('output', {})['save_model'] = args.save_model
    if args.monitor_memory is not None: config.setdefault('system', {})['monitor_memory'] = args.monitor_memory

    # --- 確保預設區段存在 ---
    for section in ['data', 'graph', 'model', 'train', 'output', 'system']:
        config.setdefault(section, {})

    # --- 設定預設輸出目錄（如果未提供） ---
    config['output'].setdefault('model_dir', 'models') # 使用相對路徑
    config['output'].setdefault('result_dir', 'results')
    config['output'].setdefault('visualization_dir', 'visualizations')
    config['output'].setdefault('memory_report_dir', 'memory_reports')
    config['output'].setdefault('checkpoint_dir', 'checkpoints')

    return config

def setup_environment(config):
    """根據配置設定環境。"""
    # 設定日誌級別
    log_level = get_nested_config(config, 'system.log_level', 'INFO').upper()
    logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))

    # 設定隨機種子
    seed = safe_int_convert(get_nested_config(config, 'system.seed', 42), 42)
    set_seed(seed)
    logger.info(f"隨機種子設定為: {seed}")

    # 決定裝置
    use_gpu_config = get_nested_config(config, 'system.device', 'cuda') == 'cuda'
    use_gpu = use_gpu_config and torch.cuda.is_available()
    gpu_id = safe_int_convert(get_nested_config(config, 'system.gpu_id', 0), 0)
    device_str = f'cuda:{gpu_id}' if use_gpu else 'cpu'
    device = torch.device(device_str)
    logger.info(f"使用裝置: {device}")
    if use_gpu:
        logger.info(f"CUDA 裝置名稱: {torch.cuda.get_device_name(gpu_id)}")

    # 限制 GPU 記憶體（如果已配置）
    limit_mb = safe_int_convert(get_nested_config(config, 'system.limit_gpu_memory', 0), 0)
    if limit_mb > 0 and use_gpu:
        limit_gpu_memory(limit_mb, gpu_id)

    # 創建輸出目錄（相對於基礎目錄）
    base_output_dir = get_nested_config(config, 'output.base_dir')
    if not base_output_dir:
         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
         base_output_dir = os.path.join(project_root, 'output', f'run_{timestamp}') # 預設基礎目錄
         config['output']['base_dir'] = base_output_dir # 存回配置中

    output_dirs = {}
    for key, default_subdir in [
        ('model_dir', 'models'), ('result_dir', 'results'),
        ('visualization_dir', 'visualizations'), ('memory_report_dir', 'memory_reports'),
        ('checkpoint_dir', 'checkpoints')]:
        # 使用配置中的路徑（如果絕對），否則相對於基礎目錄
        subdir_path_config = get_nested_config(config, f'output.{key}', default_subdir)
        if not os.path.isabs(subdir_path_config):
             dir_path = os.path.join(base_output_dir, subdir_path_config)
        else:
            dir_path = subdir_path_config
        create_dir(dir_path)
        output_dirs[key] = dir_path
        config['output'][key] = dir_path # 用絕對路徑更新配置

    # 設定日誌文件處理器以使用輸出目錄
    log_file_path = os.path.join(output_dirs['base_dir'], log_filename) # log_filename 在全域定義
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8') # 指定 UTF-8 編碼
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    # 移除可能已存在的預設處理器，避免重複記錄到文件
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        if isinstance(h, logging.FileHandler):
            root_logger.removeHandler(h)
    root_logger.addHandler(file_handler) # 添加到根日誌記錄器

    # 保存最終生效的配置
    save_config(config, os.path.join(output_dirs['base_dir'], 'effective_config.yaml'))
    logger.info(f"最終生效的配置已保存至 {output_dirs['base_dir']}")

    return config, device, output_dirs

@track_memory_usage("主執行函數") # 使用繁體中文
def main():
    """主要執行函數"""
    args = parse_args()

    # 加載和更新配置
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        logger.error(f"配置文件錯誤: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"解析配置文件錯誤: {e}")
        sys.exit(1)

    config = update_config(config, args)

    # 設定環境、裝置和輸出目錄
    config, device, output_dirs = setup_environment(config)

    logger.info("--- 開始執行 TGAT IDS ---")
    logger.info(f"模式: {args.mode.upper()}")
    logger.info(f"輸出目錄: {output_dirs['base_dir']}")

    # --- 開始記憶體監控 ---
    memory_monitor = None
    if get_nested_config(config, 'system.monitor_memory', False):
        memory_monitor = MemoryMonitor(
            interval=safe_int_convert(get_nested_config(config, 'system.memory_monitor_interval', 60), 60),
            report_dir=output_dirs['memory_report_dir'],
            enable_gpu=(device.type == 'cuda')
        )
        memory_monitor.start()
        logger.info("記憶體監控已啟動。")

    results = {}
    model = None
    trainer = None

    try:
        # --- 資料載入與預處理 ---
        logger.info("初始化資料載入器...")
        # 傳遞整個配置對象，DataLoader 建構子需要它
        data_loader = OptimizedDataLoader(config)
        logger.info("載入資料...")
        # 如果載入預處理資料失敗，load_data 可能返回 None
        if data_loader.load_data() is None and args.mode != 'predict': # predict 模式可能不需要 df
             logger.info("可能需要重新載入或預處理。")
             # 假設 preprocess 會處理這種情況或在需要時報錯
        logger.info("預處理資料...")
        features, target = data_loader.preprocess()
        if features is None or target is None:
             raise ValueError("資料預處理未能返回特徵或目標。")
        logger.info("分割資料...")
        X_train, X_test, y_train, y_test = data_loader.split_data()
        attack_types = data_loader.get_attack_types()
        # 處理標籤編碼器可能產生的非連續類別索引
        unique_target_values = sorted(target.unique())
        class_names = [attack_types.get(i, f"類別_{i}") for i in unique_target_values] # 使用 .get 以確保安全
        logger.info(f"找到的類別: {class_names}")
        num_classes = len(unique_target_values)
        logger.info(f"類別數量: {num_classes}")

        # --- 圖建立 ---
        logger.info("初始化圖建立器...")
        # 確保圖建立器獲取必要的配置部分
        graph_builder_config = {k: config[k] for k in ['data', 'graph', 'model', 'system'] if k in config}
        graph_builder = OptimizedGraphBuilder(graph_builder_config, device=str(device))
        # 注意：一次性建立完整圖可能非常耗記憶體。
        # 根據 TrainingEngine 的不同，圖的建立可能在每個批次中動態進行。
        # 目前，我們先建立測試圖用於評估。
        logger.info("建立測試圖（用於評估/預測）...")
        # 假設 build_graph 存在於 graph_builder 中
        test_graph = graph_builder.build_graph(X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test,
                                              y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test)
        if not isinstance(test_graph, dgl.DGLGraph):
             raise TypeError("圖建立器未返回 DGLGraph。")
        logger.info(f"測試圖建立完成: {test_graph.num_nodes()} 節點, {test_graph.num_edges()} 邊")


        # --- 模型初始化 ---
        logger.info("初始化模型...")
        model_config = config.setdefault('model', {})
        input_dim = features.shape[1] # 從實際特徵獲取維度
        logger.info(f"確定的 input_dim={input_dim}, num_classes={num_classes}")

        model = OptimizedTGATModel(
            in_dim=input_dim,
            hidden_dim=safe_int_convert(get_nested_config(model_config, 'hidden_dim', 64), 64),
            out_dim=safe_int_convert(get_nested_config(model_config, 'out_dim', 64), 64),
            time_dim=safe_int_convert(get_nested_config(model_config, 'time_dim', 16), 16),
            num_layers=safe_int_convert(get_nested_config(model_config, 'num_layers', 2), 2),
            num_heads=safe_int_convert(get_nested_config(model_config, 'num_heads', 4), 4),
            dropout=safe_float_convert(get_nested_config(model_config, 'dropout', 0.2), 0.2),
            num_classes=num_classes
            # 根據需要從配置傳遞其他模型特定標誌
            # use_memory=get_nested_config(model_config, 'use_memory', False),
            # memory_size=safe_int_convert(get_nested_config(model_config, 'memory_size', 1000), 1000),
            # use_position_embedding=get_nested_config(model_config, 'use_position_embedding', False)
        )

        # 應用記憶體優化到模型
        if get_nested_config(config, 'model.use_mixed_precision', False): model.enable_mixed_precision()
        if get_nested_config(config, 'model.use_gradient_checkpointing', False): model.enable_gradient_checkpointing()
        if get_nested_config(config, 'model.use_gradient_accumulation', False):
            steps = safe_int_convert(get_nested_config(config, 'model.gradient_accumulation_steps', 4), 4)
            model.enable_gradient_accumulation(steps)

        model.to(device)
        logger.info(f"模型已初始化: {model.__class__.__name__}")
        print_memory_usage() # 記錄模型初始化後的記憶體

        # --- 優化器與損失函數 ---
        learning_rate = safe_float_convert(get_nested_config(config, 'train.learning_rate', 0.001), 0.001)
        weight_decay = safe_float_convert(get_nested_config(config, 'train.weight_decay', 5e-5), 5e-5)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = torch.nn.CrossEntropyLoss()
        logger.info(f"優化器: Adam (學習率={learning_rate}, 權重衰減={weight_decay})")

        # --- 模式執行 ---
        mode = args.mode
        trainer = None # 初始化 trainer

        if mode == 'train':
            logger.info("--- 開始訓練模式 ---")
            # 注意：這裡假設訓練圖已建立或將在訓練引擎內部動態建立
            logger.info("建立訓練圖...")
            train_graph = graph_builder.build_graph(X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train,
                                                   y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train)
            if not isinstance(train_graph, dgl.DGLGraph):
                raise TypeError("圖建立器未返回 DGLGraph。")
            logger.info(f"訓練圖建立完成: {train_graph.num_nodes()} 節點, {train_graph.num_edges()} 邊")

            # 初始化訓練引擎
            trainer = TrainingEngine(model, optimizer, criterion, device, config)

            # 訓練模型
            # TrainingEngine 需要 data_loader 和 graph_builder
            # 注意：目前的 TrainingEngine 可能需要調整以接受 graph_builder
            # 為了保持一致性，我們先傳入已建立的圖
            training_results = trainer.train(data_loader, graph_builder, validation_loader=None) # 傳遞必要的組件

            results = training_results
            best_acc = results.get('best_val_metrics', {}).get('accuracy', 'N/A')
            logger.info(f"訓練完成。最佳驗證準確率: {best_acc if best_acc != 'N/A' else '未記錄':.4f}")

            # 可選：訓練後在測試集上進行最終評估
            logger.info("在測試集上評估最終（最佳）模型...")
            # 載入最佳模型進行最終評估
            if get_nested_config(config, 'output.save_model', True):
                 best_model_path = os.path.join(output_dirs['model_dir'], 'best_model.pt')
                 if os.path.exists(best_model_path):
                     if trainer.load_checkpoint(best_model_path): # 使用 trainer 的載入方法
                          logger.info(f"已載入最佳模型 {best_model_path} 進行最終評估。")
                     else:
                          logger.warning("載入最佳模型失敗，將評估最後狀態。")
                 else:
                     logger.warning("未找到最佳模型文件，將評估最後狀態。")

            # 確保使用正確的標籤格式（Tensor）
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.long) if isinstance(y_test, pd.Series) else torch.tensor(y_test, dtype=torch.long)
            test_loss, test_metrics = trainer.validate(data_loader, graph_builder, subset_ratio=1.0) # 這裡需要傳遞測試數據，目前使用 data_loader
            logger.info(f"最終測試集效能 - 損失: {test_loss:.4f}, 準確率: {test_metrics.get('accuracy', 'N/A'):.4f}")
            results['test_metrics'] = test_metrics

            # 保存評估結果
            eval_file_path = os.path.join(output_dirs['result_dir'], f'final_evaluation_{get_timestamp()}.json')
            save_results(test_metrics, eval_file_path) # 使用工具函數


        elif mode == 'eval':
            logger.info("--- 開始評估模式 ---")
            # 確定模型路徑
            model_path = args.model_path or get_nested_config(config, 'model_path', os.path.join(output_dirs['model_dir'], 'best_model.pt'))
            if not model_path or not os.path.exists(model_path):
                logger.error(f"評估模式需要有效的模型路徑。嘗試路徑: {model_path}")
                raise FileNotFoundError(f"找不到模型文件: {model_path}")

            logger.info(f"從以下路徑載入模型: {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            # 檢查 checkpoint 結構
            if 'model_state_dict' in checkpoint:
                 model.load_state_dict(checkpoint['model_state_dict'])
                 logger.info("從 checkpoint['model_state_dict'] 載入模型狀態。")
            elif isinstance(checkpoint, dict) and all(k.startswith('layers') or k.startswith('feat_project') or k.startswith('classifier') for k in checkpoint.keys()):
                 model.load_state_dict(checkpoint) # 假設文件本身就是 state_dict
                 logger.info("直接從文件載入模型狀態。")
            else:
                 logger.error("無法識別的模型文件格式。")
                 raise ValueError("模型文件格式無法識別。")


            # 初始化訓練引擎（僅用於評估方法）
            # 雖然不需要訓練，但 TrainingEngine 包含了方便的 validate 方法
            trainer = TrainingEngine(model, optimizer, criterion, device, config)

            logger.info("在測試集上評估模型...")
            # 確保使用正確的標籤格式（Tensor）
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.long) if isinstance(y_test, pd.Series) else torch.tensor(y_test, dtype=torch.long)
            test_loss, test_metrics = trainer.validate(data_loader, graph_builder, subset_ratio=1.0) # 使用測試分割邏輯
            logger.info(f"評估結果 - 損失: {test_loss:.4f}, 準確率: {test_metrics.get('accuracy', 'N/A'):.4f}")
            logger.info(f"詳細評估報告:\n{format_metrics(test_metrics)}")
            results['test_metrics'] = test_metrics

            # 保存評估結果
            eval_file_path = os.path.join(output_dirs['result_dir'], f'evaluation_{get_timestamp()}.json')
            save_results(test_metrics, eval_file_path)

            # 可視化（如果需要）
            if get_nested_config(config, 'output.visualize', False):
                logger.info("產生視覺化圖表...")
                visualizer = NetworkVisualizer()
                # 產生混淆矩陣等
                cm_path = os.path.join(output_dirs['visualization_dir'], f'confusion_matrix_eval_{get_timestamp()}.png')
                if 'confusion_matrix' in test_metrics:
                     # plot_confusion_matrix 可能需要調整以直接接受矩陣
                     # visualizer.plot_confusion_matrix(test_metrics['confusion_matrix'], class_names=class_names, save_path=cm_path)
                     logger.info(f"混淆矩陣視覺化已保存（需要實現 plot_confusion_matrix(matrix, ...)）。")
                else:
                     logger.warning("評估結果中未找到混淆矩陣。")


        elif mode == 'predict':
            logger.info("--- 開始預測模式 ---")
            model_path = args.model_path or get_nested_config(config, 'model_path', os.path.join(output_dirs['model_dir'], 'best_model.pt'))
            if not model_path or not os.path.exists(model_path):
                logger.error(f"預測模式需要有效的模型路徑。嘗試路徑: {model_path}")
                raise FileNotFoundError(f"找不到模型文件: {model_path}")

            logger.info(f"從以下路徑載入模型: {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                 model.load_state_dict(checkpoint['model_state_dict'])
            else:
                 model.load_state_dict(checkpoint)

            model.eval() # 設置為評估模式
            predictions = []

            # 假設 data_loader 可以提供預測數據（無標籤）
            # 這部分需要根據 data_loader 的實際能力實現
            logger.warning("預測資料的載入/迭代邏輯需要實現。")
            predict_dataloader = data_loader.get_predict_dataloader() # 假設有此方法
            if predict_dataloader is None:
                 logger.error("無法獲取預測資料載入器。")
                 raise NotImplementedError("需要實現 get_predict_dataloader 方法。")

            logger.info("開始對預測資料進行推斷...")
            with torch.no_grad():
                for batch_data in predict_dataloader: # 假設返回批次數據
                    # 假設 batch_data 包含特徵和索引/ID
                    batch_features = batch_data['features'].to(device)
                    # 需要從 batch_data 獲取節點 ID 或索引來建立圖
                    node_ids = batch_data['node_ids'] # 假設存在

                    # 為當前批次建立圖
                    # 注意：這裡可能需要僅包含當前批次節點的圖
                    batch_graph = graph_builder.build_inference_graph(node_ids, batch_features) # 假設有此方法

                    if batch_graph is None or batch_graph.num_nodes() == 0:
                         logger.warning("為預測批次建立的圖為空，跳過。")
                         continue

                    batch_graph = batch_graph.to(device)

                    outputs = model(batch_graph) # 假設模型只需要圖
                    _, predicted = torch.max(outputs, 1)
                    predictions.extend(predicted.cpu().numpy())

            # 保存預測結果
            pred_file_path = os.path.join(output_dirs['result_dir'], f'predictions_{get_timestamp()}.csv')
            # 可能需要將預測結果與原始數據的 ID 或索引對應起來
            pd.DataFrame({'prediction': predictions}).to_csv(pred_file_path, index=False)
            logger.info(f"預測結果已保存至 {pred_file_path}")
            results['predictions_path'] = pred_file_path

    except Exception as e:
        logger.error(f"執行過程中發生錯誤: {e}", exc_info=True) # 記錄 traceback
        results['status'] = 'error'
        results['error_message'] = str(e)
    else:
        results.setdefault('status', 'success') # 確保 status 存在
    finally:
        # --- 停止記憶體監控 ---
        if memory_monitor:
            memory_monitor.stop()
            logger.info("記憶體監控已停止。")

        # --- 清理資源 ---
        logger.info("清理資源...")
        clean_memory(aggressive=True)
        print_memory_usage() # 最終記憶體使用情況
        logger.info("--- TGAT IDS 執行完畢 ---")

    return results

if __name__ == '__main__':
    final_results = main()
    final_status = final_results.get('status', 'unknown')
    logger.info(f"最終執行狀態: {final_status}")
    if 'error_message' in final_results:
        logger.error(f"錯誤詳情: {final_results['error_message']}")
    # 根據狀態碼退出，以便於腳本自動化
    sys.exit(0 if final_status == 'success' else 1)
