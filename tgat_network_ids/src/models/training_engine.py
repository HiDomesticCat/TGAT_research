#!/usr/bin/env python
# coding: utf-8 -*-

"""
TGAT模型訓練引擎

提供完整的模型訓練功能，包括訓練循環、評估、早停、混合精度訓練等優化技術。
支援記憶體優化和性能監控。
"""

import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report, precision_score, recall_score, f1_score # 導入更多指標
)
import dgl # 導入 dgl
from dgl.dataloading import DataLoader as DGLDataLoader, NeighborSampler # 導入 DGL DataLoader 和 Sampler
import torch.nn.functional as F # 導入 F
from tqdm import tqdm # 導入 tqdm

# 導入工具函數 (假設路徑正確)
# 注意：導入路徑可能需要根據您的實際專案結構調整
try:
     from ..utils.utils import save_results, format_metrics
     from ..utils.memory_utils import MemoryMonitor, clean_memory, limit_gpu_memory, print_memory_usage, get_memory_usage
except ImportError:
     print("警告：訓練引擎無法從相對路徑導入工具，嘗試直接導入。")
     # 如果直接執行此文件或結構不同，則回退
     try:
          from utils.utils import save_results, format_metrics
          from utils.memory_utils import MemoryMonitor, clean_memory, limit_gpu_memory, print_memory_usage, get_memory_usage
     except ImportError as ie:
          print(f"直接導入失敗: {ie}。請確保 utils 模組可用。")
          # 定義虛設函數以允許運行
          def save_results(data, path): print(f"虛設 save_results 調用: {path}")
          def format_metrics(metrics): return str(metrics)
          class MemoryMonitor: pass
          def clean_memory(*args, **kwargs): pass
          def limit_gpu_memory(*args, **kwargs): pass
          def print_memory_usage(*args, **kwargs): pass
          def get_memory_usage(*args, **kwargs): return {"cpu_percent": 0, "gpu_used_gb": 0}

logger = logging.getLogger(__name__)

class TrainingEngine:
    """TGAT模型訓練引擎"""

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.Module,
                 device: torch.device,
                 config: Dict[str, Any]):
        """
        初始化訓練引擎

        參數:
            model: 要訓練的模型實例
            optimizer: 優化器實例
            criterion: 損失函數實例
            device: 訓練設備 (torch.device)
            config: 包含所有配置的字典
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config

        # --- 訓練相關配置 ---
        train_config = config.get('train', {})
        self.epochs = int(train_config.get('epochs', 50))
        self.patience = int(train_config.get('patience', 5))
        # DGL DataLoader 的批次大小 (種子節點數量)
        self.batch_size = int(train_config.get('batch_size', 1024))
        # 鄰居採樣數量 (列表長度應等於模型層數)
        self.num_neighbors = train_config.get('num_neighbors', [-1, -1])
        if not isinstance(self.num_neighbors, list): # 確保是列表
             try:
                 # 嘗試從字串解析，例如 "10,5"
                 self.num_neighbors = [int(x.strip()) for x in str(self.num_neighbors).split(',')]
             except:
                 logger.warning(f"無法解析鄰居採樣數量 '{self.num_neighbors}'，將使用預設 [-1, -1]。")
                 self.num_neighbors = [-1, -1]
        # 確保鄰居列表長度與模型層數匹配 (假設模型有 num_layers 屬性)
        model_layers = getattr(model, 'num_layers', len(self.num_neighbors)) # 嘗試獲取模型層數
        if len(self.num_neighbors) != model_layers:
             logger.warning(f"鄰居採樣數量列表長度 ({len(self.num_neighbors)}) 與模型層數 ({model_layers}) 不匹配。將重複使用最後一個採樣數。")
             last_neighbor_count = self.num_neighbors[-1] if self.num_neighbors else -1
             self.num_neighbors = [last_neighbor_count] * model_layers

        # --- 系統相關配置 ---
        system_config = config.get('system', {})
        self.num_workers = int(system_config.get('num_workers', 0)) # DGL DataLoader 工作線程數
        self.pin_memory = bool(system_config.get('pin_memory', False)) and device.type == 'cuda'

        # --- 模型優化配置 ---
        model_config = config.get('model', {})
        self.use_mixed_precision = bool(model_config.get('use_mixed_precision', False)) and device.type == 'cuda'
        self.use_gradient_accumulation = bool(model_config.get('use_gradient_accumulation', False))
        self.gradient_accumulation_steps = int(model_config.get('gradient_accumulation_steps', 1))
        if self.gradient_accumulation_steps <= 0: self.gradient_accumulation_steps = 1
        self.max_grad_norm = float(train_config.get('max_grad_norm', 1.0)) # 梯度裁剪閾值

        # --- 輸出配置 ---
        output_config = config.get('output', {})
        self.model_dir = output_config.get('model_dir', './output/models') # 使用更合理的預設路徑
        self.result_dir = output_config.get('result_dir', './output/results')
        self.checkpoint_dir = output_config.get('checkpoint_dir', './output/checkpoints')

        # 創建目錄
        for directory in [self.model_dir, self.result_dir, self.checkpoint_dir]:
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as e:
                logger.error(f"無法創建目錄 {directory}: {e}")
                # 可以選擇拋出錯誤或繼續（如果目錄已存在則沒問題）

        # 初始化混合精度訓練
        self.scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None
        if self.use_mixed_precision: logger.info("啟用混合精度訓練 (GradScaler)")
        else: logger.info("混合精度訓練未啟用")

        # 訓練狀態跟蹤
        self.current_epoch = 0
        self.global_step = 0
        # 使用 f1_weighted 作為主要監控指標，loss 作為次要指標
        self.best_val_metrics = {'f1_weighted': -1.0, 'loss': float('inf'), 'epoch': 0}
        self.early_stop_counter = 0
        self.training_history = []

        # 學習率調度器
        scheduler_patience = max(1, self.patience // 2) # Scheduler 的耐心值應至少為 1
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max', # 因為我們監控 f1_weighted (越大越好)
            factor=0.5,
            patience=scheduler_patience,
            verbose=True,
            min_lr=1e-7 # 設置最小學習率防止過小
        )
        logger.info(f"學習率調度器: ReduceLROnPlateau (基於驗證F1, 耐心={scheduler_patience}, 最小LR=1e-7)")

        logger.info(f"訓練引擎初始化完成: 設備={device}, 混合精度={self.use_mixed_precision}, 梯度累積={self.use_gradient_accumulation}(步數={self.gradient_accumulation_steps})")
        logger.info(f"DGL DataLoader: 批次大小(種子節點)={self.batch_size}, 鄰居採樣={self.num_neighbors}, 工作線程={self.num_workers}, 固定記憶體={self.pin_memory}")


    def train(self, train_graph: dgl.DGLGraph, val_graph: Optional[dgl.DGLGraph] = None):
        """
        執行完整訓練流程

        參數:
            train_graph: DGL訓練圖 (需包含 'feat' 和 'label' 節點數據)
            val_graph: DGL驗證圖 (需包含 'feat' 和 'label' 節點數據, 可選)

        返回:
            包含訓練結果的字典
        """
        # --- 驗證輸入圖 ---
        if not isinstance(train_graph, dgl.DGLGraph):
            raise TypeError(f"train_graph 必須是 dgl.DGLGraph, 但收到 {type(train_graph)}")
        if 'feat' not in train_graph.ndata or 'label' not in train_graph.ndata:
             raise ValueError("訓練圖必須包含 'feat' 和 'label' 節點數據")
        if val_graph is not None:
             if not isinstance(val_graph, dgl.DGLGraph):
                  raise TypeError(f"val_graph 必須是 dgl.DGLGraph, 但收到 {type(val_graph)}")
             if 'feat' not in val_graph.ndata or 'label' not in val_graph.ndata:
                  raise ValueError("驗證圖必須包含 'feat' 和 'label' 節點數據")
        # 建議圖的主要計算發生在 GPU 上，但 DataLoader 可能需要圖在 CPU
        # DGL DataLoader 會自動處理數據轉移，這裡不需要手動 to('cpu')

        logger.info(f"開始訓練流程: 總輪次={self.epochs}")
        logger.info(f"訓練圖: {train_graph.num_nodes()} 節點, {train_graph.num_edges()} 邊")
        if val_graph:
            logger.info(f"驗證圖: {val_graph.num_nodes()} 節點, {val_graph.num_edges()} 邊")
        else:
            logger.warning("未提供驗證圖，模型選擇和早停將基於訓練集表現（不推薦）。")

        start_time = time.time()

        # 主訓練循環 (從 self.current_epoch 開始，支持斷點續訓)
        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch # 更新當前輪次
            epoch_start_time = time.time()

            # --- 訓練一個輪次 ---
            train_loss, train_metrics = self.train_one_epoch(train_graph)

            # --- 驗證 ---
            if val_graph is not None:
                val_loss, val_metrics = self.validate(val_graph)
                # 使用驗證集的 F1 加權平均值作為主要監控指標
                val_primary_metric = val_metrics.get('f1_weighted', 0.0)
            else:
                # 如果沒有驗證集，使用訓練集指標（並發出警告）
                val_loss = train_loss
                val_metrics = train_metrics
                val_primary_metric = train_metrics.get('f1_weighted', 0.0)

            # 更新學習率調度器 (基於驗證指標)
            self.scheduler.step(val_primary_metric)
            current_lr = self.get_current_lr() # 獲取更新後的學習率

            # 記錄訓練歷史
            epoch_duration = time.time() - epoch_start_time
            epoch_history = {
                'epoch': epoch + 1, # 記錄完成的輪次 (從 1 開始)
                'train_loss': train_loss,
                'train_metrics': train_metrics,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'duration': epoch_duration,
                'learning_rate': current_lr
            }
            self.training_history.append(epoch_history)

            # 格式化指標以便日誌輸出
            formatted_train_metrics = format_metrics(train_metrics)
            formatted_val_metrics = format_metrics(val_metrics)

            # 輸出日誌
            logger.info(f"Epoch {epoch+1}/{self.epochs} 完成:")
            logger.info(f"  訓練 - 損失: {train_loss:.4f}, 指標: {formatted_train_metrics}")
            logger.info(f"  驗證 - 損失: {val_loss:.4f}, 指標: {formatted_val_metrics}")
            logger.info(f"  學習率: {current_lr:.1e}, 耗時: {epoch_duration:.2f}s")

            # 檢查模型是否有改進 (基於驗證指標)
            improved = self._check_improvement(val_metrics, val_loss)
            if improved:
                logger.info(f"*** 發現更好的模型 (Epoch {epoch+1})! 驗證 F1: {val_metrics.get('f1_weighted', 0.0):.4f} ***")
                self.save_best_model() # 保存最佳模型
                self.early_stop_counter = 0 # 重置早停計數器
            else:
                self.early_stop_counter += 1
                logger.info(f"模型未改進，早停計數: {self.early_stop_counter}/{self.patience}")

            # 保存檢查點 (定期保存)
            save_checkpoint_interval = self.config.get('train', {}).get('save_checkpoint_interval', 5)
            if save_checkpoint_interval > 0 and (epoch + 1) % save_checkpoint_interval == 0:
                self.save_checkpoint()

            # 檢查早停條件
            if self.early_stop_counter >= self.patience:
                logger.info(f"達到早停耐心值 ({self.patience})，在第 {epoch+1} 輪次後停止訓練。")
                break # 跳出訓練循環

            # 檢查是否需要調整批次大小（基於記憶體，但調整 DGL DataLoader 較難）
            self._adjust_batch_size_if_needed()

        # 訓練結束
        total_training_time = time.time() - start_time
        logger.info("-" * 50)
        logger.info(f"訓練流程完成，總耗時: {total_training_time:.2f}s")
        logger.info(f"共完成 {self.current_epoch + 1} 個輪次。")
        logger.info(f"記錄的最佳驗證指標: F1={self.best_val_metrics['f1_weighted']:.4f}, "
                   f"損失={self.best_val_metrics['loss']:.4f} (在第 {self.best_val_metrics.get('epoch', 'N/A')} 輪)")

        # 最後保存一次檢查點和訓練歷史
        self.save_checkpoint()
        self.save_training_history()

        # 返回最終結果
        return {
            'best_val_metrics': self.best_val_metrics,
            'training_history': self.training_history,
            'total_time_seconds': total_training_time,
            'epochs_completed': self.current_epoch + 1
        }


    def train_one_epoch(self, graph: dgl.DGLGraph):
        """
        訓練一個輪次 (使用 DGL DataLoader 和 NeighborSampler)

        參數:
            graph: 包含 'feat' 和 'label' 的 DGL 訓練圖

        返回:
            (平均訓練損失, 訓練指標字典)
        """
        self.model.train() # 設置為訓練模式
        total_loss = 0.0
        total_samples = 0 # 計算處理的總樣本數（目標節點數）
        all_predictions = []
        all_labels = []

        # --- 創建 DGL NeighborSampler 和 DataLoader ---
        sampler = NeighborSampler(
            self.num_neighbors,  # 每層採樣的鄰居數
            # 可以添加其他採樣器參數，如內存限制、概率等
            # prefetch_node_feats=['feat'], # 預取節點特徵
            # prefetch_labels=['label']     # 預取標籤
        )
        # 使用圖中所有節點作為訓練的種子節點
        train_nids = torch.arange(graph.num_nodes(), dtype=graph.idtype)

        dataloader = DGLDataLoader(
            graph,                          # 要採樣的圖
            train_nids,                     # 種子節點 ID
            sampler,                        # 鄰居採樣器
            device=self.device,             # 指定 DataLoader 輸出的設備
            batch_size=self.batch_size,     # 每個批次的種子節點數量
            shuffle=True,                   # 在每個 epoch 開始時打亂種子節點順序
            drop_last=False,                # 不丟棄最後一個不完整的批次
            num_workers=self.num_workers,   # 使用多少子進程加載數據
            pin_memory=self.pin_memory,     # 是否使用鎖頁記憶體
            use_prefetch_thread = (self.num_workers > 0), # DGL 推薦在多線程時開啟
            # use_uva = self.pin_memory and self.device.type == 'cuda' # 嘗試 UVA 加速
        )
        logger.debug(f"DGL DataLoader (Train) 已創建，共 {len(dataloader)} 個批次")

        # accumulation_counter = 0
        self.optimizer.zero_grad() # 在 epoch 開始時清零梯度

        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch+1} Train", leave=False, unit="batch")

        for step, (input_nodes, output_nodes, blocks) in enumerate(progress_bar):
            # input_nodes: 計算所需的原始圖節點 ID (包含鄰居和目標節點)
            # output_nodes: 當前批次要計算損失和預測的目標節點的原始圖 ID
            # blocks: Message Flow Graphs (MFGs) 列表，從外層到內層
            # blocks 和 input_nodes 已經在指定的 device 上 (由 DataLoader 完成)

            # 獲取輸入特徵 (在第一個 block 的源節點上) 和輸出標籤 (在最後一個 block 的目標節點上)
            batch_features = blocks[0].srcdata['feat'] # 確保模型使用 'feat'
            batch_labels = blocks[-1].dstdata['label'] # 確保標籤在圖中

            # --- 前向傳播 ---
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    # 模型接收 blocks 列表和輸入特徵
                    batch_logits = self.model(blocks, batch_features)
                    # 計算損失 (確保標籤是 LongTensor)
                    loss = self.criterion(batch_logits, batch_labels.long())
                    if self.use_gradient_accumulation:
                        loss = loss / self.gradient_accumulation_steps
                # 反向傳播 (使用 GradScaler)
                self.scaler.scale(loss).backward()
            else:
                batch_logits = self.model(blocks, batch_features)
                loss = self.criterion(batch_logits, batch_labels.long())
                if self.use_gradient_accumulation:
                    loss = loss / self.gradient_accumulation_steps
                # 反向傳播
                loss.backward()

            # --- 梯度累積與更新 ---
            # accumulation_counter += 1
            # if accumulation_counter % self.gradient_accumulation_steps == 0:
            # 每個 accumulation_steps 執行一次更新，或者在最後一個 step 執行
            if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                if self.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer) # 在裁剪前 unscale
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer) # 更新參數
                    self.scaler.update() # 更新 scaler 狀態
                    self.optimizer.zero_grad(set_to_none=True) # 使用 set_to_none 可能更高效
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                # accumulation_counter = 0 # 重置計數器 (如果使用計數器方式)

            # --- 收集結果用於計算指標 ---
            batch_size_actual = batch_labels.shape[0] # 當前批次的目標節點數量
            # 記錄未縮放的損失
            total_loss += loss.item() * batch_size_actual * (1.0 if not self.use_gradient_accumulation else self.gradient_accumulation_steps)
            total_samples += batch_size_actual
            _, predicted = torch.max(batch_logits, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

            # 更新進度條顯示未縮放的損失
            loss_display = loss.item() * (1.0 if not self.use_gradient_accumulation else self.gradient_accumulation_steps)
            progress_bar.set_postfix(loss=f"{loss_display:.4f}")
            self.global_step += 1

            # 釋放不再需要的變數
            del input_nodes, output_nodes, blocks, batch_features, batch_labels, batch_logits, loss, predicted
            # clean_memory() # 可選：在每個 step 後清理

        # 計算整個 epoch 的平均損失和指標
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        # 訓練集通常不計算 AUC，因此不傳遞概率
        metrics = self._compute_metrics(all_predictions, all_labels)
        return avg_loss, metrics


    def validate(self, graph: dgl.DGLGraph):
        """
        驗證模型性能 (使用 DGL DataLoader，計算概率)

        參數:
            graph: 包含 'feat' 和 'label' 的 DGL 驗證圖

        返回:
            (平均驗證損失, 驗證指標字典)
        """
        self.model.eval() # 設置為評估模式
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_labels = []
        all_probabilities = [] # 收集概率用於計算 AUC

        # --- 使用 DGL NeighborSampler 和 DataLoader ---
        sampler = NeighborSampler(self.num_neighbors) # 使用與訓練時相同的採樣鄰居數
        val_nids = torch.arange(graph.num_nodes(), dtype=graph.idtype)
        dataloader = DGLDataLoader(
            graph, val_nids, sampler,
            device=self.device,
            batch_size=self.batch_size * 2, # 驗證時通常可以使用更大的批次
            shuffle=False, # 驗證時不需要打亂
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            use_prefetch_thread=(self.num_workers > 0)
            # use_uva=self.pin_memory and self.device.type == 'cuda'
        )
        logger.debug(f"DGL DataLoader (Validate) 已創建，共 {len(dataloader)} 個批次")

        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch+1} Validate", leave=False, unit="batch")

        with torch.no_grad(): # 驗證時禁用梯度計算
            for step, (input_nodes, output_nodes, blocks) in enumerate(progress_bar):
                blocks = [b.to(self.device) for b in blocks]
                batch_features = blocks[0].srcdata['feat']
                batch_labels = blocks[-1].dstdata['label']

                # --- 前向傳播 ---
                # 注意：即使在 no_grad 下，如果模型內部有 Dropout 等層，eval() 模式會禁用它們
                # 混合精度在 torch.no_grad() 下通常不是必需的，但模型內部可能仍轉換類型
                batch_logits = self.model(blocks, batch_features)
                loss = self.criterion(batch_logits, batch_labels.long())

                # --- 收集結果 ---
                batch_size_actual = batch_labels.shape[0]
                total_loss += loss.item() * batch_size_actual
                total_samples += batch_size_actual

                probabilities = F.softmax(batch_logits, dim=1) # 計算 Softmax 概率
                _, predicted = torch.max(probabilities, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                # 收集概率 (確保是 float32，避免混合精度影響)
                all_probabilities.extend(probabilities.cpu().float().numpy())

                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

                del input_nodes, output_nodes, blocks, batch_features, batch_labels, batch_logits, loss, probabilities, predicted
                # clean_memory()

        # 計算平均損失和指標
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        # 將概率傳遞給指標計算函數以計算 AUC
        metrics = self._compute_metrics(all_predictions, all_labels, all_probabilities)
        return avg_loss, metrics


    def _compute_metrics(self, predictions: List[int], labels: List[int],
                         probabilities: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        """計算評估指標 (修正版：接收概率計算 AUC)"""
        if not predictions or not labels or len(predictions) != len(labels):
            logger.warning(f"預測列表 ({len(predictions)}) 或標籤列表 ({len(labels)}) 為空或長度不匹配，無法計算指標。")
            # 返回包含所有鍵的預設字典
            return {'accuracy': 0.0, 'precision_macro': 0.0, 'recall_macro': 0.0, 'f1_macro': 0.0,
                    'precision_weighted': 0.0, 'recall_weighted': 0.0, 'f1_weighted': 0.0, 'roc_auc': 0.0,
                    'confusion_matrix': []}

        metrics = {}
        try:
            predictions = np.array(predictions)
            labels = np.array(labels)
            unique_labels_in_data = np.unique(labels)
            num_classes = len(unique_labels_in_data)
            is_binary = num_classes <= 2

            # --- 計算基礎分類指標 ---
            metrics['accuracy'] = accuracy_score(labels, predictions)
            # 計算宏平均和加權平均 (處理零除問題)
            metrics['precision_macro'] = precision_score(labels, predictions, average='macro', zero_division=0)
            metrics['recall_macro'] = recall_score(labels, predictions, average='macro', zero_division=0)
            metrics['f1_macro'] = f1_score(labels, predictions, average='macro', zero_division=0)
            metrics['precision_weighted'] = precision_score(labels, predictions, average='weighted', zero_division=0)
            metrics['recall_weighted'] = recall_score(labels, predictions, average='weighted', zero_division=0)
            metrics['f1_weighted'] = f1_score(labels, predictions, average='weighted', zero_division=0)

            # --- 計算混淆矩陣 ---
            # 確保標籤是從 0 開始的連續整數，以獲得完整的混淆矩陣
            label_map = {label: i for i, label in enumerate(sorted(unique_labels_in_data))}
            mapped_labels = np.array([label_map.get(l, -1) for l in labels])
            mapped_predictions = np.array([label_map.get(p, -1) for p in predictions])
            # 計算混淆矩陣，使用從 0 到 num_classes-1 的標籤
            cm = confusion_matrix(mapped_labels, mapped_predictions, labels=list(range(num_classes)))
            metrics['confusion_matrix'] = cm.tolist() # 轉為列表方便 JSON 序列化

            # --- 計算 AUC (如果提供了概率) ---
            roc_auc = 0.0 # 預設值
            if probabilities is not None:
                 probabilities = np.array(probabilities)
                 if probabilities.shape[0] == len(labels): # 確保概率數量與標籤數量一致
                     if is_binary:
                         # 二分類: 使用標籤為 1 的概率
                         if set(unique_labels_in_data) == {0, 1}: # 嚴格檢查是否為 0/1 標籤
                             try:
                                  # 假設概率數組的第二列是正類別的概率
                                  if probabilities.shape[1] >= 2:
                                       probs_positive = probabilities[:, 1]
                                       roc_auc = roc_auc_score(labels, probs_positive)
                                  elif probabilities.shape[1] == 1: # 如果只有一列概率
                                       roc_auc = roc_auc_score(labels, probabilities[:, 0])
                                  else:
                                       logger.warning("二分類 AUC 計算失敗：概率數組列數異常。")
                             except ValueError as e:
                                  logger.warning(f"計算二分類 AUC 失敗（可能只有一個類別存在）: {e}")
                         else:
                              logger.warning(f"二分類標籤不是 {{0, 1}} (而是 {unique_labels_in_data})，無法直接計算 AUC。")
                     else: # 多分類
                         try:
                              # 確保概率矩陣的列數與實際類別數匹配
                              if probabilities.shape[1] == num_classes:
                                   # 使用原始標籤和對應的概率進行 OvR AUC 計算
                                   roc_auc = roc_auc_score(labels, probabilities, multi_class='ovr', average='weighted', labels=sorted(unique_labels_in_data))
                              else:
                                   logger.warning(f"多分類 AUC 計算失敗：概率列數 ({probabilities.shape[1]}) 與數據中類別數 ({num_classes}) 不匹配。")
                         except ValueError as e:
                              logger.warning(f"計算多分類 AUC 失敗（可能某些類別未出現或概率格式錯誤）: {e}")
                 else:
                      logger.warning("概率數組長度與標籤數量不匹配，無法計算 AUC。")

            metrics['roc_auc'] = roc_auc

            # --- 可選：分類報告 ---
            try:
                # 使用原始標籤生成報告
                target_names = [f"Class_{l}" for l in sorted(unique_labels_in_data)]
                report = classification_report(labels, predictions, labels=sorted(unique_labels_in_data), target_names=target_names, zero_division=0, output_dict=True)
                metrics['classification_report_dict'] = report # 保存為字典方便後續處理
            except Exception as e:
                logger.warning(f"生成分類報告時出錯: {e}")
                metrics['classification_report_dict'] = {}

        except Exception as e:
             logger.error(f"計算指標時發生嚴重錯誤: {e}", exc_info=True)
             # 返回包含所有鍵的預設字典
             metrics = {'accuracy': 0.0, 'precision_macro': 0.0, 'recall_macro': 0.0, 'f1_macro': 0.0,
                        'precision_weighted': 0.0, 'recall_weighted': 0.0, 'f1_weighted': 0.0, 'roc_auc': 0.0,
                        'confusion_matrix': [], 'classification_report_dict': {}}

        return metrics


    def _check_improvement(self, metrics: Dict[str, Any], loss: float) -> bool:
        """檢查模型是否有改進 (基於 f1_weighted, 其次是 loss)"""
        current_f1 = metrics.get('f1_weighted', -1.0) # 使用 -1 作為初始比較值，確保第一次總能更新

        # 優先比較主要指標 F1 (越大越好)
        # 使用一個小的容忍度 (epsilon) 來處理浮點數比較問題
        epsilon = 1e-6
        if current_f1 > self.best_val_metrics.get('f1_weighted', -1.0) + epsilon:
            self.best_val_metrics = {
                'f1_weighted': current_f1,
                'accuracy': metrics.get('accuracy', 0.0),
                'loss': loss,
                'epoch': self.current_epoch + 1 # 記錄達到最佳的 epoch 編號
            }
            return True
        # 如果 F1 基本相同，則比較損失（越小越好）
        elif abs(current_f1 - self.best_val_metrics.get('f1_weighted', -1.0)) <= epsilon and loss < self.best_val_metrics.get('loss', float('inf')) - epsilon:
            logger.info(f"驗證 F1 基本相同 (~{current_f1:.4f})，但損失更低 ({loss:.4f} < {self.best_val_metrics.get('loss'):.4f})，更新最佳損失。")
            self.best_val_metrics['loss'] = loss
            self.best_val_metrics['epoch'] = self.current_epoch + 1 # 同樣更新 epoch
            # 雖然主要指標未突破，但損失降低也算是一種改進，重置早停
            return True
        return False


    def _adjust_batch_size_if_needed(self) -> bool:
        """根據 GPU 顯存使用情況建議調整批次大小 (不實際調整 DGL DataLoader)"""
        if self.device.type != 'cuda' or not self.config.get('train', {}).get('use_dynamic_batch_size', False):
            return False

        try:
            gpu_mem_info = get_memory_usage(self.device.index if isinstance(self.device.index, int) else 0) # 獲取 GPU 資訊
            gpu_used_gb = gpu_mem_info.get('gpu_used_gb', 0)
            gpu_total_gb = gpu_mem_info.get('gpu_total_gb', 0)

            if gpu_total_gb > 0:
                gpu_memory_util = gpu_used_gb / gpu_total_gb
                memory_threshold = self.config.get('train', {}).get('dynamic_batch_memory_threshold', 0.85)

                if gpu_memory_util > memory_threshold:
                    new_batch_size_suggestion = max(self.batch_size // 2, 32) # 建議減半
                    if new_batch_size_suggestion < self.batch_size:
                        logger.warning(f"GPU 顯存使用率 ({gpu_memory_util:.1%}) 超過閾值 ({memory_threshold:.1%})，"
                                       f"建議減小 DGL DataLoader 的批次大小至 {new_batch_size_suggestion}。")
                        logger.warning("注意：當前實現無法在訓練中動態調整 DGL DataLoader 的批次大小。請在下次運行時修改配置。")
                        # 不實際修改 self.batch_size，因為 DataLoader 已創建
                        return True # 表示檢測到需要調整
            else:
                 logger.warning("無法獲取有效的 GPU 總顯存信息，跳過動態批次大小檢查。")

        except Exception as e:
             logger.error(f"檢查 GPU 顯存時出錯: {e}", exc_info=True)

        return False


    def get_current_lr(self) -> float:
        """獲取當前優化器的學習率"""
        if not self.optimizer.param_groups:
             return 0.0
        # 返回第一個參數組的學習率
        return self.optimizer.param_groups[0].get('lr', 0.0)


    def save_best_model(self):
        """保存當前記錄的最佳模型狀態"""
        model_path = os.path.join(self.model_dir, 'best_model.pt')
        try:
            # 保存模型狀態、相關指標和 epoch
            torch.save({
                'epoch': self.best_val_metrics.get('epoch', self.current_epoch + 1),
                'model_state_dict': self.model.state_dict(),
                # 保存優化器和調度器狀態對於從最佳點恢復訓練很有用
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_metrics': self.best_val_metrics,
                'config': self.config # 保存當時的配置
            }, model_path)
            logger.info(f"*** 最佳模型已更新並保存至 {model_path} (來自 Epoch: {self.best_val_metrics.get('epoch', 'N/A')}) ***")
        except Exception as e:
             logger.error(f"保存最佳模型時出錯: {e}", exc_info=True)


    def save_checkpoint(self):
        """保存當前訓練狀態的檢查點"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 保存完成的輪次數
        checkpoint_epoch = self.current_epoch + 1
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{checkpoint_epoch}_{timestamp}.pt")
        try:
            torch.save({
                'epoch': checkpoint_epoch, # 保存完成的輪次數
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_metrics': self.best_val_metrics, # 保存當前記錄的最佳指標
                'training_history': self.training_history, # 保存訓練歷史以便恢復圖表
                'config': self.config # 保存配置
            }, checkpoint_path)
            logger.info(f"檢查點已保存至 {checkpoint_path}")
        except Exception as e:
             logger.error(f"保存檢查點時出錯: {e}", exc_info=True)


    def save_training_history(self):
        """將訓練歷史保存為 JSON 文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = os.path.join(self.result_dir, f"training_history_{timestamp}.json")
        try:
            # 準備可序列化的歷史記錄，處理 NumPy 數組和數值類型
            serializable_history = []
            for epoch_data in self.training_history:
                 serializable_epoch = {}
                 for key, value in epoch_data.items():
                      if key == 'train_metrics' or key == 'val_metrics': # 處理指標字典
                           serializable_epoch[key] = {k: (v.tolist() if isinstance(v, np.ndarray)
                                                         else float(v) if isinstance(v, (np.floating, np.integer))
                                                         else v)
                                                     for k, v in value.items()}
                      elif isinstance(value, np.ndarray):
                           serializable_epoch[key] = value.tolist()
                      elif isinstance(value, (np.floating, np.integer)):
                            serializable_epoch[key] = float(value) # 轉換 numpy 數值類型
                      elif isinstance(value, torch.Tensor):
                           serializable_epoch[key] = value.item() # 轉換 Tensor 標量
                      else:
                           serializable_epoch[key] = value
                 serializable_history.append(serializable_epoch)

            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_history, f, indent=2, ensure_ascii=False)
            logger.info(f"訓練歷史已保存至 {history_path}")
        except Exception as e:
             logger.error(f"保存訓練歷史時出錯: {e}", exc_info=True)


    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        從檢查點加載模型和訓練狀態

        參數:
            checkpoint_path: 檢查點文件路徑

        返回:
            bool: 是否成功加載
        """
        if not os.path.exists(checkpoint_path):
            logger.error(f"檢查點文件 '{checkpoint_path}' 不存在")
            return False
        try:
            logger.info(f"正在從檢查點加載: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # 加載模型狀態字典
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # 加載優化器狀態字典
            if 'optimizer_state_dict' in checkpoint:
                 self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                 logger.warning("檢查點中未找到優化器狀態，將使用初始狀態。")

            # 加載學習率調度器狀態字典
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                 logger.warning("檢查點中未找到調度器狀態，將使用初始狀態。")

            # 恢復訓練狀態
            # epoch 保存的是完成的輪次數，所以訓練從下一輪開始
            self.current_epoch = checkpoint.get('epoch', 0)
            self.global_step = checkpoint.get('global_step', 0)
            self.best_val_metrics = checkpoint.get('best_val_metrics', {'f1_weighted': -1.0, 'loss': float('inf'), 'epoch': 0})
            self.training_history = checkpoint.get('training_history', [])

            logger.info(f"成功從檢查點恢復: 將從第 {self.current_epoch + 1} 輪開始訓練")
            logger.info(f"已恢復的最佳驗證指標: {self.best_val_metrics}")
            # 可選：將模型移到設備（如果 map_location 未生效）
            self.model.to(self.device)
            return True
        except FileNotFoundError:
             logger.error(f"檢查點文件未找到: {checkpoint_path}")
             return False
        except KeyError as e:
             logger.error(f"加載檢查點失敗：缺少鍵 '{e}'。檢查點文件可能已損壞或格式不兼容。")
             return False
        except Exception as e:
            logger.error(f"加載檢查點時發生未知錯誤: {str(e)}", exc_info=True)
            # 重置狀態以避免潛在問題
            self._reset_training_state()
            return False

    def _reset_training_state(self):
         """重置訓練狀態變數"""
         logger.warning("重置訓練狀態為初始值。")
         self.current_epoch = 0
         self.global_step = 0
         self.best_val_metrics = {'f1_weighted': -1.0, 'loss': float('inf'), 'epoch': 0}
         self.training_history = []
         # 可能需要重新初始化 optimizer 和 scheduler 狀態，取決於錯誤嚴重程度
         # self.optimizer = torch.optim.Adam(...)
         # self.scheduler = ReduceLROnPlateau(...)
