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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

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
            model: TGAT模型實例
            optimizer: 優化器
            criterion: 損失函數
            device: 訓練設備 (CPU/GPU)
            config: 配置字典
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        
        # 獲取訓練相關配置
        train_config = config.get('train', {})
        self.epochs = train_config.get('epochs', 50)
        self.patience = train_config.get('patience', 5)
        self.batch_size = train_config.get('batch_size', 1024)
        self.use_dynamic_batch_size = train_config.get('use_dynamic_batch_size', False)
        self.memory_threshold = train_config.get('memory_threshold', 0.8)
        self.use_progressive_training = train_config.get('use_progressive_training', False)
        self.progressive_training_initial_ratio = train_config.get('progressive_training_initial_ratio', 0.5)
        self.progressive_training_growth_rate = train_config.get('progressive_training_growth_rate', 0.2)
        
        # 獲取模型相關配置
        model_config = config.get('model', {})
        self.use_mixed_precision = model_config.get('use_mixed_precision', False)
        self.use_gradient_accumulation = model_config.get('use_gradient_accumulation', False)
        self.gradient_accumulation_steps = model_config.get('gradient_accumulation_steps', 4)
        
        # 獲取輸出相關配置
        output_config = config.get('output', {})
        self.model_dir = output_config.get('model_dir', './models')
        self.result_dir = output_config.get('result_dir', './results')
        self.checkpoint_dir = output_config.get('checkpoint_dir', './checkpoints')
        
        # 創建目錄
        for directory in [self.model_dir, self.result_dir, self.checkpoint_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # 初始化混合精度訓練
        if self.use_mixed_precision and device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("啟用混合精度訓練")
        else:
            self.use_mixed_precision = False
            logger.info("混合精度訓練未啟用或不支援")
        
        # 訓練狀態跟蹤
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metrics = {'accuracy': 0.0, 'loss': float('inf')}
        self.early_stop_counter = 0
        self.training_history = []
        
        # 設置學習率調度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5,
            patience=self.patience // 2,
            verbose=True
        )
        
        logger.info(f"訓練引擎初始化完成: 設備={device}, 混合精度={self.use_mixed_precision}, "
                   f"梯度累積={self.use_gradient_accumulation}")
    
    def train(self, data_loader, graph_builder, validation_loader=None):
        """
        執行完整訓練流程
        
        參數:
            data_loader: 數據加載器
            graph_builder: 圖構建器
            validation_loader: 驗證數據加載器 (可選)
            
        返回:
            訓練結果字典
        """
        logger.info(f"開始訓練: 總輪次={self.epochs}, 批次大小={self.batch_size}")
        start_time = time.time()
        
        # 初始化漸進式訓練的數據比例
        if self.use_progressive_training:
            data_ratio = self.progressive_training_initial_ratio
            logger.info(f"使用漸進式訓練，初始數據比例: {data_ratio*100:.1f}%")
        else:
            data_ratio = 1.0
        
        # 主訓練循環
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # 更新數據比例（漸進式訓練）
            if self.use_progressive_training and epoch > 0:
                data_ratio = min(1.0, data_ratio + self.progressive_training_growth_rate)
                logger.info(f"Epoch {epoch+1}/{self.epochs}: 數據比例 {data_ratio*100:.1f}%")
            
            # 訓練一個輪次
            train_loss, train_metrics = self.train_one_epoch(data_loader, graph_builder, data_ratio)
            
            # 驗證
            if validation_loader:
                val_loss, val_metrics = self.validate(validation_loader, graph_builder)
            else:
                # 如果沒有提供驗證集，使用訓練集的一小部分作為驗證
                val_loss, val_metrics = self.validate(data_loader, graph_builder, subset_ratio=0.2)
            
            # 更新學習率調度器
            self.scheduler.step(val_loss)
            
            # 記錄當前輪次的訓練歷史
            epoch_duration = time.time() - epoch_start_time
            epoch_history = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_metrics': train_metrics,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'duration': epoch_duration,
                'learning_rate': self.get_current_lr()
            }
            self.training_history.append(epoch_history)
            
            # 輸出日誌
            logger.info(f"Epoch {epoch+1}/{self.epochs}: "
                       f"訓練損失={train_loss:.4f}, 訓練準確率={train_metrics['accuracy']:.4f}, "
                       f"驗證損失={val_loss:.4f}, 驗證準確率={val_metrics['accuracy']:.4f}, "
                       f"耗時={epoch_duration:.2f}s")
            
            # 檢查是否為最佳模型
            improved = self._check_improvement(val_metrics, val_loss)
            if improved:
                logger.info(f"發現更好的模型! 驗證準確率: {val_metrics['accuracy']:.4f}")
                self.save_best_model()
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                logger.info(f"模型未改進: {self.early_stop_counter}/{self.patience}")
            
            # 保存檢查點
            if (epoch + 1) % 5 == 0 or epoch == self.epochs - 1:
                self.save_checkpoint()
            
            # 檢查早停
            if self.early_stop_counter >= self.patience:
                logger.info(f"達到早停條件，在 {epoch+1} 輪次後停止訓練")
                break
        
        # 訓練完成，計算總時間
        total_time = time.time() - start_time
        logger.info(f"訓練完成，總耗時: {total_time:.2f}s")
        logger.info(f"最佳驗證指標: 準確率={self.best_val_metrics['accuracy']:.4f}, "
                   f"損失={self.best_val_metrics['loss']:.4f}")
        
        # 保存訓練歷史
        self.save_training_history()
        
        return {
            'best_val_metrics': self.best_val_metrics,
            'training_history': self.training_history,
            'total_time': total_time,
            'epochs_completed': self.current_epoch + 1
        }
    
    def train_one_epoch(self, data_loader, graph_builder, data_ratio=1.0):
        """
        訓練一個輪次
        
        參數:
            data_loader: 數據加載器
            graph_builder: 圖構建器
            data_ratio: 使用的數據比例 [0.0, 1.0]
            
        返回:
            (平均損失, 指標字典)
        """
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        all_predictions = []
        all_labels = []
        
        # 獲取訓練批次
        train_batches = data_loader.get_train_batches(ratio=data_ratio)
        total_batches = len(train_batches)
        
        # 梯度累積計數器
        accumulation_counter = 0
        
        # 訓練循環
        for batch_idx, batch_data in enumerate(train_batches):
            try:
                # 動態調整批次大小
                if self.use_dynamic_batch_size and batch_idx % 10 == 0 and torch.cuda.is_available():
                    self._adjust_batch_size_if_needed(data_loader)
                
                # 準備數據
                batch_graph = graph_builder.build_batch_graph(batch_data)
                batch_features = batch_data['features'].to(self.device)
                batch_labels = batch_data['labels'].to(self.device)
                
                # 梯度累積的第一個步驟清除梯度
                if accumulation_counter == 0 or not self.use_gradient_accumulation:
                    self.optimizer.zero_grad()
                
                # 前向傳播（混合精度或普通）
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_graph, batch_features)
                        loss = self.criterion(outputs, batch_labels)
                        
                        # 縮放損失以進行梯度累積
                        if self.use_gradient_accumulation:
                            loss = loss / self.gradient_accumulation_steps
                    
                    # 反向傳播（混合精度）
                    self.scaler.scale(loss).backward()
                    
                    # 是否更新參數
                    if (accumulation_counter == self.gradient_accumulation_steps - 1) or not self.use_gradient_accumulation:
                        # 梯度裁剪
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        # 優化器步驟
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        accumulation_counter = 0
                    else:
                        accumulation_counter += 1
                else:
                    # 標準前向傳播
                    outputs = self.model(batch_graph, batch_features)
                    loss = self.criterion(outputs, batch_labels)
                    
                    # 縮放損失以進行梯度累積
                    if self.use_gradient_accumulation:
                        loss = loss / self.gradient_accumulation_steps
                    
                    # 反向傳播
                    loss.backward()
                    
                    # 是否更新參數
                    if (accumulation_counter == self.gradient_accumulation_steps - 1) or not self.use_gradient_accumulation:
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        # 優化器步驟
                        self.optimizer.step()
                        accumulation_counter = 0
                    else:
                        accumulation_counter += 1
                
                # 收集預測和標籤用於指標計算
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                
                # 更新統計信息
                total_loss += loss.item() * (1.0 if not self.use_gradient_accumulation else self.gradient_accumulation_steps)
                batch_count += 1
                self.global_step += 1
                
                # 定期輸出日誌
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                    logger.debug(f"Batch {batch_idx+1}/{total_batches}: Loss={loss.item():.4f}")
                
            except Exception as e:
                logger.error(f"處理批次 {batch_idx} 時出錯: {str(e)}")
                continue
        
        # 計算平均損失和指標
        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        metrics = self._compute_metrics(all_predictions, all_labels)
        
        return avg_loss, metrics
    
    def validate(self, data_loader, graph_builder, subset_ratio=1.0):
        """
        驗證模型性能
        
        參數:
            data_loader: 數據加載器
            graph_builder: 圖構建器
            subset_ratio: 使用的數據比例 [0.0, 1.0]
            
        返回:
            (平均損失, 指標字典)
        """
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        all_predictions = []
        all_labels = []
        
        # 獲取驗證批次
        val_batches = data_loader.get_val_batches(ratio=subset_ratio)
        
        # 不計算梯度
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_batches):
                try:
                    # 準備數據
                    batch_graph = graph_builder.build_batch_graph(batch_data)
                    batch_features = batch_data['features'].to(self.device)
                    batch_labels = batch_data['labels'].to(self.device)
                    
                    # 前向傳播
                    outputs = self.model(batch_graph, batch_features)
                    loss = self.criterion(outputs, batch_labels)
                    
                    # 收集預測和標籤
                    _, predicted = torch.max(outputs, 1)
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(batch_labels.cpu().numpy())
                    
                    # 更新統計信息
                    total_loss += loss.item()
                    batch_count += 1
                    
                except Exception as e:
                    logger.error(f"驗證批次 {batch_idx} 時出錯: {str(e)}")
                    continue
        
        # 計算平均損失和指標
        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        metrics = self._compute_metrics(all_predictions, all_labels)
        
        return avg_loss, metrics
    
    def _compute_metrics(self, predictions, labels):
        """計算評估指標"""
        if not predictions or not labels:
            return {'accuracy': 0.0}
        
        # 轉換為numpy數組
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # 計算基本指標
        accuracy = accuracy_score(labels, predictions)
        
        # 如果是二分類問題，計算更多指標
        if len(np.unique(labels)) == 2:
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='binary')
            
            # 嘗試計算AUC (需要預測機率)
            try:
                auc = roc_auc_score(labels, predictions)
            except:
                auc = 0.0
                
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc
            }
        
        # 多分類問題
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro')
            
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _check_improvement(self, metrics, loss):
        """
        檢查模型是否有改進
        
        返回:
            改進標誌
        """
        # 主要使用準確率作為指標
        if metrics['accuracy'] > self.best_val_metrics['accuracy']:
            self.best_val_metrics = {
                'accuracy': metrics['accuracy'],
                'loss': loss,
                'epoch': self.current_epoch + 1,
                **{k: v for k, v in metrics.items() if k != 'accuracy'}
            }
            return True
        
        # 準確率相同時，使用損失作為輔助指標
        if metrics['accuracy'] == self.best_val_metrics['accuracy'] and loss < self.best_val_metrics['loss']:
            self.best_val_metrics['loss'] = loss
            return True
            
        return False
    
    def _adjust_batch_size_if_needed(self, data_loader):
        """根據顯存使用情況動態調整批次大小"""
        if not torch.cuda.is_available() or not self.use_dynamic_batch_size:
            return False
        
        # 獲取顯存使用情況
        memory_allocated = torch.cuda.memory_allocated(self.device)
        memory_reserved = torch.cuda.memory_reserved(self.device)
        
        if memory_reserved > 0:
            memory_util = memory_allocated / memory_reserved
        else:
            return False
        
        # 如果顯存使用率超過閾值，減小批次大小
        if memory_util > self.memory_threshold:
            new_batch_size = max(32, self.batch_size // 2)
            
            if new_batch_size != self.batch_size:
                self.batch_size = new_batch_size
                data_loader.update_batch_size(new_batch_size)
                logger.info(f"動態調整批次大小至 {new_batch_size} (顯存使用率: {memory_util:.2f})")
                return True
        
        return False
    
    def get_current_lr(self):
        """獲取當前學習率"""
        return self.optimizer.param_groups[0]['lr']
    
    def save_best_model(self):
        """保存最佳模型"""
        model_path = os.path.join(self.model_dir, 'best_model.pt')
        
        # 保存模型
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.best_val_metrics,
            'config': self.config
        }, model_path)
        
        logger.info(f"最佳模型已保存至 {model_path}")
    
    def save_checkpoint(self):
        """保存檢查點"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{self.current_epoch+1}_{timestamp}.pt")
        
        # 保存檢查點
        torch.save({
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_metrics': self.best_val_metrics,
            'config': self.config
        }, checkpoint_path)
        
        logger.info(f"檢查點已保存至 {checkpoint_path}")
    
    def save_training_history(self):
        """保存訓練歷史"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = os.path.join(self.result_dir, f"history_{timestamp}.json")
        
        # 保存歷史記錄
        with open(history_path, 'w', encoding='utf-8') as f:
            # 確保 numpy 數組被序列化為列表
            history = [{k: v.tolist() if isinstance(v, np.ndarray) else v 
                        for k, v in epoch.items()} 
                       for epoch in self.training_history]
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        logger.info(f"訓練歷史已保存至 {history_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        加載檢查點
        
        參數:
            checkpoint_path: 檢查點文件路徑
        """
        if not os.path.exists(checkpoint_path):
            logger.error(f"檢查點文件不存在: {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 加載模型參數
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # 加載優化器參數
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 加載調度器參數
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # 恢復訓練狀態
            self.current_epoch = checkpoint['epoch']
            self.global_step = checkpoint.get('global_step', 0)
            self.best_val_metrics = checkpoint['best_val_metrics']
            
            logger.info(f"從檢查點恢復: 輪次={self.current_epoch+1}, 最佳驗證準確率={self.best_val_metrics['accuracy']:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"加載檢查點時出錯: {str(e)}")
            return False
