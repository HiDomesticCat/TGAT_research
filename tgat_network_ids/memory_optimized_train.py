#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
記憶體優化版模型訓練與評估模組

此模組是 train.py 的記憶體優化版本，提供以下功能：
1. 混合精度訓練
2. 梯度累積
3. 梯度檢查點
4. 動態批次大小
5. 漸進式訓練
6. 記憶體使用監控
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime
import gc

# 導入記憶體優化工具
from memory_utils import (
    clean_memory, memory_usage_decorator, print_memory_usage,
    get_memory_usage, print_optimization_suggestions
)

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryOptimizedTGATTrainer:
    """記憶體優化版 TGAT 模型訓練器"""
    
    def __init__(self, model, config, device='cpu'):
        """
        初始化模型訓練器
        
        參數:
            model: TGAT 模型
            config (dict): 配置字典，包含以下鍵：
                - train.lr: 學習率
                - train.weight_decay: 權重衰減係數
                - train.batch_size: 訓練批次大小
                - train.use_dynamic_batch_size: 是否使用動態批次大小
                - train.memory_threshold: 記憶體使用率閾值
                - train.use_progressive_training: 是否使用漸進式訓練
                - train.progressive_training_initial_ratio: 漸進式訓練的初始比例
                - train.progressive_training_growth_rate: 漸進式訓練的增長率
                - model.use_mixed_precision: 是否使用混合精度訓練
                - model.use_gradient_accumulation: 是否使用梯度累積
                - model.gradient_accumulation_steps: 梯度累積步數
                - model.use_gradient_checkpointing: 是否使用梯度檢查點
                - output.model_dir: 模型儲存路徑
                - output.checkpoint_dir: 檢查點儲存路徑
            device (str): 計算裝置 ('cpu' 或 'cuda')
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        # 從配置中提取訓練相關設置
        train_config = config.get('train', {})
        model_config = config.get('model', {})
        output_config = config.get('output', {})
        
        # 訓練參數
        self.lr = train_config.get('lr', 0.001)
        self.weight_decay = train_config.get('weight_decay', 1e-5)
        self.batch_size = train_config.get('batch_size', 128)
        self.use_dynamic_batch_size = train_config.get('use_dynamic_batch_size', True)
        self.memory_threshold = train_config.get('memory_threshold', 0.8)
        self.use_progressive_training = train_config.get('use_progressive_training', True)
        self.progressive_training_initial_ratio = train_config.get('progressive_training_initial_ratio', 0.3)
        self.progressive_training_growth_rate = train_config.get('progressive_training_growth_rate', 0.1)
        
        # 模型優化參數
        self.use_mixed_precision = model_config.get('use_mixed_precision', True)
        self.use_gradient_accumulation = model_config.get('use_gradient_accumulation', True)
        self.gradient_accumulation_steps = model_config.get('gradient_accumulation_steps', 4)
        self.use_gradient_checkpointing = model_config.get('use_gradient_checkpointing', True)
        
        # 輸出路徑
        self.save_dir = output_config.get('model_dir', './models')
        self.checkpoint_dir = output_config.get('checkpoint_dir', './checkpoints')
        
        # 確保目錄存在
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 初始化優化器
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        # 初始化學習率調度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        
        # 分類損失函數
        self.criterion = nn.CrossEntropyLoss()
        
        # 混合精度訓練
        if self.use_mixed_precision and device != 'cpu':
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # 啟用梯度檢查點
        if self.use_gradient_checkpointing:
            self.model.apply(self._enable_gradient_checkpointing)
        
        # 訓練記錄
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        self.current_epoch = 0
        
        logger.info(f"初始化記憶體優化版 TGAT 訓練器: lr={self.lr}, weight_decay={self.weight_decay}, 裝置={device}")
        logger.info(f"混合精度訓練: {self.use_mixed_precision}, 梯度累積: {self.use_gradient_accumulation} (步數={self.gradient_accumulation_steps}), 梯度檢查點: {self.use_gradient_checkpointing}")
        logger.info(f"動態批次大小: {self.use_dynamic_batch_size}, 漸進式訓練: {self.use_progressive_training}")
    
    def _enable_gradient_checkpointing(self, module):
        """啟用模組的梯度檢查點"""
        if hasattr(module, 'checkpoint') and callable(module.checkpoint):
            module.checkpoint = True
            logger.info(f"啟用梯度檢查點: {module.__class__.__name__}")
    
    def _adjust_batch_size(self):
        """根據記憶體使用情況調整批次大小"""
        if not self.use_dynamic_batch_size:
            return self.batch_size
        
        # 獲取當前記憶體使用情況
        mem_info = get_memory_usage()
        system_memory_percent = mem_info['system_memory_percent']
        
        # 如果記憶體使用率超過閾值，減小批次大小
        if system_memory_percent > self.memory_threshold:
            new_batch_size = max(16, self.batch_size // 2)
            if new_batch_size != self.batch_size:
                logger.info(f"記憶體使用率 {system_memory_percent:.1f}% 超過閾值 {self.memory_threshold*100:.1f}%，批次大小從 {self.batch_size} 減小到 {new_batch_size}")
                self.batch_size = new_batch_size
        
        return self.batch_size
    
    def _get_progressive_training_size(self, total_size, epoch):
        """獲取漸進式訓練的資料大小"""
        if not self.use_progressive_training:
            return total_size
        
        # 計算當前比例
        ratio = min(1.0, self.progressive_training_initial_ratio + epoch * self.progressive_training_growth_rate)
        
        # 計算資料大小
        size = int(total_size * ratio)
        
        logger.info(f"漸進式訓練: 第 {epoch+1} 輪，使用 {ratio:.2f} 比例的資料 ({size}/{total_size})")
        
        return size
    
    @memory_usage_decorator
    def train_epoch(self, graph, labels, epoch):
        """
        訓練一個 epoch
        
        參數:
            graph (dgl.DGLGraph): 訓練圖
            labels (torch.Tensor): 節點標籤
            epoch (int): 當前 epoch
            
        返回:
            float: 訓練損失
            float: 訓練精度
        """
        self.model.train()
        
        # 將圖和標籤移至指定裝置
        graph = graph.to(self.device)
        labels = labels.to(self.device)
        
        # 獲取漸進式訓練的資料大小
        total_nodes = graph.num_nodes()
        train_size = self._get_progressive_training_size(total_nodes, epoch)
        
        # 如果使用漸進式訓練，隨機選擇節點
        if train_size < total_nodes:
            indices = torch.randperm(total_nodes)[:train_size]
            subgraph = dgl.node_subgraph(graph, indices)
            sub_labels = labels[indices]
        else:
            subgraph = graph
            sub_labels = labels
        
        # 調整批次大小
        batch_size = self._adjust_batch_size()
        
        # 計算批次數量
        num_nodes = subgraph.num_nodes()
        num_batches = (num_nodes + batch_size - 1) // batch_size
        
        # 初始化損失和精度
        total_loss = 0.0
        total_correct = 0
        
        # 清空梯度
        self.optimizer.zero_grad()
        
        # 使用 tqdm 顯示進度條
        progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}")
        
        for batch_idx in progress_bar:
            # 獲取批次索引
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_nodes)
            batch_indices = torch.arange(start_idx, end_idx)
            
            # 獲取批次子圖
            batch_subgraph = dgl.node_subgraph(subgraph, batch_indices)
            batch_labels = sub_labels[batch_indices]
            
            # 前向傳播 (使用混合精度)
            if self.use_mixed_precision and self.device != 'cpu':
                with torch.cuda.amp.autocast():
                    logits = self.model(batch_subgraph)
                    loss = self.criterion(logits, batch_labels)
                    
                    # 如果使用梯度累積，縮放損失
                    if self.use_gradient_accumulation:
                        loss = loss / self.gradient_accumulation_steps
                
                # 反向傳播
                self.scaler.scale(loss).backward()
                
                # 如果使用梯度累積，每 N 步更新一次參數
                if self.use_gradient_accumulation:
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or batch_idx == num_batches - 1:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                else:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # 不使用混合精度
                logits = self.model(batch_subgraph)
                loss = self.criterion(logits, batch_labels)
                
                # 如果使用梯度累積，縮放損失
                if self.use_gradient_accumulation:
                    loss = loss / self.gradient_accumulation_steps
                
                # 反向傳播
                loss.backward()
                
                # 如果使用梯度累積，每 N 步更新一次參數
                if self.use_gradient_accumulation:
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or batch_idx == num_batches - 1:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                else:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # 計算精度
            _, pred = torch.max(logits, dim=1)
            correct = (pred == batch_labels).sum().item()
            
            # 更新統計信息
            total_loss += loss.item() * (self.gradient_accumulation_steps if self.use_gradient_accumulation else 1)
            total_correct += correct
            
            # 更新進度條
            progress_bar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': total_correct / (end_idx)
            })
            
            # 定期清理記憶體
            if batch_idx % 10 == 0:
                clean_memory()
        
        # 計算平均損失和精度
        avg_loss = total_loss / num_batches
        avg_acc = total_correct / num_nodes
        
        return avg_loss, avg_acc
    
    @memory_usage_decorator
    def evaluate(self, graph, labels, class_names=None):
        """
        評估模型
        
        參數:
            graph (dgl.DGLGraph): 評估圖
            labels (torch.Tensor): 節點標籤
            class_names (list, optional): 類別名稱
            
        返回:
            float: 評估損失
            float: 評估精度
            dict: 評估指標
        """
        # 確保正確導入 classification_report
        from sklearn.metrics import classification_report
        
        # 將圖和標籤移至指定裝置
        graph = graph.to(self.device)
        labels = labels.to(self.device)
        
        # 設置為評估模式
        self.model.eval()
        
        # 調整批次大小
        batch_size = self._adjust_batch_size()
        
        # 計算批次數量
        num_nodes = graph.num_nodes()
        num_batches = (num_nodes + batch_size - 1) // batch_size
        
        # 初始化
        all_logits = []
        
        # 不計算梯度
        with torch.no_grad():
            # 使用 tqdm 顯示進度條
            progress_bar = tqdm(range(num_batches), desc="Evaluating")
            
            for batch_idx in progress_bar:
                # 獲取批次索引
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_nodes)
                batch_indices = torch.arange(start_idx, end_idx)
                
                # 獲取批次子圖
                batch_subgraph = dgl.node_subgraph(graph, batch_indices)
                
                # 前向傳播 (使用混合精度)
                if self.use_mixed_precision and self.device != 'cpu':
                    with torch.cuda.amp.autocast():
                        logits = self.model(batch_subgraph)
                else:
                    logits = self.model(batch_subgraph)
                
                all_logits.append(logits)
                
                # 定期清理記憶體
                if batch_idx % 10 == 0:
                    clean_memory()
            
            # 合併所有批次的輸出
            all_logits = torch.cat(all_logits, dim=0)
            
            # 計算損失
            loss = self.criterion(all_logits, labels)
            
            # 預測
            _, pred = torch.max(all_logits, dim=1)
            
            # 將 tensor 移動到 CPU
            pred_cpu = pred.cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            
            # 計算指標
            try:
                metrics = {
                    'accuracy': accuracy_score(labels_cpu, pred_cpu),
                    'precision_macro': precision_score(labels_cpu, pred_cpu, average='macro', zero_division=0),
                    'recall_macro': recall_score(labels_cpu, pred_cpu, average='macro', zero_division=0),
                    'f1_macro': f1_score(labels_cpu, pred_cpu, average='macro', zero_division=0),
                    'precision_weighted': precision_score(labels_cpu, pred_cpu, average='weighted', zero_division=0),
                    'recall_weighted': recall_score(labels_cpu, pred_cpu, average='weighted', zero_division=0),
                    'f1_weighted': f1_score(labels_cpu, pred_cpu, average='weighted', zero_division=0),
                    'confusion_matrix': confusion_matrix(labels_cpu, pred_cpu),
                    'report': classification_report(
                        labels_cpu, 
                        pred_cpu, 
                        target_names=class_names,
                        zero_division=0
                    )
                }
            except Exception as e:
                logger.warning(f"計算指標時發生錯誤: {e}")
                metrics = {}
            
            # 計算精度
            correct = (pred == labels).sum().item()
            acc = correct / len(labels)
        
        return loss.item(), acc, metrics
    
    @memory_usage_decorator
    def train(self, train_graph, train_labels, val_graph=None, val_labels=None, 
              epochs=100, patience=10, eval_every=1, class_names=None):
        """
        訓練模型
        
        參數:
            train_graph (dgl.DGLGraph): 訓練圖
            train_labels (torch.Tensor): 訓練標籤
            val_graph (dgl.DGLGraph, optional): 驗證圖
            val_labels (torch.Tensor, optional): 驗證標籤
            epochs (int): 訓練 epochs 數量
            patience (int): 早停耐心值
            eval_every (int): 每隔多少 epochs 評估一次
            class_names (list, optional): 類別名稱
            
        返回:
            dict: 訓練歷史紀錄
        """
        logger.info(f"開始訓練: epochs={epochs}, patience={patience}")
        
        # 將標籤移至指定裝置
        train_labels = train_labels.to(self.device)
        if val_graph is not None and val_labels is not None:
            val_labels = val_labels.to(self.device)
        
        best_val_acc = self.best_val_acc
        patience_counter = 0
        best_model_path = os.path.join(self.save_dir, 'best_model.pt')
        
        # 從上次訓練的地方繼續
        start_epoch = self.current_epoch
        
        for epoch in range(start_epoch, start_epoch + epochs):
            # 訓練一個 epoch
            train_loss, train_acc = self.train_epoch(train_graph, train_labels, epoch)
            
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # 評估模型
            if val_graph is not None and val_labels is not None and (epoch + 1) % eval_every == 0:
                val_loss, val_acc, val_metrics = self.evaluate(val_graph, val_labels, class_names)
                
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                
                # 更新學習率調度器
                self.scheduler.step(val_metrics['f1_weighted'])
                
                logger.info(
                    f"Epoch [{epoch+1}/{start_epoch+epochs}] - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                    f"F1 (Weighted): {val_metrics['f1_weighted']:.4f}"
                )
                
                # 檢查是否為最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.best_val_acc = best_val_acc
                    patience_counter = 0
                    
                    # 儲存最佳模型
                    self.save_model(best_model_path)
                    logger.info(f"保存最佳模型，驗證精度: {best_val_acc:.4f}")
                else:
                    patience_counter += 1
                    
                # 早停
                if patience_counter >= patience:
                    logger.info(f"早停: {patience} 個 epochs 內驗證精度未改善")
                    break
                
                # 保存檢查點
                self.save_checkpoint(epoch + 1)
            else:
                logger.info(
                    f"Epoch [{epoch+1}/{start_epoch+epochs}] - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                )
            
            # 更新當前 epoch
            self.current_epoch = epoch + 1
            
            # 清理記憶體
            clean_memory()
        
        # 訓練結束後載入最佳模型
        if val_graph is not None and val_labels is not None:
            if os.path.exists(best_model_path):
                self.load_model(best_model_path)
                logger.info(f"載入最佳模型，驗證精度: {best_val_acc:.4f}")
        
        # 返回訓練歷史紀錄
        history = {
            'train_loss': self.train_losses,
            'train_acc': self.train_accuracies,
            'val_loss': self.val_losses,
            'val_acc': self.val_accuracies,
            'best_val_acc': best_val_acc,
            'current_epoch': self.current_epoch
        }
        
        return history
    
    def save_model(self, path):
        """
        儲存模型
        
        參數:
            path (str): 儲存路徑
        """
        # 確保目錄存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': self.best_val_acc,
            'current_epoch': self.current_epoch
        }, path)
        
        logger.info(f"模型已儲存至: {path}")
    
    def load_model(self, path):
        """
        載入模型
        
        參數:
            path (str): 模型路徑
        """
        if not os.path.exists(path):
            logger.warning(f"模型文件不存在: {path}")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        
        if 'train_accuracies' in checkpoint:
            self.train_accuracies = checkpoint['train_accuracies']
        
        if 'val_accuracies' in checkpoint:
            self.val_accuracies = checkpoint['val_accuracies']
        
        if 'best_val_acc' in checkpoint:
            self.best_val_acc = checkpoint['best_val_acc']
        
        if 'current_epoch' in checkpoint:
            self.current_epoch = checkpoint['current_epoch']
        
        logger.info(f"模型已載入: {path}")
    
    def save_checkpoint(self, epoch):
        """
        儲存檢查點
        
        參數:
            epoch (int): 當前 epoch
        """
        # 確保目錄存在
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 生成檢查點文件名
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        
        # 儲存檢查點
        self.save_model(checkpoint_path)
        
        logger.info(f"檢查點已儲存至: {checkpoint_path}")
    
    def load_checkpoint(self, epoch=None):
        """
        載入檢查點
        
        參數:
            epoch (int, optional): 要載入的 epoch，如果為 None 則載入最新的檢查點
        """
        # 獲取所有檢查點文件
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pt")]
        
        if not checkpoint_files:
            logger.warning(f"未找到檢查點文件: {self.checkpoint_dir}")
            return
        
        if epoch is not None:
            # 載入指定 epoch 的檢查點
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            if not os.path.exists(checkpoint_path):
                logger.warning(f"未找到指定 epoch 的檢查點: {checkpoint_path}")
                return
        else:
            # 載入最新的檢查點
            checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_files[-1])
        
        # 載入檢查點
        self.load_model(checkpoint_path)
        
        logger.info(f"已載入檢查點: {checkpoint_path}")
    
    def plot_training_history(self, save_path=None):
        """
        繪製訓練歷史紀錄
        
        參數:
            save_path (str, optional): 圖表儲存路徑
        """
        plt.figure(figsize=(12, 5))
        
        # 繪製損失曲線
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='訓練損失')
        if self.val_losses:
            plt.plot(self.val_losses, label='驗證損失')
        plt.xlabel('Epoch')
        plt.ylabel('損失')
        plt.title('訓練與驗證損失')
        plt.legend()
        
        # 繪製精度曲線
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='訓練精度')
        if self.val_accuracies:
            plt.plot(self.val_accuracies, label='驗證精度')
        plt.xlabel('Epoch')
        plt.ylabel('精度')
        plt.title('訓練與驗證精度')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            # 確保目錄存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            plt.savefig(save_path)
            logger.info(f"訓練歷史圖表已儲存至: {save_path}")
        
        plt.close()
    
    def plot_confusion_matrix(self, graph, labels, class_names=None, save_path=None):
        """
        繪製混淆矩陣
        
        參數:
            graph (dgl.DGLGraph): 輸入圖
            labels (torch.Tensor): 標籤
            class_names (list, optional): 類別名稱
            save_path (str, optional): 圖表儲存路徑
        """
        # 評估模型
        _, _, metrics = self.evaluate(graph, labels, class_names)
        
        # 獲取混淆矩陣
        cm = metrics['confusion_matrix']
        
        # 繪製混淆矩陣
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('預測類別')
        plt.ylabel('真實類別')
        plt.title('混淆矩陣')
        
        if save_path:
            # 確保目錄存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            plt.savefig(save_path)
            logger.info(f"混淆矩陣已儲存至: {save_path}")
        
        plt.close()
        
        return cm
    
    def get_memory_usage(self):
        """
        獲取模型的記憶體使用情況
        
        返回:
            dict: 記憶體使用信息
        """
        # 計算模型參數的記憶體使用
        model_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        
        # 計算優化器狀態的記憶體使用
        optimizer_size = sum(state['exp_avg'].numel() * state['exp_avg'].element_size() + 
                            state['exp_avg_sq'].numel() * state['exp_avg_sq'].element_size()
                            for group in self.optimizer.param_groups
                            for p in group['params']
                            for state in [self.optimizer.state[p]])
        
        # 計算梯度的記憶體使用
        grad_size = sum(p.grad.numel() * p.grad.element_size() for p in self.model.parameters() if p.grad is not None)
        
        # 計算訓練歷史的記憶體使用
        history_size = (len(self.train_losses) + len(self.val_losses) + 
                       len(self.train_accuracies) + len(self.val_accuracies)) * 8  # 假設每個值是 float64 (8 bytes)
        
        # 轉換為 MB
        total_size = (model_size + optimizer_size + grad_size + history_size) / (1024 * 1024)
        
        return {
            'model_mb': model_size / (1024 * 1024),
            'optimizer_mb': optimizer_size / (1024 * 1024),
            'grad_mb': grad_size / (1024 * 1024),
            'history_mb': history_size / (1024 * 1024),
            'total_mb': total_size
        }
    
    def print_memory_usage(self):
        """打印模型的記憶體使用情況"""
        mem_info = self.get_memory_usage()
        
        logger.info("模型記憶體使用情況:")
        logger.info(f"  模型參數: {mem_info['model_mb']:.2f} MB")
        logger.info(f"  優化器狀態: {mem_info['optimizer_mb']:.2f} MB")
        logger.info(f"  梯度: {mem_info['grad_mb']:.2f} MB")
        logger.info(f"  訓練歷史: {mem_info['history_mb']:.2f} MB")
        logger.info(f"  總計: {mem_info['total_mb']:.2f} MB")

# 測試訓練器
if __name__ == "__main__":
    import yaml
    import dgl
    from tgat_model import TGAT
    
    # 載入配置
    with open("memory_optimized_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # 建立一個簡單的圖用於測試
    src = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    dst = torch.tensor([1, 2, 3, 4, 0, 2, 3, 4, 0, 1])
    g = dgl.graph((src, dst))
    
    # 添加節點特徵
    num_nodes = 5
    in_dim = 10
    h = torch.randn(num_nodes, in_dim)
    g.ndata['h'] = h
    
    # 添加邊時間特徵
    time = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    g.edata['time'] = time
    
    # 建立測試標籤
    labels = torch.tensor([0, 1, 0, 1, 0])
    
    # 初始化 TGAT 模型
    model = TGAT(
        in_dim=in_dim,
        hidden_dim=16,
        out_dim=16,
        time_dim=8,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        num_classes=2
    )
    
    # 初始化訓練器
    trainer = MemoryOptimizedTGATTrainer(
        model=model,
        config=config,
        device='cpu'
    )
    
    # 測試單 epoch 訓練
    train_loss, train_acc = trainer.train_epoch(g, labels, 0)
    print(f"訓練損失: {train_loss:.4f}, 訓練精度: {train_acc:.4f}")
    
    # 測試評估
    val_loss, val_acc, metrics = trainer.evaluate(g, labels, class_names=['正常', '攻擊'])
    print(f"驗證損失: {val_loss:.4f}, 驗證精度: {val_acc:.4f}")
    print(f"F1 分數 (加權): {metrics['f1_weighted']:.4f}")
    
    # 測試完整訓練
    history = trainer.train(g, labels, g, labels, epochs=3, patience=5, eval_every=1, class_names=['正常', '攻擊'])
    
    # 測試繪製訓練歷史
    trainer.plot_training_history()
    
    # 測試繪製混淆矩陣
    trainer.plot_confusion_matrix(g, labels, class_names=['正常', '攻擊'])
    
    # 測試記憶體使用情況
    trainer.print_memory_usage()
