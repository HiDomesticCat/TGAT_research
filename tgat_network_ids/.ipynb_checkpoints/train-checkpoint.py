#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型訓練與評估模組

此模組負責：
1. 模型訓練
2. 模型評估
3. 模型儲存與載入
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

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TGATTrainer:
    """TGAT 模型訓練器"""
    
    def __init__(self, model, device='cpu', lr=0.001, weight_decay=1e-5, save_dir='./models'):
        """
        初始化模型訓練器
        
        參數:
            model: TGAT 模型
            device (str): 計算裝置 ('cpu' 或 'cuda')
            lr (float): 學習率
            weight_decay (float): 權重衰減係數
            save_dir (str): 模型儲存路徑
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        # 分類損失函數
        self.criterion = nn.CrossEntropyLoss()
        
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        
        logger.info(f"初始化 TGAT 訓練器: lr={lr}, weight_decay={weight_decay}, 裝置={device}")
    
    def train_epoch(self, graph, labels):
        """
        訓練一個 epoch
        
        參數:
            graph (dgl.DGLGraph): 訓練圖
            labels (torch.Tensor): 節點標籤
            
        返回:
            float: 訓練損失
            float: 訓練精度
        """
        self.model.train()
        
        # 將圖和標籤移至指定裝置
        graph = graph.to(self.device)
        labels = labels.to(self.device)
        
        # 清空梯度
        self.optimizer.zero_grad()
        
        # 前向傳播
        logits = self.model(graph)
        
        # 計算損失
        loss = self.criterion(logits, labels)
        
        # 反向傳播
        loss.backward()
        
        # 更新參數
        self.optimizer.step()
        
        # 計算精度
        _, pred = torch.max(logits, dim=1)
        correct = (pred == labels).sum().item()
        acc = correct / len(labels)
        
        return loss.item(), acc
    
    def evaluate(self, graph, labels):
        """
        評估模型
        
        參數:
            graph (dgl.DGLGraph): 驗證圖
            labels (torch.Tensor): 節點標籤
            
        返回:
            float: 驗證損失
            float: 驗證精度
            dict: 各類別的性能指標
        """
        self.model.eval()
        
        # 將圖和標籤移至指定裝置
        graph = graph.to(self.device)
        labels = labels.to(self.device)
        
        with torch.no_grad():
            # 前向傳播
            logits = self.model(graph)
            
            # 計算損失
            loss = self.criterion(logits, labels)
            
            # 計算精度
            _, pred = torch.max(logits, dim=1)
            correct = (pred == labels).sum().item()
            acc = correct / len(labels)
            
            # 將預測和標籤移回 CPU 進行詳細評估
            pred_cpu = pred.cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            
            # 計算各類別性能指標
            metrics = {
                'accuracy': accuracy_score(labels_cpu, pred_cpu),
                'precision_macro': precision_score(labels_cpu, pred_cpu, average='macro', zero_division=0),
                'recall_macro': recall_score(labels_cpu, pred_cpu, average='macro', zero_division=0),
                'f1_macro': f1_score(labels_cpu, pred_cpu, average='macro', zero_division=0),
                'precision_weighted': precision_score(labels_cpu, pred_cpu, average='weighted', zero_division=0),
                'recall_weighted': recall_score(labels_cpu, pred_cpu, average='weighted', zero_division=0),
                'f1_weighted': f1_score(labels_cpu, pred_cpu, average='weighted', zero_division=0),
                'confusion_matrix': confusion_matrix(labels_cpu, pred_cpu)
            }
            
            try:
                # 各類別精度
                metrics['precision_per_class'] = precision_score(labels_cpu, pred_cpu, average=None, zero_division=0)
                # 各類別召回率
                metrics['recall_per_class'] = recall_score(labels_cpu, pred_cpu, average=None, zero_division=0)
                # 各類別 F1 分數
                metrics['f1_per_class'] = f1_score(labels_cpu, pred_cpu, average=None, zero_division=0)
            except:
                logger.warning("無法計算各類別詳細指標")
        
        return loss.item(), acc, metrics
    
    def train(self, train_graph, train_labels, val_graph=None, val_labels=None, 
              epochs=100, patience=10, eval_every=1):
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
            
        返回:
            dict: 訓練歷史紀錄
        """
        logger.info(f"開始訓練: epochs={epochs}, patience={patience}")
        
        train_labels = train_labels.to(self.device)
        if val_graph is not None and val_labels is not None:
            val_labels = val_labels.to(self.device)
        
        best_val_acc = 0.0
        patience_counter = 0
        best_model_path = os.path.join(self.save_dir, 'best_model.pt')
        
        for epoch in range(epochs):
            # 訓練一個 epoch
            train_loss, train_acc = self.train_epoch(train_graph, train_labels)
            
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # 評估模型
            if val_graph is not None and val_labels is not None and (epoch + 1) % eval_every == 0:
                val_loss, val_acc, val_metrics = self.evaluate(val_graph, val_labels)
                
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                
                logger.info(
                    f"Epoch [{epoch+1}/{epochs}] - "
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
            else:
                logger.info(
                    f"Epoch [{epoch+1}/{epochs}] - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                )
        
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
            'best_val_acc': best_val_acc
        }
        
        return history
    
    def save_model(self, path):
        """
        儲存模型
        
        參數:
            path (str): 儲存路徑
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load_model(self, path):
        """
        載入模型
        
        參數:
            path (str): 模型路徑
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
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
            plt.savefig(save_path)
            logger.info(f"訓練歷史圖表已儲存至: {save_path}")
        
        plt.show()
    
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
        _, _, metrics = self.evaluate(graph, labels)
        
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
            plt.savefig(save_path)
            logger.info(f"混淆矩陣已儲存至: {save_path}")
        
        plt.show()
        
        return cm

# 測試訓練器
if __name__ == "__main__":
    import dgl
    from tgat_model import TGAT
    
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
    trainer = TGATTrainer(
        model=model,
        device='cpu',
        lr=0.01,
        weight_decay=1e-5,
        save_dir='./test_models'
    )
    
    # 測試單 epoch 訓練
    train_loss, train_acc = trainer.train_epoch(g, labels)
    print(f"訓練損失: {train_loss:.4f}, 訓練精度: {train_acc:.4f}")
    
    # 測試評估
    val_loss, val_acc, metrics = trainer.evaluate(g, labels)
    print(f"驗證損失: {val_loss:.4f}, 驗證精度: {val_acc:.4f}")
    print(f"F1 分數 (加權): {metrics['f1_weighted']:.4f}")
    
    # 測試完整訓練
    history = trainer.train(g, labels, g, labels, epochs=10, patience=5, eval_every=1)
    
    # 測試繪製訓練歷史
    trainer.plot_training_history()
    
    # 測試繪製混淆矩陣
    trainer.plot_confusion_matrix(g, labels, class_names=['正常', '攻擊'])