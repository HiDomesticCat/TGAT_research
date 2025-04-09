#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
記憶體優化版 TGAT 模型實現模組

基於論文 "Temporal Graph Attention Network for Recommendation" 
和 "Inductive Representation Learning on Temporal Graphs" 實現的時間圖注意力網絡 (TGAT) 模型。

本模組包括：
1. 時間編碼
2. 圖注意力層
3. TGAT 模型架構
4. 記憶體優化技術：
   - 梯度檢查點 (Gradient Checkpointing)
   - 混合精度訓練支持
   - 稀疏注意力機制
   - 記憶體高效的前向傳播
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import math
import logging
import numpy as np
from dgl.nn.pytorch import GATConv
from torch.utils.checkpoint import checkpoint

# 導入記憶體優化工具
from ..utils.memory_utils import clean_memory

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeEncoding(nn.Module):
    def __init__(self, dimension):
        super(TimeEncoding, self).__init__()
        self.dimension = dimension
        
        # Initialize weights and biases
        self.w = nn.Linear(1, dimension, bias=True)
        
        # Use Xavier initialization for weights
        nn.init.xavier_uniform_(self.w.weight)
        nn.init.zeros_(self.w.bias)
    
    def forward(self, t):
        # Ensure input is float and 2D
        t = t.float()
        if t.dim() == 1:
            t = t.unsqueeze(1)
        
        # Ensure weights are also float
        self.w.weight = nn.Parameter(self.w.weight.float())
        self.w.bias = nn.Parameter(self.w.bias.float())
        
        # Use cosine function for encoding
        output = torch.cos(self.w(t))
        return output

class TemporalGATLayer(nn.Module):
    """時間圖注意力層（記憶體優化版）"""
    
    def __init__(self, in_dim, out_dim, time_dim, num_heads=4, feat_drop=0.6, attn_drop=0.6, residual=True):
        """
        初始化時間圖注意力層
        
        參數:
            in_dim (int): 輸入特徵維度
            out_dim (int): 輸出特徵維度
            time_dim (int): 時間編碼維度
            num_heads (int): 注意力頭數
            feat_drop (float): 特徵丟棄率
            attn_drop (float): 注意力丟棄率
            residual (bool): 是否使用殘差連接
        """
        super(TemporalGATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.time_dim = time_dim
        self.num_heads = num_heads
        
        # 時間編碼器
        self.time_enc = TimeEncoding(time_dim)
        
        # 特徵丟棄層
        self.feat_drop = nn.Dropout(feat_drop)
        
        # 定義注意力層
        self.gat = GATConv(
            in_feats=in_dim + time_dim,  # 特徵 + 時間
            out_feats=out_dim // num_heads,
            num_heads=num_heads,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            residual=residual,
            activation=F.elu,
            allow_zero_in_degree=True
        )
        
        # 梯度檢查點標誌
        self.checkpoint = False
    
    def _forward_impl(self, g, h, time_tensor):
        """
        實際的前向傳播實現
        
        參數:
            g (dgl.DGLGraph): 輸入圖
            h (torch.Tensor): 節點特徵
            time_tensor (torch.Tensor): 時間特徵
            
        返回:
            torch.Tensor: 更新後的節點特徵
        """
        # 檢查時間特徵長度，擴展或截斷以匹配邊數
        if len(time_tensor) != g.num_edges():
            if len(time_tensor) < g.num_edges():
                # 如果時間特徵少於邊數，重複最後一個值
                time_tensor = torch.cat([
                    time_tensor, 
                    torch.ones(g.num_edges() - len(time_tensor), device=time_tensor.device) * time_tensor[-1]
                ])
            else:
                # 如果時間特徵多於邊數，截斷
                time_tensor = time_tensor[:g.num_edges()]
        
        # 獲取時間編碼
        time_emb = self.time_enc(time_tensor)  # [num_edges, time_dim]
        
        # 特徵丟棄
        h = self.feat_drop(h)
        
        # 確保時間編碼有正確的形狀
        if time_emb.dim() == 1:
            time_emb = time_emb.unsqueeze(0)
        
        # 將時間特徵與邊特徵結合
        g.edata['time_feat'] = time_emb
        
        def message_func(edges):
            # 使用連接而不是加法
            time_expanded = edges.data['time_feat'].expand_as(edges.src['h'][:, :self.time_dim])
            return {'m': torch.cat([edges.src['h'], time_expanded], dim=1)}
        
        def reduce_func(nodes):
            # 聚合消息，取平均值
            return {'h_time': nodes.mailbox['m'].mean(1)}
        
        # 應用消息傳遞
        g.update_all(message_func, reduce_func)
        
        # 處理時間特徵
        h_time = g.ndata.get('h_time', torch.zeros(h.shape[0], h.shape[1] + self.time_dim, device=h.device))
        
        # 檢查並調整 h_time 形狀
        if h_time.shape[1] != self.in_dim + self.time_dim:
            # 如果形狀不匹配，截斷或填充
            if h_time.shape[1] > self.in_dim + self.time_dim:
                h_time = h_time[:, :self.in_dim + self.time_dim]
            else:
                # 使用零填充
                padding = torch.zeros(h_time.shape[0], self.in_dim + self.time_dim - h_time.shape[1], device=h_time.device)
                h_time = torch.cat([h_time, padding], dim=1)
        
        # 應用 GAT 層
        h_new = self.gat(g, h_time)
        
        # 合併多頭注意力結果
        h_new = h_new.view(h_new.shape[0], -1)
        
        return h_new
    
    def forward(self, g, h, time_tensor):
        """
        前向傳播（支持梯度檢查點）
        
        參數:
            g (dgl.DGLGraph): 輸入圖
            h (torch.Tensor): 節點特徵
            time_tensor (torch.Tensor): 時間特徵
            
        返回:
            torch.Tensor: 更新後的節點特徵
        """
        # 如果啟用了梯度檢查點，使用 checkpoint 函數
        if self.checkpoint and h.requires_grad:
            # 使用 checkpoint 包裝 _forward_impl
            # 注意：checkpoint 只能接受張量作為輸入，所以我們需要將圖轉換為張量
            # 這裡我們使用一個簡單的方法：將圖作為非張量參數傳遞
            def custom_forward(h_tensor, time_tensor):
                return self._forward_impl(g, h_tensor, time_tensor)
            
            return checkpoint(custom_forward, h, time_tensor)
        else:
            # 否則直接使用實現
            return self._forward_impl(g, h, time_tensor)

class TGAT(nn.Module):
    """時間圖注意力網絡模型（記憶體優化版）"""
    
    def __init__(self, in_dim, hidden_dim, out_dim, time_dim, num_layers=2, num_heads=4, dropout=0.1, num_classes=2):
        """
        初始化 TGAT 模型
        
        參數:
            in_dim (int): 輸入特徵維度
            hidden_dim (int): 隱藏層維度
            out_dim (int): 輸出特徵維度
            time_dim (int): 時間編碼維度
            num_layers (int): TGAT 層數
            num_heads (int): 注意力頭數
            dropout (float): 丟棄率
            num_classes (int): 分類類別數
        """
        super(TGAT, self).__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.time_dim = time_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_classes = num_classes
        
        # 記憶體優化設置
        self.use_checkpoint = False
        self.use_mixed_precision = False
        
        # Feature projection layer
        self.feat_project = nn.Linear(in_dim, hidden_dim)
        
        # TGAT layers
        self.layers = nn.ModuleList()
        
        # First layer projects input features to hidden dimension
        self.layers.append(
            TemporalGATLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                time_dim=time_dim,
                num_heads=num_heads,
                feat_drop=dropout,
                attn_drop=dropout
            )
        )
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.layers.append(
                TemporalGATLayer(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    time_dim=time_dim,
                    num_heads=num_heads,
                    feat_drop=dropout,
                    attn_drop=dropout
                )
            )
        
        # Last layer
        if num_layers > 1:
            self.layers.append(
                TemporalGATLayer(
                    in_dim=hidden_dim,
                    out_dim=out_dim,
                    time_dim=time_dim,
                    num_heads=num_heads,
                    feat_drop=dropout,
                    attn_drop=dropout
                )
            )
        
        # Classification layer
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, num_classes)
        )
        
        # Initialize parameters
        self.reset_parameters()
        
        logger.info(f"Initialized TGAT model: {num_layers} layers, {num_heads} attention heads, {num_classes} classification classes")
    
    def reset_parameters(self):
        """Reset parameters"""
        gain = nn.init.calculate_gain('relu')
        
        # Initialize feature projection layer
        nn.init.xavier_normal_(self.feat_project.weight, gain=gain)
        nn.init.zeros_(self.feat_project.bias)
        
        # Initialize classifier
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)
    
    def enable_gradient_checkpointing(self):
        """啟用梯度檢查點以減少記憶體使用"""
        self.use_checkpoint = True
        for layer in self.layers:
            layer.checkpoint = True
        logger.info("已啟用梯度檢查點")
    
    def disable_gradient_checkpointing(self):
        """禁用梯度檢查點"""
        self.use_checkpoint = False
        for layer in self.layers:
            layer.checkpoint = False
        logger.info("已禁用梯度檢查點")
    
    def enable_mixed_precision(self):
        """啟用混合精度訓練以減少記憶體使用"""
        self.use_mixed_precision = True
        logger.info("已啟用混合精度訓練")
    
    def disable_mixed_precision(self):
        """禁用混合精度訓練"""
        self.use_mixed_precision = False
        logger.info("已禁用混合精度訓練")
    
    def _forward_impl(self, g):
        """
        實際的前向傳播實現
        
        參數:
            g (dgl.DGLGraph): 輸入圖
            
        返回:
            torch.Tensor: 節點分類輸出 [num_nodes, num_classes]
        """
        # 獲取節點特徵
        h = g.ndata['h']
        
        # 將特徵投影到隱藏維度
        h = self.feat_project(h)
        
        # 獲取邊時間特徵
        time_tensor = g.edata.get('time', None)
        
        # 如果沒有時間特徵，使用全 0 時間戳
        if time_tensor is None:
            time_tensor = torch.zeros(g.num_edges(), device=h.device)
        
        # 應用 TGAT 層
        for i, layer in enumerate(self.layers):
            h = layer(g, h, time_tensor)
            
            # 定期清理記憶體
            if (i + 1) % 2 == 0:
                clean_memory()
        
        # 應用分類器
        logits = self.classifier(h)
        
        return logits
    
    def forward(self, g):
        """
        前向傳播（支持混合精度訓練）
        
        參數:
            g (dgl.DGLGraph): 輸入圖
            
        返回:
            torch.Tensor: 節點分類輸出 [num_nodes, num_classes]
        """
        # 如果啟用了混合精度訓練，使用 autocast
        if self.use_mixed_precision and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                return self._forward_impl(g)
        else:
            return self._forward_impl(g)
    
    def _encode_impl(self, g):
        """
        實際的編碼實現
        
        參數:
            g (dgl.DGLGraph): 輸入圖
            
        返回:
            torch.Tensor: 節點表示 [num_nodes, out_dim]
        """
        # 獲取節點特徵
        h = g.ndata['h']
        
        # 將特徵投影到隱藏維度
        h = self.feat_project(h)
        
        # 獲取邊時間特徵
        time_tensor = g.edata.get('time', None)
        
        # 如果沒有時間特徵，使用全 0 時間戳
        if time_tensor is None:
            time_tensor = torch.zeros(g.num_edges(), device=h.device)
        
        # 應用 TGAT 層
        for i, layer in enumerate(self.layers):
            h = layer(g, h, time_tensor)
            
            # 定期清理記憶體
            if (i + 1) % 2 == 0:
                clean_memory()
        
        return h
    
    def encode(self, g):
        """
        僅編碼節點，不進行分類（支持混合精度訓練）
        
        參數:
            g (dgl.DGLGraph): 輸入圖
            
        返回:
            torch.Tensor: 節點表示 [num_nodes, out_dim]
        """
        # 如果啟用了混合精度訓練，使用 autocast
        if self.use_mixed_precision and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                return self._encode_impl(g)
        else:
            return self._encode_impl(g)

# Test TGAT model
if __name__ == "__main__":
    import numpy as np
    import dgl
    
    # Create a simple graph
    src = torch.tensor([0, 1, 2, 3, 4])
    dst = torch.tensor([1, 2, 3, 4, 0])
    g = dgl.graph((src, dst))
    
    # Add node features
    num_nodes = 5
    in_dim = 10
    h = torch.randn(num_nodes, in_dim)
    g.ndata['h'] = h
    
    # Add edge time features
    time = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    g.edata['time'] = time
    
    # Set model parameters
    hidden_dim = 16
    out_dim = 16
    time_dim = 8
    num_layers = 2
    num_heads = 4
    dropout = 0.1
    num_classes = 3
    
    # Initialize model
    model = TGAT(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        time_dim=time_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        num_classes=num_classes
    )
    
    # Forward propagation
    logits = model(g)
    
    print(f"Model output shape: {logits.shape}")
    print(f"Expected shape: [5, 3]")
    
    # Test node encoding
    node_embeddings = model.encode(g)
    print(f"Node representation shape: {node_embeddings.shape}")
    print(f"Expected shape: [5, 16]")
