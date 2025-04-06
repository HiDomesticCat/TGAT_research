#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TGAT 模型實作模組

實作 Temporal Graph Attention Network (TGAT) 模型，
參考論文 "Temporal Graph Attention Network for Recommendation" 
和 "Inductive Representation Learning on Temporal Graphs"

此模組包含:
1. 時間編碼 (Temporal Encoding)
2. 圖注意力層 (Graph Attention Layers)
3. TGAT 模型架構
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

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeEncoding(nn.Module):
    def __init__(self, dimension):
        super(TimeEncoding, self).__init__()
        self.dimension = dimension
        
        # 初始化權重和偏置
        self.w = nn.Linear(1, dimension, bias=True)
        
        # 使用 Xavier 初始化權重
        nn.init.xavier_uniform_(self.w.weight)
        nn.init.zeros_(self.w.bias)
    
    def forward(self, t):
        # 確保輸入是浮點型且二維
        t = t.float()
        if t.dim() == 1:
            t = t.unsqueeze(1)
        
        # 確保權重也是浮點型
        self.w.weight = nn.Parameter(self.w.weight.float())
        self.w.bias = nn.Parameter(self.w.bias.float())
        
        # 使用餘弦函數進行編碼
        output = torch.cos(self.w(t))
        return output

class TemporalGATLayer(nn.Module):
    """時間圖注意力層"""
    
    def __init__(self, in_dim, out_dim, time_dim, num_heads=4, feat_drop=0.6, attn_drop=0.6, residual=True):
        """
        初始化時間圖注意力層
        
        參數:
            in_dim (int): 輸入特徵維度
            out_dim (int): 輸出特徵維度
            time_dim (int): 時間編碼維度
            num_heads (int): 注意力頭數量
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
    
    def forward(self, g, h, time_tensor):
        # 獲得時間編碼
        # 檢查時間特徵的長度，如果不匹配，則擴展或截斷
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
        
        time_emb = self.time_enc(time_tensor)  # [num_edges, time_dim]
        
        # 特徵丟棄
        h = self.feat_drop(h)
        
        # 確保時間編碼的形狀正確
        if time_emb.dim() == 1:
            time_emb = time_emb.unsqueeze(0)
        
        # 將時間特徵和邊特徵合併
        g.edata['time_feat'] = time_emb
        
        def message_func(edges):
            # 使用串聯而不是加法
            time_expanded = edges.data['time_feat'].expand_as(edges.src['h'][:, :self.time_dim])
            return {'m': torch.cat([edges.src['h'], time_expanded], dim=1)}
        
        def reduce_func(nodes):
            # 聚合消息，取平均
            return {'h_time': nodes.mailbox['m'].mean(1)}
        
        # 應用消息傳遞
        g.update_all(message_func, reduce_func)
        
        # 處理時間特徵
        h_time = g.ndata.get('h_time', torch.zeros(h.shape[0], h.shape[1] + self.time_dim, device=h.device))
        
        # 檢查並調整 h_time 的形狀
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

class TGAT(nn.Module):
    """時間圖注意力網絡模型"""
    
    def __init__(self, in_dim, hidden_dim, out_dim, time_dim, num_layers=2, num_heads=4, dropout=0.1, num_classes=2):
        """
        初始化 TGAT 模型
        
        參數:
            in_dim (int): 輸入特徵維度
            hidden_dim (int): 隱藏層維度
            out_dim (int): 輸出特徵維度
            time_dim (int): 時間編碼維度
            num_layers (int): TGAT 層數量
            num_heads (int): 注意力頭數量
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
        
        # 特徵投影層
        self.feat_project = nn.Linear(in_dim, hidden_dim)
        
        # TGAT 層
        self.layers = nn.ModuleList()
        
        # 第一層將輸入特徵投影到隱藏維度
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
        
        # 中間層
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
        
        # 最後一層
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
        
        # 分類層
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, num_classes)
        )
        
        # 初始化參數
        self.reset_parameters()
        
        logger.info(f"初始化 TGAT 模型: {num_layers} 層, {num_heads} 個注意力頭, {num_classes} 分類類別")
    
    def reset_parameters(self):
        """重置參數"""
        gain = nn.init.calculate_gain('relu')
        
        # 初始化特徵投影層
        nn.init.xavier_normal_(self.feat_project.weight, gain=gain)
        nn.init.zeros_(self.feat_project.bias)
        
        # 初始化分類器
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)
    
    def forward(self, g):
        """
        前向傳播
        
        參數:
            g (dgl.DGLGraph): 輸入圖
            
        返回:
            torch.Tensor: 節點分類輸出 [num_nodes, num_classes]
        """
        # 獲取節點特徵
        h = g.ndata['h']
        
        # 投影特徵到隱藏維度
        h = self.feat_project(h)
        
        # 獲取邊的時間特徵
        time_tensor = g.edata.get('time', None)
        
        # 如果沒有時間特徵，則使用全 0 時間戳記
        if time_tensor is None:
            time_tensor = torch.zeros(g.num_edges(), device=h.device)
        
        # 應用 TGAT 層
        for i, layer in enumerate(self.layers):
            h = layer(g, h, time_tensor)
        
        # 應用分類器
        logits = self.classifier(h)
        
        return logits
    
    def encode(self, g):
        """
        僅編碼節點，不進行分類
        
        參數:
            g (dgl.DGLGraph): 輸入圖
            
        返回:
            torch.Tensor: 節點表示 [num_nodes, out_dim]
        """
        # 獲取節點特徵
        h = g.ndata['h']
        
        # 投影特徵到隱藏維度
        h = self.feat_project(h)
        
        # 獲取邊的時間特徵
        time_tensor = g.edata.get('time', None)
        
        # 如果沒有時間特徵，則使用全 0 時間戳記
        if time_tensor is None:
            time_tensor = torch.zeros(g.num_edges(), device=h.device)
        
        # 應用 TGAT 層
        for i, layer in enumerate(self.layers):
            h = layer(g, h, time_tensor)
        
        return h

# 測試 TGAT 模型
if __name__ == "__main__":
    import numpy as np
    import dgl
    
    # 建立一個簡單的圖
    src = torch.tensor([0, 1, 2, 3, 4])
    dst = torch.tensor([1, 2, 3, 4, 0])
    g = dgl.graph((src, dst))
    
    # 添加節點特徵
    num_nodes = 5
    in_dim = 10
    h = torch.randn(num_nodes, in_dim)
    g.ndata['h'] = h
    
    # 添加邊時間特徵
    time = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    g.edata['time'] = time
    
    # 設置模型參數
    hidden_dim = 16
    out_dim = 16
    time_dim = 8
    num_layers = 2
    num_heads = 4
    dropout = 0.1
    num_classes = 3
    
    # 初始化模型
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
    
    # 前向傳播
    logits = model(g)
    
    print(f"模型輸出形狀: {logits.shape}")
    print(f"預期形狀: [5, 3]")
    
    # 測試節點編碼
    node_embeddings = model.encode(g)
    print(f"節點表示形狀: {node_embeddings.shape}")
    print(f"預期形狀: [5, 16]")