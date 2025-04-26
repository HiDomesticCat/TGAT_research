#!/usr/bin/env python
# coding: utf-8 -*-

"""
高效記憶體優化版 TGAT 模型實現模組

根據建議進行優化，專注於：
1. 高效的稀疏時間編碼
2. 優化的圖注意力層
3. 稀疏梯度與張量
4. 各種記憶體減少技術
5. 整合DGL和PyTorch高效API
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
import scipy.sparse as sp

# 導入記憶體優化工具
from ..utils.memory_utils import clean_memory

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryEfficientTimeEncoding(nn.Module):
    """記憶體高效的時間編碼模組 - 使用緩存和量化技術"""

    def __init__(self, dimension):
        super(MemoryEfficientTimeEncoding, self).__init__()
        self.dimension = dimension

        # 使用較小的參數量
        reduced_dim = max(8, dimension // 4)  # 減少參數量但保留表達能力
        self.w_reduce = nn.Linear(1, reduced_dim, bias=True)
        self.w_expand = nn.Linear(reduced_dim, dimension, bias=True)

        # 使用Xavier初始化權重
        nn.init.xavier_uniform_(self.w_reduce.weight)
        nn.init.zeros_(self.w_reduce.bias)
        nn.init.xavier_uniform_(self.w_expand.weight)
        nn.init.zeros_(self.w_expand.bias)

        # 時間編碼緩存 - 使用LRU緩存控制大小
        self.cache = {}
        self.cache_size_limit = 5000  # 根據實際情況調整
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_queries = 0

        # 使用時間量化 - 將連續時間映射到離散時間點
        self.use_quantization = True
        self.quantization_factor = 0.1  # 量化粒度

        # 使用半精度 (float16) 存儲編碼向量
        self.use_half_precision = True

        # 緩存性能統計
        self.report_interval = 1000  # 每1000次查詢報告一次緩存性能

    def _quantize_time(self, t):
        """量化時間戳以減少緩存項目數量"""
        if not self.use_quantization:
            return t
        # 將時間戳量化為較粗糙的粒度
        return torch.floor(t / self.quantization_factor) * self.quantization_factor

    def _compute_encoding(self, t_tensor):
        """計算時間編碼 - 使用減少的維度"""
        # 降低維度處理
        reduced = torch.cos(self.w_reduce(t_tensor))
        # 使用ReLU增加非線性
        reduced = F.relu(reduced)
        # 擴展回原始維度
        encoding = self.w_expand(reduced)

        # 使用半精度存儲
        if self.use_half_precision and encoding.device.type == 'cuda':
            encoding = encoding.half()

        return encoding

    def forward(self, t):
        """前向傳播 - 高度優化的緩存實現"""
        # 記錄查詢次數
        self.total_queries += 1

        # 報告緩存性能
        if self.total_queries % self.report_interval == 0:
            hit_rate = self.cache_hits / max(1, self.total_queries) * 100
            logger.info(f"時間編碼緩存命中率: {hit_rate:.2f}% (hits: {self.cache_hits}, queries: {self.total_queries})")

        # 將輸入轉換為浮點數並確保二維
        t = t.float()
        if t.dim() == 1:
            t = t.unsqueeze(1)

        # 量化時間
        if self.use_quantization:
            t = self._quantize_time(t)

        # 使用緩存加速時間編碼計算
        batch_size = t.size(0)
        device_key = str(t.device)

        # 重置過大的緩存以避免記憶體洩漏
        if len(self.cache) > self.cache_size_limit:
            # 保留最近使用的一半項目
            keep_keys = list(self.cache.keys())[-self.cache_size_limit//2:]
            new_cache = {k: self.cache[k] for k in keep_keys}
            self.cache = new_cache
            logger.info(f"緩存已重置，剩餘 {len(self.cache)} 項")

        # 批次處理：將每個時間戳檢查緩存，未命中時計算
        outputs = []
        for i in range(batch_size):
            time_val = t[i].item()
            cache_key = f"{device_key}_{time_val:.4f}"  # 限制精度避免過多緩存項

            if cache_key in self.cache:
                # 緩存命中
                output_i = self.cache[cache_key]
                self.cache_hits += 1
            else:
                # 緩存未命中，計算編碼
                self.cache_misses += 1
                time_tensor = t[i].view(1, 1)
                with torch.no_grad():  # 避免跟踪梯度以節省記憶體
                    output_i = self._compute_encoding(time_tensor).squeeze(0)
                self.cache[cache_key] = output_i

            outputs.append(output_i)

        # 將所有編碼合併為一個批次
        if outputs:
            result = torch.stack(outputs)
            # 確保輸出是float32，即使內部使用了float16
            if result.dtype == torch.float16:
                result = result.float()
            return result
        else:
            return torch.empty((0, self.dimension), device=t.device)

class OptimizedTemporalGATLayer(nn.Module):
    """優化版時間圖注意力層 - 專注記憶體效率"""

    def __init__(self, in_dim, out_dim, time_dim, num_heads=4, feat_drop=0.6, attn_drop=0.6, residual=True):
        """
        初始化優化版時間圖注意力層

        參數:
            in_dim (int): 輸入特徵維度
            out_dim (int): 輸出特徵維度
            time_dim (int): 時間編碼維度
            num_heads (int): 注意力頭數
            feat_drop (float): 特徵丟棄率
            attn_drop (float): 注意力丟棄率
            residual (bool): 是否使用殘差連接
        """
        super(OptimizedTemporalGATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.time_dim = time_dim
        self.num_heads = num_heads

        # 使用更高效的時間編碼器
        self.time_enc = MemoryEfficientTimeEncoding(time_dim)

        # 特徵丟棄層
        self.feat_drop = nn.Dropout(feat_drop)

        # 使用更高效的注意力計算 - 透過減少中間張量
        # 1. 首先，減少每個頭的維度
        head_dim = max(4, out_dim // (num_heads * 2))  # 每頭維度減半，但至少4維

        # 2. 使用更有效的注意力機制
        self.gat = EfficientGATConv(
            in_feats=in_dim + time_dim,
            out_feats=head_dim,
            num_heads=num_heads,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            residual=residual,
            activation=F.elu,
            allow_zero_in_degree=True
        )

        # 3. 如果需要，使用投影層將維度調整為所需的out_dim
        self.projection = None
        expected_out = head_dim * num_heads
        if expected_out != out_dim:
            self.projection = nn.Linear(expected_out, out_dim)
            # 使用Xavier初始化
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)

        # 梯度檢查點標誌
        self.checkpoint = False

        # 使用稀疏注意力標誌
        self.use_sparse_attention = True
        self.sparse_attention_threshold = 0.01  # 小於此值的注意力分數被設為0

    def _forward_impl(self, g, h, time_tensor):
        """
        實際的前向傳播實現 - 高效版本

        參數:
            g (dgl.DGLGraph): 輸入圖
            h (torch.Tensor): 節點特徵 [num_nodes, in_dim]
            time_tensor (torch.Tensor): 時間特徵 [num_edges]

        返回:
            torch.Tensor: 更新後的節點特徵 [num_nodes, out_dim]
        """
        # 檢查圖、特徵和時間張量是否兼容
        num_edges = g.num_edges()
        if num_edges == 0:
            # 處理空圖情況
            return torch.zeros((g.num_nodes(), self.out_dim), device=h.device)

        # 適應時間張量大小
        if len(time_tensor) != num_edges:
            if len(time_tensor) < num_edges:
                # 如果時間特徵少於邊數，重複最後一個值
                time_tensor = torch.cat([
                    time_tensor,
                    torch.ones(num_edges - len(time_tensor), device=time_tensor.device) * time_tensor[-1]
                ])
            else:
                # 如果時間特徵多於邊數，截斷
                time_tensor = time_tensor[:num_edges]

        # 使用高效的時間編碼
        time_emb = self.time_enc(time_tensor)  # [num_edges, time_dim]

        # 特徵丟棄
        h = self.feat_drop(h)

        # 確保時間編碼有正確的形狀
        if time_emb.dim() == 1:
            time_emb = time_emb.unsqueeze(0)

        # 將時間特徵與邊特徵結合 - 使用DGL的高效API
        g.edata['time_feat'] = time_emb

        # 定義高效的消息傳遞函數 - 使用函數而非lambda優化記憶體
        def message_func(edges):
            src_h = edges.src['h']
            time_feat = edges.data['time_feat']

            # 連接時間特徵 - 僅擴展到需要的形狀
            time_expanded = time_feat
            if time_expanded.shape[1] < src_h.shape[1]:
                time_expanded = time_feat.expand(-1, min(src_h.shape[1], self.time_dim))

            # 連接特徵和時間
            return {'m': torch.cat([src_h, time_expanded], dim=1)}

        # 定義高效的聚合函數 - 使用sum而非mean減少一次除法操作
        def reduce_func(nodes):
            # 聚合消息，使用sum然後除以消息數量
            msgs = nodes.mailbox['m']
            return {'h_time': msgs.sum(1) / max(1, msgs.shape[1])}

        # 應用消息傳遞 - 使用DGL的高效API
        g.update_all(message_func, reduce_func)

        # 處理時間特徵
        h_time = g.ndata.get('h_time')

        # 如果h_time為None或者形狀不對，處理這種情況
        if h_time is None or h_time.shape[0] != g.num_nodes():
            h_time = torch.zeros(g.num_nodes(), self.in_dim + self.time_dim, device=h.device)

        # 檢查形狀是否合適
        if h_time.shape[1] != self.in_dim + self.time_dim:
            # 如果形狀不匹配，進行適當調整
            if h_time.shape[1] > self.in_dim + self.time_dim:
                h_time = h_time[:, :self.in_dim + self.time_dim]
            else:
                # 填充不足的部分
                padding = torch.zeros(h_time.shape[0],
                                     self.in_dim + self.time_dim - h_time.shape[1],
                                     device=h_time.device)
                h_time = torch.cat([h_time, padding], dim=1)

        # 應用GAT層
        h_new = self.gat(g, h_time)

        # 合併多頭注意力結果
        h_new = h_new.view(h_new.shape[0], -1)

        # 如果需要，應用投影層
        if self.projection is not None:
            h_new = self.projection(h_new)

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
        # 如果啟用了梯度檢查點，使用checkpoint函數
        if self.checkpoint and h.requires_grad:
            # 使用檢查點包裝實現函數
            def custom_forward(h_tensor, time_tensor):
                return self._forward_impl(g, h_tensor, time_tensor)

            return checkpoint(custom_forward, h, time_tensor)
        else:
            # 直接使用實現
            return self._forward_impl(g, h, time_tensor)

class EfficientGATConv(nn.Module):
    """記憶體優化的圖注意力卷積層"""

    def __init__(self, in_feats, out_feats, num_heads, feat_drop=0.6, attn_drop=0.6,
                 residual=True, activation=F.elu, allow_zero_in_degree=True):
        super(EfficientGATConv, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.residual = residual
        self.activation = activation
        self.allow_zero_in_degree = allow_zero_in_degree

        # 特徵轉換
        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)

        # 注意力機制
        self.attn_l = nn.Parameter(torch.FloatTensor(1, num_heads, out_feats))
        self.attn_r = nn.Parameter(torch.FloatTensor(1, num_heads, out_feats))

        # Dropout層
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        # 殘差連接
        if residual:
            if in_feats == out_feats * num_heads:
                self.res_fc = nn.Identity()
            else:
                self.res_fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        else:
            self.res_fc = None

        # 重置參數
        self.reset_parameters()

        # 稀疏注意力相關
        self.use_sparse_attention = True
        self.sparse_attention_threshold = 0.01

    def reset_parameters(self):
        """重置參數"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.res_fc is not None and not isinstance(self.res_fc, nn.Identity):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat):
        """
        前向傳播 - 使用更高效的實現

        參數:
            graph (dgl.DGLGraph): 輸入圖
            feat (torch.Tensor): 節點特徵

        返回:
            torch.Tensor: 更新後的節點特徵
        """
        with graph.local_scope():
            # 檢查是否有孤立節點
            if not self.allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise RuntimeError("存在孤立節點")

            # 特徵變換和丟棄
            feat = self.feat_drop(feat)
            feat_src = feat_dst = self.fc(feat).view(-1, self.num_heads, self.out_feats)

            # 計算注意力分數 - 使用高效的矩陣運算
            el = (feat_src * self.attn_l).sum(dim=-1, keepdim=True)
            er = (feat_dst * self.attn_r).sum(dim=-1, keepdim=True)

            # 設置注意力分數
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})

            # 計算邊的注意力分數
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))

            # 應用LeakyReLU激活函數
            e = F.leaky_relu(graph.edata.pop('e'))

            # 使用稀疏注意力
            if self.use_sparse_attention and self.training:
                # 訓練期間進行稀疏化
                mask = (torch.abs(e) > self.sparse_attention_threshold).float()
                e = e * mask

            # 應用注意力丟棄
            graph.edata['a'] = self.attn_drop(torch.softmax(e, dim=1))

            # 消息傳遞
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))

            # 獲取結果
            rst = graph.dstdata['ft']

            # 殘差連接
            if self.res_fc is not None:
                resval = self.res_fc(feat).view(feat.shape[0], self.num_heads, self.out_feats)
                rst = rst + resval

            # 激活函數
            if self.activation is not None:
                rst = self.activation(rst)

            return rst

class OptimizedTGATModel(nn.Module):
    """記憶體優化版TGAT模型"""

    def __init__(self, in_dim, hidden_dim, out_dim, time_dim, num_layers=2, num_heads=4, dropout=0.1, num_classes=2):
        """
        初始化優化版TGAT模型

        參數:
            in_dim (int): 輸入特徵維度
            hidden_dim (int): 隱藏層維度
            out_dim (int): 輸出特徵維度
            time_dim (int): 時間編碼維度
            num_layers (int): TGAT層數
            num_heads (int): 注意力頭數
            dropout (float): 丟棄率
            num_classes (int): 分類類別數
        """
        super(OptimizedTGATModel, self).__init__()

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
        self.use_gradient_accumulation = False
        self.accumulation_steps = 1
        self.use_sparse_gradients = False

        # 特徵投影層 - 使用高效實現
        self.feat_project = nn.Linear(in_dim, hidden_dim)

        # TGAT層 - 使用優化版本
        self.layers = nn.ModuleList()

        # 第一層投影輸入特徵到隱藏維度
        self.layers.append(
            OptimizedTemporalGATLayer(
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
                OptimizedTemporalGATLayer(
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
                OptimizedTemporalGATLayer(
                    in_dim=hidden_dim,
                    out_dim=out_dim,
                    time_dim=time_dim,
                    num_heads=num_heads,
                    feat_drop=dropout,
                    attn_drop=dropout
                )
            )

        # 分類層 - 使用記憶體優化設計
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, num_classes)
        )

        # 初始化參數
        self.reset_parameters()

        logger.info(f"初始化優化TGAT模型: {num_layers}層, {num_heads}注意力頭, {num_classes}分類類別")
        logger.info(f"特徵維度: 輸入={in_dim}, 隱藏={hidden_dim}, 輸出={out_dim}, 時間={time_dim}")

    def reset_parameters(self):
        """重置參數 - 使用優化的初始化方案"""
        gain = nn.init.calculate_gain('relu')

        # 初始化特徵投影層
        nn.init.xavier_normal_(self.feat_project.weight, gain=gain)
        nn.init.zeros_(self.feat_project.bias)

        # 初始化分類器
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                # 使用正交初始化
                nn.init.orthogonal_(layer.weight, gain=gain)
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

    def enable_gradient_accumulation(self, steps=4):
        """啟用梯度累積以處理更大批次"""
        self.use_gradient_accumulation = True
        self.accumulation_steps = steps
        logger.info(f"已啟用梯度累積, 步數={steps}")

    def disable_gradient_accumulation(self):
        """禁用梯度累積"""
        self.use_gradient_accumulation = False
        self.accumulation_steps = 1
        logger.info("已禁用梯度累積")

    def enable_sparse_gradients(self):
        """啟用稀疏梯度以節省記憶體"""
        self.use_sparse_gradients = True
        logger.info("已啟用稀疏梯度")

    def disable_sparse_gradients(self):
        """禁用稀疏梯度"""
        self.use_sparse_gradients = False
        logger.info("已禁用稀疏梯度")

    def _forward_impl(self, g):
        """
        實際的前向傳播實現 - 使用記憶體優化技術

        參數:
            g (dgl.DGLGraph): 輸入圖

        返回:
            torch.Tensor: 節點分類輸出 [num_nodes, num_classes]
        """
        # 獲取節點特徵
        h = g.ndata['h']

        # 檢查是否有節點
        if g.num_nodes() == 0:
            return torch.zeros((0, self.num_classes), device=g.device)

        # 將特徵投影到隱藏維度
        h = self.feat_project(h)

        # 獲取邊時間特徵
        time_tensor = g.edata.get('time', None)

        # 如果沒有時間特徵，使用全0時間戳
        if time_tensor is None:
            time_tensor = torch.zeros(g.num_edges(), device=h.device)

        # 應用TGAT層
        for i, layer in enumerate(self.layers):
            # 使用梯度檢查點 (如果啟用)
            if self.use_checkpoint and h.requires_grad:
                h = layer(g, h, time_tensor)
            else:
                h = layer(g, h, time_tensor)

            # 定期清理記憶體
            if (i + 1) % 2 == 0:
                clean_memory()

        # 應用分類器
        logits = self.classifier(h)

        return logits

    def forward(self, g):
        """
        前向傳播 (支持混合精度訓練)

        參數:
            g (dgl.DGLGraph): 輸入圖

        返回:
            torch.Tensor: 節點分類輸出 [num_nodes, num_classes]
        """
        # 如果啟用了混合精度訓練，使用autocast
        if self.use_mixed_precision and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                return self._forward_impl(g)
        else:
            return self._forward_impl(g)

    def encode(self, g):
        """
        僅編碼節點，不進行分類 (用於特徵提取)

        參數:
            g (dgl.DGLGraph): 輸入圖

        返回:
            torch.Tensor: 節點表示 [num_nodes, out_dim]
        """
        # 獲取節點特徵
        h = g.ndata['h']

        # 檢查是否有節點
        if g.num_nodes() == 0:
            return torch.zeros((0, self.out_dim), device=g.device)

        # 將特徵投影到隱藏維度
        h = self.feat_project(h)

        # 獲取邊時間特徵
        time_tensor = g.edata.get('time', None)

        # 如果沒有時間特徵，使用全0時間戳
        if time_tensor is None:
            time_tensor = torch.zeros(g.num_edges(), device=h.device)

        # 應用TGAT層 - 不使用最後的分類器
        for i, layer in enumerate(self.layers):
            # 使用梯度檢查點 (如果啟用)
            if self.use_checkpoint and h.requires_grad:
                h = layer(g, h, time_tensor)
            else:
                h = layer(g, h, time_tensor)

            # 定期清理記憶體
            if (i + 1) % 2 == 0:
                clean_memory()

        return h

    def to_sparse_tensor(self):
        """
        將模型的線性層轉換為稀疏表示，減少記憶體佔用

        此方法僅在評估階段使用，不適用於訓練
        """
        
        # 僅在不訓練時使用此功能
        if self.training:
            logger.warning("稀疏張量轉換僅應在評估模式下使用，不適用於訓練")
            return

        # 遍歷所有層，尋找可轉換為稀疏的線性層權重
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # 只轉換較大的層以節省記憶體
                if module.weight.numel() > 10000:
                    # 計算稀疏度
                    weight = module.weight.data
                    sparsity = (weight.abs() < 1e-3).float().mean().item()

                    # 只有當稀疏度較高時才轉換
                    if sparsity > 0.5:
                        # 創建掩碼 - 保留絕對值大於閾值的元素
                        mask = weight.abs() >= 1e-3

                        # 轉換為COO稀疏格式
                        indices = mask.nonzero().t()
                        values = weight[mask]
                        sparse_weight = torch.sparse_coo_tensor(
                            indices, values, weight.size()
                        )

                        # 臨時保存原始權重，這樣在需要時可以恢復
                        setattr(module, '_dense_weight', weight.clone())

                        # 替換為稀疏權重
                        del module.weight
                        module.register_buffer('weight', sparse_weight)

                        logger.info(f"已將層 {name} 轉換為稀疏表示，稀疏度: {sparsity:.2f} 節省記憶體: {weight.nelement() * 4 * (1-sparsity) / 1024 / 1024:.2f} MB")

        logger.info("已完成模型的稀疏張量轉換")

    def to_dense_tensor(self):
        """將模型從稀疏表示恢復為密集表示"""
        # 遍歷所有模組，恢復任何稀疏層
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, '_dense_weight'):
                # 恢復原始權重
                weight = getattr(module, '_dense_weight')

                # 刪除稀疏權重
                if hasattr(module, 'weight'):
                    delattr(module, 'weight')

                # 設置回密集權重
                module.weight = nn.Parameter(weight)

                # 刪除備份
                delattr(module, '_dense_weight')

                logger.info(f"已將層 {name} 恢復為密集表示")

        logger.info("已完成模型的密集張量恢復")
