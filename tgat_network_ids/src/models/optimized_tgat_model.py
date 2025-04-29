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
# from dgl.nn.pytorch import GATConv # 改用 EfficientGATConv
from torch.utils.checkpoint import checkpoint
import scipy.sparse as sp
from typing import Dict, Any, Optional, Union, List

# 導入記憶體優化工具和時間編碼
# 假設 utils 和 time_encoding 在相應的路徑下
# 注意：導入路徑可能需要根據您的實際專案結構調整
try:
    from ..utils.memory_utils import clean_memory
    # 注意: MemoryEfficientTimeEncoding 之前在此文件中定義，現在假設它來自 time_encoding.py
    from .time_encoding import MemoryEfficientTimeEncoding
except ImportError:
     print("警告：模型模組無法從相對路徑導入工具或時間編碼，嘗試直接導入。")
     # 如果直接執行此文件或結構不同，則回退
     try:
        from utils.memory_utils import clean_memory
        # 嘗試從上一層級的 models 目錄導入 time_encoding (如果 time_encoding.py 在 models 目錄下)
        # 或者根據您的實際結構調整
        from time_encoding import MemoryEfficientTimeEncoding
     except ImportError as ie:
         print(f"直接導入失敗: {ie}。請確保 utils 和 time_encoding 模組可用。")
         # 定義一個虛設的時間編碼器以允許程式碼運行，但功能會受限
         class MemoryEfficientTimeEncoding(nn.Module):
             """虛設的時間編碼器"""
             def __init__(self, dimension):
                 super().__init__()
                 self.dimension = dimension
                 print("警告: 使用了虛設的 MemoryEfficientTimeEncoding！功能將受限。")
             def forward(self, t):
                 if t is None or t.shape[0] == 0:
                      return torch.empty((0, self.dimension), device='cpu') # 返回空張量
                 # 返回一個形狀正確但值為零的張量
                 return torch.zeros((t.shape[0], self.dimension), device=t.device)


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- EfficientGATConv class ---
# (這個類別的定義與之前提供的修正版相同，請確保您使用的是修正後的版本)
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
        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(1, num_heads, out_feats))
        self.attn_r = nn.Parameter(torch.FloatTensor(1, num_heads, out_feats))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        if residual:
            # 如果維度匹配，使用 Identity
            # **修正**：應該比較 in_feats 和 out_feats * num_heads
            if in_feats == out_feats * num_heads:
                self.res_fc = nn.Identity()
            else:
                # 否則使用線性層調整維度
                # **修正**：res_fc 的輸入維度應該是原始的 in_feats，而不是 GAT 輸入的維度
                # 但是 GAT 的輸入是 in_feats + time_dim。這裡的殘差連接邏輯需要重新思考
                # 暫時保持原樣，但標註此處可能需要根據模型設計調整
                logger.warning("EfficientGATConv 中的殘差連接維度處理可能需要根據時間特徵整合方式調整")
                # 假設殘差連接作用在 GAT 輸入上（拼接後）
                # self.res_fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
                # 或者殘差連接作用在原始節點特徵上，然後再與 GAT 輸出結合，這更常見
                self.res_fc = nn.Linear(self.in_feats - self.time_dim if hasattr(self, 'time_dim') else self.in_feats, # 假設 time_dim 已知
                                        out_feats * num_heads, bias=False)


        else:
            self.res_fc = None
        self.reset_parameters()
        self.use_sparse_attention = True # 可設為參數
        self.sparse_attention_threshold = 0.01

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.res_fc is not None and isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat):
        """前向傳播"""
        with graph.local_scope():
            # --- 特徵處理和維度檢查 ---
            is_block = isinstance(graph, dgl.DGLHeteroGraph) and hasattr(graph, 'is_block') and graph.is_block
            if is_block:
                feat_src = feat
                feat_dst = feat[:graph.number_of_dst_nodes()]
                # **修正**: 獲取用於殘差連接的目標節點原始特徵
                # 這需要假設 feat 包含了所有源節點的特徵，且目標節點在前
                # 殘差連接通常作用在 *原始* 輸入特徵上
                feat_dst_orig_for_res = feat_dst[:, :self.res_fc.in_features] if self.res_fc else None
            else:
                feat_src = feat_dst = feat
                feat_dst_orig_for_res = feat[:, :self.res_fc.in_features] if self.res_fc else None

            if not self.allow_zero_in_degree and (graph.in_degrees() == 0).any():
                logger.warning("圖中存在入度為0的節點")

            # --- 特徵變換 ---
            h_src = self.feat_drop(feat_src)
            h_dst = self.feat_drop(feat_dst)
            # 確保 fc 輸入維度正確
            if h_src.shape[1] != self.fc.in_features:
                 raise ValueError(f"EfficientGATConv 前向傳播錯誤：輸入特徵維度 {h_src.shape[1]} 與 fc 層期望的 {self.fc.in_features} 不匹配。")

            feat_src_fc = self.fc(h_src).view(-1, self.num_heads, self.out_feats)
            # feat_dst_fc = self.fc(h_dst).view(-1, self.num_heads, self.out_feats) # er 計算不需要 fc

            # --- 計算注意力分數 ---
            el = (feat_src_fc * self.attn_l).sum(dim=-1, keepdim=True)
            # **修正**: er 應該作用在目標節點的 fc 輸出上
            # 由於 GATv1 的 er 只依賴目標節點，我們需要計算它
            er = (self.fc(h_dst).view(-1, self.num_heads, self.out_feats) * self.attn_r).sum(dim=-1, keepdim=True)


            graph.srcdata.update({'ft': feat_src_fc, 'el': el})
            graph.dstdata.update({'er': er})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = F.leaky_relu(graph.edata.pop('e'))

            # --- 稀疏注意力 ---
            if self.use_sparse_attention and self.training:
                mask = (torch.abs(e) > self.sparse_attention_threshold).float()
                e = e * mask

            # --- 計算權重並進行消息傳遞 ---
            graph.edata['a'] = self.attn_drop(dgl.ops.edge_softmax(graph, e))
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            rst = graph.dstdata['ft'] # [num_dst_nodes, num_heads, out_feats]

            # --- 殘差連接 ---
            if self.res_fc is not None:
                # **修正**: 確保殘差連接作用在正確的特徵上且維度匹配
                if feat_dst_orig_for_res is not None and feat_dst_orig_for_res.shape[1] == self.res_fc.in_features:
                     resval = self.res_fc(feat_dst_orig_for_res).view(graph.number_of_dst_nodes(), self.num_heads, self.out_feats)
                     rst = rst + resval
                else:
                     logger.warning(f"跳過殘差連接，因為維度不匹配或 feat_dst_orig_for_res 為空。需要維度: {self.res_fc.in_features if self.res_fc else 'N/A'}")


            # --- 激活函數 ---
            if self.activation is not None:
                rst = self.activation(rst)

            # --- 返回結果 ---
            # 將多頭結果合併
            return rst.view(graph.number_of_dst_nodes(), -1) # [num_dst_nodes, num_heads * out_feats]

# --- OptimizedTemporalGATLayer class (修正版) ---
class OptimizedTemporalGATLayer(nn.Module):
    """優化版時間圖注意力層 - 專注記憶體效率 (修正版)"""

    def __init__(self, in_dim, out_dim, time_dim, num_heads=4, feat_drop=0.6, attn_drop=0.6, residual=True):
        """初始化"""
        super(OptimizedTemporalGATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.time_dim = time_dim
        self.num_heads = num_heads

        # 時間編碼器
        self.time_enc = MemoryEfficientTimeEncoding(time_dim)
        # 特徵丟棄（注意：在 GATConv 前應用還是內部應用？）
        # 通常在傳遞給 GATConv 前應用一次即可
        self.feat_drop = nn.Dropout(feat_drop)

        # GAT 層的輸入維度是拼接後的維度
        gat_in_dim = in_dim + time_dim
        # GAT 層每個頭的輸出維度
        head_out_dim = out_dim // num_heads

        # 處理維度不能整除的情況
        if head_out_dim * num_heads != out_dim:
             logger.warning(f"OptimizedTemporalGATLayer: 輸出維度 ({out_dim}) 不能被注意力頭數 ({num_heads}) 整除。"
                            f"將調整頭輸出維度為 {head_out_dim}，實際層輸出維度為 {head_out_dim * num_heads}。")
        self.final_out_dim = head_out_dim * num_heads # 記錄實際輸出維度

        # 實例化 GAT 層
        self.gat = EfficientGATConv(
            in_feats=gat_in_dim, # 輸入是拼接後的特徵
            out_feats=head_out_dim,
            num_heads=num_heads,
            feat_drop=feat_drop, # GATConv 內部也處理 feat_drop
            attn_drop=attn_drop,
            residual=residual, # 在 GATConv 內部處理殘差連接
            activation=F.elu,
            allow_zero_in_degree=True
        )
        # 將 time_dim 傳遞給 GAT 層，以便殘差連接知道原始特徵維度
        # (這需要在 EfficientGATConv 中添加支持)
        # self.gat.time_dim = time_dim

        self.checkpoint = False # 梯度檢查點標誌

    def _forward_impl(self, g, h, time_tensor):
        """實際前向傳播 (修正時間整合邏輯)"""
        num_edges = g.num_edges()
        num_src_nodes = g.num_src_nodes()
        num_dst_nodes = g.num_dst_nodes()

        # --- 準備時間編碼 ---
        if num_edges == 0:
            # 沒有邊，直接返回零向量，維度匹配期望輸出
            return torch.zeros((num_dst_nodes, self.final_out_dim), device=h.device)

        if time_tensor is None or len(time_tensor) == 0:
            time_tensor = torch.zeros(num_edges, device=h.device)
        elif len(time_tensor) != num_edges:
            # 調整時間張量大小 (與之前邏輯相同)
            logger.debug(f"調整時間張量大小: {len(time_tensor)} -> {num_edges}")
            if len(time_tensor) < num_edges:
                 padding = time_tensor[-1:].expand(num_edges - len(time_tensor))
                 time_tensor = torch.cat([time_tensor, padding])
            else:
                 time_tensor = time_tensor[:num_edges]

        time_emb = self.time_enc(time_tensor) # [num_edges, time_dim]
        if time_emb.shape[0] != num_edges: # 再次檢查編碼器輸出
             logger.warning(f"時間編碼器輸出數量 ({time_emb.shape[0]}) 與邊數 ({num_edges}) 不符，使用零向量。")
             time_emb = torch.zeros((num_edges, self.time_dim), device=h.device)
        g.edata['time_feat'] = time_emb

        # --- 整合時間信息到節點特徵 ---
        # 聚合邊時間編碼到 *源* 節點上
        g.update_all(fn.copy_e('time_feat', 'm_time'), fn.mean('m_time', 'agg_time_feat_src'))
        agg_time_feat_src = g.srcdata.pop('agg_time_feat_src', None)

        # 處理可能沒有收到消息的源節點
        if agg_time_feat_src is None or agg_time_feat_src.shape[0] < num_src_nodes:
            full_agg_time_feat = torch.zeros(num_src_nodes, self.time_dim, device=h.device)
            if agg_time_feat_src is not None and agg_time_feat_src.shape[0] > 0:
                 valid_indices = torch.arange(agg_time_feat_src.shape[0], device=h.device)
                 full_agg_time_feat.index_copy_(0, valid_indices, agg_time_feat_src)
            agg_time_feat_src = full_agg_time_feat
        # 確保維度匹配
        if agg_time_feat_src.shape[1] != self.time_dim:
             # 進行填充或截斷
             if agg_time_feat_src.shape[1] < self.time_dim:
                 padding = torch.zeros(num_src_nodes, self.time_dim - agg_time_feat_src.shape[1], device=h.device)
                 agg_time_feat_src = torch.cat([agg_time_feat_src, padding], dim=1)
             else:
                  agg_time_feat_src = agg_time_feat_src[:, :self.time_dim]


        # --- 準備 GATConv 輸入 ---
        # 在拼接前應用 Dropout 到原始節點特徵
        h_src_dropped = self.feat_drop(h) # 維度 [num_src_nodes, in_dim]

        # 檢查維度
        if h_src_dropped.shape[0] != num_src_nodes:
             raise ValueError(f"Dropout 後的節點特徵數量 ({h_src_dropped.shape[0]}) 與源節點數 ({num_src_nodes}) 不匹配")
        if h_src_dropped.shape[1] != self.in_dim:
            raise ValueError(f"Dropout 後的節點特徵維度 ({h_src_dropped.shape[1]}) 與期望的輸入維度 ({self.in_dim}) 不匹配")
        if agg_time_feat_src.shape[0] != num_src_nodes:
            raise ValueError(f"聚合時間特徵數量 ({agg_time_feat_src.shape[0]}) 與源節點數 ({num_src_nodes}) 不匹配")
        if agg_time_feat_src.shape[1] != self.time_dim:
            raise ValueError(f"聚合時間特徵維度 ({agg_time_feat_src.shape[1]}) 與期望的時間維度 ({self.time_dim}) 不匹配")


        # 拼接特徵
        combined_feat = torch.cat([h_src_dropped, agg_time_feat_src], dim=1) # [num_src_nodes, in_dim + time_dim]

        # --- 應用 GAT 層 ---
        h_new = self.gat(g, combined_feat) # [num_dst_nodes, final_out_dim]

        return h_new

    def forward(self, g, h, time_tensor):
        """前向傳播（支持梯度檢查點）"""
        if self.checkpoint and self.training and h.requires_grad:
            # 梯度檢查點需要輸入是 Tensor，DGL 圖/塊可能不直接支持
            # logger.warning("梯度檢查點在此層可能不適用於 DGL 圖/塊，暫不啟用。")
            # 嘗試包裝，但可能失敗
            def wrapper(h_input, time_input):
                 # g 是外部變量
                 return self._forward_impl(g, h_input, time_input)
            try:
                 # 非重入式可能更兼容
                 return checkpoint(wrapper, h, time_tensor, use_reentrant=False)
            except Exception as e:
                 logger.warning(f"應用梯度檢查點失敗: {e}，執行標準前向傳播。")
                 return self._forward_impl(g, h, time_tensor)
        else:
            return self._forward_impl(g, h, time_tensor)


class OptimizedTGATModel(nn.Module):
    """記憶體優化版TGAT模型 (修正版：使用配置初始化)"""

    def __init__(self, config: Dict[str, Any]):
        """初始化"""
        super(OptimizedTGATModel, self).__init__()

        model_config = config.get('model', {})
        time_config = config.get('time_encoding', {})

        # --- 基本參數 ---
        self.in_dim: int = model_config.get('node_features', 0)
        self.hidden_dim: int = model_config.get('hidden_dim', 64)
        self.out_dim: int = model_config.get('out_dim', self.hidden_dim) # GAT 最後一層輸出維度
        self.time_dim: int = time_config.get('dimension', 16)
        self.num_layers: int = model_config.get('num_layers', 2)
        self.num_heads: int = model_config.get('num_heads', 4)
        self.dropout: float = model_config.get('dropout', 0.1)
        self.num_classes: int = model_config.get('num_classes', 2)

        if self.num_layers <= 0: raise ValueError("模型層數 'num_layers' 必須大於 0")

        # --- 優化標誌 ---
        self.use_checkpoint: bool = model_config.get('use_gradient_checkpointing', False)
        self.use_mixed_precision: bool = model_config.get('use_mixed_precision', False)
        # 梯度累積在 Trainer 中處理，模型不需要知道
        # self.use_gradient_accumulation: bool = model_config.get('use_gradient_accumulation', False)
        # self.accumulation_steps: int = model_config.get('gradient_accumulation_steps', 1)
        self.use_sparse_gradients: bool = model_config.get('use_sparse_gradients', False)

        # --- 層定義 ---
        # 輸入投影層 (延遲創建)
        self.feat_project: Optional[nn.Linear] = None
        if self.in_dim > 0:
            self.feat_project = nn.Linear(self.in_dim, self.hidden_dim)
            logger.info(f"基於配置創建了輸入維度為 {self.in_dim} 的投影層。")
        else:
             logger.warning("模型輸入維度 'node_features' 未提供或為0，投影層將延遲創建。")

        # GAT 層
        self.layers = nn.ModuleList()
        current_layer_in_dim = self.hidden_dim # 第一層 GAT 的輸入維度
        for i in range(self.num_layers):
            is_last_layer = (i == self.num_layers - 1)
            current_layer_out_dim = self.out_dim if is_last_layer else self.hidden_dim

            layer = OptimizedTemporalGATLayer(
                in_dim=current_layer_in_dim,
                out_dim=current_layer_out_dim,
                time_dim=self.time_dim,
                num_heads=self.num_heads,
                feat_drop=self.dropout,
                attn_drop=self.dropout,
                residual=True # 啟用殘差連接
            )
            self.layers.append(layer)
            current_layer_in_dim = layer.final_out_dim # 下一層的輸入維度是當前層的實際輸出維度

        # 分類器
        classifier_in_dim = current_layer_in_dim # 最後一層 GAT 的輸出維度
        classifier_hidden_dim = max(16, classifier_in_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(classifier_hidden_dim, self.num_classes)
        )

        self.reset_parameters() # 初始化參數
        logger.info(f"初始化優化 TGAT 模型: {self.num_layers} 層, {self.num_heads} 注意力頭, {self.num_classes} 分類類別")
        logger.info(f"維度: 輸入(待定)={self.in_dim}, 隱藏={self.hidden_dim}, GAT輸出={classifier_in_dim}, 時間={self.time_dim}")

    def set_input_dim(self, in_dim: int):
        """動態設置輸入維度並創建投影層"""
        if self.feat_project is None and in_dim > 0:
              self.in_dim = in_dim
              self.feat_project = nn.Linear(self.in_dim, self.hidden_dim)
              # 初始化新創建的層
              gain = nn.init.calculate_gain('relu')
              nn.init.xavier_normal_(self.feat_project.weight, gain=gain)
              if self.feat_project.bias is not None: nn.init.zeros_(self.feat_project.bias)
              logger.info(f"模型輸入維度已設置為: {self.in_dim}，輸入投影層已創建。")
        elif self.feat_project is not None and self.in_dim != in_dim:
              logger.warning(f"嘗試重新設置輸入維度，但投影層已存在。舊維度={self.in_dim}, 新維度={in_dim}。忽略此操作。")
        elif in_dim <= 0:
              logger.error(f"嘗試設置無效的輸入維度 (<= 0): {in_dim}")

    def reset_parameters(self):
        """重置參數"""
        gain = nn.init.calculate_gain('relu')
        if self.feat_project is not None:
            nn.init.xavier_normal_(self.feat_project.weight, gain=gain)
            if self.feat_project.bias is not None: nn.init.zeros_(self.feat_project.bias)

        for layer_module in self.layers: # 初始化 GAT 層內部參數
            if hasattr(layer_module, 'gat') and hasattr(layer_module.gat, 'reset_parameters'):
                layer_module.gat.reset_parameters()

        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=gain) # 使用正交初始化可能更好
                if layer.bias is not None: nn.init.zeros_(layer.bias)

    # --- enable/disable 方法 ---
    def enable_gradient_checkpointing(self):
        self.use_checkpoint = True
        for layer in self.layers: layer.checkpoint = True # 傳遞標誌
        logger.info("已啟用梯度檢查點 (應用於 TemporalGATLayer)")
    def disable_gradient_checkpointing(self):
        self.use_checkpoint = False
        for layer in self.layers: layer.checkpoint = False
        logger.info("已禁用梯度檢查點")
    # 其他 enable/disable 方法保持不變
    def enable_mixed_precision(self): self.use_mixed_precision = True; logger.info("啟用混合精度")
    def disable_mixed_precision(self): self.use_mixed_precision = False; logger.info("禁用混合精度")
    def enable_sparse_gradients(self): self.use_sparse_gradients = True; logger.info("啟用稀疏梯度 (實驗性)")
    def disable_sparse_gradients(self): self.use_sparse_gradients = False; logger.info("禁用稀疏梯度")


    def _forward_impl(self, g: Union[dgl.DGLGraph, List[dgl.DGLHeteroGraph]], h: torch.Tensor):
        """實際的前向傳播實現"""
        if self.feat_project is None:
             raise RuntimeError("模型輸入維度尚未設置 (feat_project is None)。請在訓練前調用 set_input_dim()。")

        is_blocks = isinstance(g, list)
        graph_for_check = g[0] if is_blocks else g

        if graph_for_check.num_nodes() == 0 or h.shape[0] == 0:
             # 處理空輸入
             num_dst_nodes = graph_for_check.num_dst_nodes() if is_blocks else graph_for_check.num_nodes()
             return torch.zeros((num_dst_nodes, self.num_classes), device=h.device)

        # 1. 特徵投影
        h = self.feat_project(h) # [num_input_nodes, hidden_dim]

        # 2. 應用 TGAT 層
        current_h = h
        for i, layer in enumerate(self.layers):
            block = g[i] if is_blocks else g

            # 獲取當前塊/圖的邊時間特徵
            time_tensor = None
            if 'time' in block.edata: time_tensor = block.edata['time']
            elif 'timestamp' in block.edata: time_tensor = block.edata['timestamp']
            # 處理時間張量
            if time_tensor is None:
                time_tensor = torch.zeros(block.num_edges(), device=current_h.device)
            else:
                # 確保時間張量在正確的設備上
                time_tensor = time_tensor.to(current_h.device)

            # 調用層的前向傳播
            current_h = layer(block, current_h, time_tensor)
            # layer 輸出的維度是 [num_dst_nodes_in_block, layer.final_out_dim]

            # 可選：內存清理
            if (i + 1) % 3 == 0: clean_memory()

        # 3. 應用分類器
        # current_h 是最後一層的輸出，維度 [num_final_dst_nodes, classifier_in_dim]
        logits = self.classifier(current_h)

        return logits


    def forward(self, g: Union[dgl.DGLGraph, List[dgl.DGLHeteroGraph]], h: Optional[torch.Tensor]=None):
        """
        前向傳播 (支持混合精度, 修正版)
        """
        # --- 獲取節點特徵 ---
        if h is None:
            graph_for_feat = g[0] if isinstance(g, list) else g
            if 'feat' in graph_for_feat.ndata:
                h = graph_for_feat.ndata['feat']
            elif 'h' in graph_for_feat.ndata: # 兼容舊鍵名
                h = graph_for_feat.ndata['h']
            else:
                raise ValueError("模型需要節點特徵 ('feat' 或 'h')，但圖/塊中未找到。")

        # --- 確保輸入維度已設置 ---
        if self.feat_project is None:
            # 如果尚未設置，根據當前輸入設置
            self.set_input_dim(h.shape[1])
            # 將模型移到正確的設備（如果 set_input_dim 新建了層）
            self.to(h.device) # 假設 h 在目標設備上

        # --- 確保數據類型 ---
        if not torch.is_floating_point(h): h = h.float()

        # --- 執行前向傳播 ---
        # 檢查是否啟用混合精度且 CUDA 可用
        use_amp = self.use_mixed_precision and torch.cuda.is_available() and h.device.type == 'cuda'

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = self._forward_impl(g, h)
                # 確保輸出是 float32 (如果損失函數需要)
                if logits.dtype == torch.float16:
                    logits = logits.float()
        else:
            logits = self._forward_impl(g, h)

        return logits


    def encode(self, g: Union[dgl.DGLGraph, List[dgl.DGLHeteroGraph]], h: Optional[torch.Tensor]=None):
        """
        僅編碼節點，不進行分類 (修正版)
        """
        # --- 獲取節點特徵 ---
        if h is None:
             graph_for_feat = g[0] if isinstance(g, list) else g
             if 'feat' in graph_for_feat.ndata: h = graph_for_feat.ndata['feat']
             elif 'h' in graph_for_feat.ndata: h = graph_for_feat.ndata['h']
             else: raise ValueError("模型需要節點特徵 ('feat' 或 'h')，但圖/塊中未找到。")

        # --- 確保輸入維度已設置 ---
        if self.feat_project is None: self.set_input_dim(h.shape[1]); self.to(h.device)

        # --- 確保數據類型 ---
        if not torch.is_floating_point(h): h = h.float()

        # --- 特徵投影 ---
        h = self.feat_project(h)

        # --- 應用 TGAT 層 ---
        current_h = h
        output_h = None
        for i, layer in enumerate(self.layers):
            block = g[i] if isinstance(g, list) else g
            # 獲取時間張量
            time_tensor = None
            if 'time' in block.edata: time_tensor = block.edata['time']
            elif 'timestamp' in block.edata: time_tensor = block.edata['timestamp']
            if time_tensor is None: time_tensor = torch.zeros(block.num_edges(), device=current_h.device)
            else: time_tensor = time_tensor.to(current_h.device)

            current_h = layer(block, current_h, time_tensor)
            output_h = current_h # 保存最後一層輸出
            if (i + 1) % 3 == 0: clean_memory()

        return output_h # 返回最後一層 GAT 的輸出


    # --- to_sparse_tensor / to_dense_tensor 方法 (保持不變) ---
    def to_sparse_tensor(self):
        """將模型的線性層轉換為稀疏表示 (僅評估)"""
        if self.training: logger.warning("稀疏張量轉換僅應在評估模式下使用"); return
        converted_count = 0
        for name, module in self.named_modules():
            # 只處理線性層，檢查權重是否存在且不是緩衝區（即參數）
            if isinstance(module, nn.Linear) and hasattr(module, 'weight') and isinstance(module.weight, nn.Parameter) \
               and not hasattr(module, '_dense_weight'):
                if module.weight.numel() > 10000: # 只轉換較大層
                    weight = module.weight.data.detach() # 使用 detach 避免影響梯度
                    sparsity = (weight.abs() < 1e-3).float().mean().item()
                    if sparsity > 0.5: # 稀疏度高於 50% 才轉換
                        mask = weight.abs() >= 1e-3
                        indices = mask.nonzero().t()
                        values = weight[mask]
                        if indices.numel() > 0: # 確保有非零元素
                             try:
                                 sparse_weight = torch.sparse_coo_tensor(indices, values, weight.size(), device=weight.device)
                                 # 保存密集權重，移除參數，註冊稀疏權重為 buffer
                                 setattr(module, '_dense_weight', weight.clone())
                                 del module.weight
                                 module.register_buffer('weight', sparse_weight) # 註冊為 buffer
                                 logger.info(f"層 {name} 轉為稀疏, 稀疏度: {sparsity:.2f}")
                                 converted_count += 1
                             except Exception as e:
                                 logger.error(f"轉換層 {name} 為稀疏時出錯: {e}")
                        else:
                             logger.info(f"層 {name} 稀疏度為 {sparsity:.2f} 但無有效權重，跳過轉換。")
        if converted_count > 0: logger.info(f"完成 {converted_count} 個線性層的稀疏張量轉換")
        else: logger.info("沒有符合條件的層被轉換為稀疏表示")

    def to_dense_tensor(self):
        """將模型從稀疏表示恢復為密集表示"""
        restored_count = 0
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, '_dense_weight'):
                dense_weight = getattr(module, '_dense_weight')
                # 移除 buffer
                if hasattr(module, 'weight') and not isinstance(module.weight, nn.Parameter):
                     delattr(module, 'weight')
                # 恢復為參數
                module.weight = nn.Parameter(dense_weight)
                delattr(module, '_dense_weight') # 移除備份
                logger.info(f"層 {name} 恢復為密集表示")
                restored_count += 1
        if restored_count > 0: logger.info(f"完成 {restored_count} 個線性層的密集張量恢復")
        else: logger.info("沒有層需要從稀疏表示恢復")
