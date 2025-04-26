#!/usr/bin/env python
# coding: utf-8 -*-

"""
時間編碼模組

實現多種時間編碼策略，以便比較不同編碼方法對模型性能的影響。
包含基本的餘弦編碼、可學習的時間嵌入、Time2Vec編碼等多種方法。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
import numpy as np
from collections import OrderedDict

logger = logging.getLogger(__name__)

class BaseTimeEncoding(nn.Module):
    """時間編碼基類
    
    所有時間編碼類別的父類，提供基本介面和功能
    """
    
    def __init__(self, dimension):
        """
        初始化時間編碼基類
        
        參數:
            dimension (int): 時間編碼的維度
        """
        super(BaseTimeEncoding, self).__init__()
        self.dimension = dimension
        
    def forward(self, t):
        """
        前向傳播方法，將時間轉換為編碼向量
        
        參數:
            t (torch.Tensor): 時間張量，可以是 1D 或 2D
            
        返回:
            torch.Tensor: 時間編碼向量
        """
        raise NotImplementedError("子類必須實現前向傳播方法")
    
    def _prepare_input(self, t):
        """
        預處理時間輸入，確保格式正確
        
        參數:
            t (torch.Tensor): 輸入時間張量
            
        返回:
            torch.Tensor: 預處理後的時間張量
        """
        # 將輸入轉換為浮點數並確保二維
        t = t.float()
        if t.dim() == 1:
            t = t.unsqueeze(1)
        return t

class MemoryEfficientTimeEncoding(BaseTimeEncoding):
    """記憶體高效的時間編碼模組 - 使用緩存和量化技術"""
    
    def __init__(self, dimension):
        super(MemoryEfficientTimeEncoding, self).__init__(dimension)
        
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
        # 準備輸入
        t = self._prepare_input(t)
        
        # 記錄查詢次數
        self.total_queries += 1
        
        # 報告緩存性能
        if self.total_queries % self.report_interval == 0:
            hit_rate = self.cache_hits / max(1, self.total_queries) * 100
            logger.debug(f"時間編碼緩存命中率: {hit_rate:.2f}% (hits: {self.cache_hits}, queries: {self.total_queries})")
        
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
            logger.debug(f"緩存已重置，剩餘 {len(self.cache)} 項")
        
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

class LearnableTimeEmbedding(BaseTimeEncoding):
    """可學習的時間嵌入模組
    
    將時間戳映射到可學習的嵌入向量空間，類似於NLP中的詞嵌入。
    支持連續時間值通過量化映射到離散嵌入。
    """
    
    def __init__(self, dimension, num_bins=1000, max_period=10000.0):
        """
        初始化可學習時間嵌入模組
        
        參數:
            dimension (int): 時間嵌入維度
            num_bins (int): 時間量化的桶數量
            max_period (float): 時間週期上限，超過此值將被循環處理
        """
        super(LearnableTimeEmbedding, self).__init__(dimension)
        
        self.num_bins = num_bins
        self.max_period = max_period
        
        # 創建可學習嵌入表
        self.time_embeddings = nn.Embedding(num_bins, dimension)
        
        # 初始化嵌入 - 使用正態分布初始化
        nn.init.normal_(self.time_embeddings.weight, mean=0.0, std=0.02)
        
        # 位置編碼用於平滑量化誤差
        self.use_position_encoding = True
        self.position_scale = 0.1
        
        # 緩存
        self.cache = {}
        self.cache_size_limit = 1000
        
    def _time_to_index(self, t):
        """將連續時間值轉換為離散索引"""
        # 首先將時間映射到[0, max_period]範圍
        t_normalized = t % self.max_period
        
        # 然後將時間映射到[0, num_bins-1]的整數索引
        indices = (t_normalized / self.max_period * self.num_bins).long()
        
        # 確保索引在有效範圍內
        indices = torch.clamp(indices, 0, self.num_bins - 1)
        
        return indices
    
    def _get_position_encoding(self, t):
        """獲取連續型位置編碼，用於補充離散化損失的信息"""
        device = t.device
        batch_size = t.size(0)
        
        # 歸一化時間
        t_normalized = (t % self.max_period) / self.max_period
        
        # 計算不同頻率的正弦和餘弦編碼
        position_enc = torch.zeros(batch_size, self.dimension, device=device)
        
        for i in range(self.dimension // 2):
            freq = math.exp(-(i * math.log(10000.0) / (self.dimension // 2)))
            position_enc[:, 2*i] = torch.sin(t_normalized * freq * 2 * math.pi)
            if 2*i + 1 < self.dimension:
                position_enc[:, 2*i+1] = torch.cos(t_normalized * freq * 2 * math.pi)
        
        return position_enc
    
    def forward(self, t):
        """前向傳播 - 計算學習的時間嵌入"""
        # 準備輸入
        t = self._prepare_input(t)
        
        # 將時間轉換為索引
        indices = self._time_to_index(t)
        
        # 獲取時間嵌入
        embeddings = self.time_embeddings(indices.squeeze())
        
        # 使用位置編碼進行平滑處理
        if self.use_position_encoding:
            position_enc = self._get_position_encoding(t)
            embeddings = embeddings + self.position_scale * position_enc
        
        return embeddings

class Time2VecEncoding(BaseTimeEncoding):
    """Time2Vec編碼模組
    
    基於論文 "Time2Vec: Learning a Vector Representation of Time"
    將時間表示為向量，結合了週期性和線性特性
    """
    
    def __init__(self, dimension):
        """
        初始化Time2Vec編碼模組
        
        參數:
            dimension (int): 輸出時間編碼維度
        """
        super(Time2VecEncoding, self).__init__(dimension)
        
        # Time2Vec中，第一個維度是線性的，其餘是週期性的
        self.periodic_dim = dimension - 1
        
        # 可學習的參數
        self.linear_w = nn.Parameter(torch.randn(1))
        self.linear_b = nn.Parameter(torch.randn(1))
        
        self.periodic_w = nn.Parameter(torch.randn(self.periodic_dim, 1))
        self.periodic_b = nn.Parameter(torch.randn(self.periodic_dim, 1))
        
        # 初始化權重
        self._reset_parameters()
        
        # 優化設置
        self.use_cache = True
        self.cache = {}
        self.cache_size_limit = 1000
    
    def _reset_parameters(self):
        """初始化模型參數"""
        # 線性部分
        nn.init.xavier_uniform_(self.linear_w.view(1, 1))
        nn.init.zeros_(self.linear_b)
        
        # 週期部分 - 初始化不同頻率
        frequency_bands = torch.exp(
            torch.arange(0, self.periodic_dim) * (-math.log(10000.0) / self.periodic_dim)
        ).view(self.periodic_dim, 1)
        
        # 設置參數以初始化合理的頻率
        nn.init.xavier_uniform_(self.periodic_w)
        self.periodic_w.data *= frequency_bands
        nn.init.uniform_(self.periodic_b, 0, 2 * math.pi)  # 均勻分佈的相位
        
    def forward(self, t):
        """前向傳播 - 計算Time2Vec編碼"""
        # 準備輸入
        t = self._prepare_input(t)
        
        batch_size = t.size(0)
        device = t.device
        
        # 檢查緩存
        if self.use_cache and batch_size == 1:
            # 對於單一時間點，可以使用緩存
            time_val = t.item()
            cache_key = f"{time_val:.4f}_{device}"
            
            if cache_key in self.cache:
                return self.cache[cache_key].view(1, -1)
        
        # 線性部分
        linear_part = self.linear_w * t + self.linear_b
        
        # 週期部分 (使用正弦函數)
        # 將時間從 [batch_size, 1] 擴展到 [batch_size, periodic_dim]
        t_expanded = t.expand(batch_size, self.periodic_dim)
        
        # 計算正弦輸入
        sin_input = torch.matmul(t_expanded, self.periodic_w.t()) + self.periodic_b.t()
        
        # 應用正弦非線性
        periodic_part = torch.sin(sin_input)
        
        # 連接線性和週期部分
        output = torch.cat([linear_part, periodic_part], dim=1)
        
        # 更新緩存
        if self.use_cache and batch_size == 1:
            if len(self.cache) >= self.cache_size_limit:
                # 簡單地清空緩存，或者可以實現更複雜的LRU策略
                self.cache = {}
            self.cache[cache_key] = output.detach().clone()
        
        return output

class FourierTimeEncoding(BaseTimeEncoding):
    """傅立葉時間編碼模組
    
    使用傅立葉級數的正弦和餘弦函數進行時間編碼，提供多頻率表示。
    支持可學習的頻率和相位。
    """
    
    def __init__(self, dimension, num_frequencies=None, learnable=True, base_freq=0.1, max_freq=None):
        """
        初始化傅立葉時間編碼模組
        
        參數:
            dimension (int): 時間編碼維度
            num_frequencies (int): 使用的頻率數量，若為None則設為dimension/2
            learnable (bool): 是否學習頻率和相位
            base_freq (float): 基本頻率
            max_freq (float): 最大頻率，若為None則設為100*base_freq
        """
        super(FourierTimeEncoding, self).__init__(dimension)
        
        self.dimension = dimension
        self.num_frequencies = num_frequencies if num_frequencies is not None else dimension // 2
        self.learnable = learnable
        self.base_freq = base_freq
        self.max_freq = max_freq if max_freq is not None else 100 * base_freq
        
        # 確保維度足夠
        if self.dimension < 2 * self.num_frequencies:
            self.num_frequencies = self.dimension // 2
            logger.warning(f"維度不足，調整頻率數量為: {self.num_frequencies}")
        
        # 初始化頻率和相位
        if self.learnable:
            # 對數空間中的均勻分佈的頻率
            log_min = math.log(self.base_freq)
            log_max = math.log(self.max_freq)
            
            # 初始化為對數均勻分佈
            init_freqs = torch.exp(torch.linspace(log_min, log_max, self.num_frequencies))
            
            # 創建可學習參數
            self.frequencies = nn.Parameter(init_freqs)
            self.phase_shifts = nn.Parameter(torch.zeros(self.num_frequencies))
        else:
            # 靜態頻率 - 對數空間中等距
            log_timescale_increment = (math.log(self.max_freq / self.base_freq) / 
                                      (self.num_frequencies - 1 if self.num_frequencies > 1 else 1))
            
            inv_timescales = self.base_freq * torch.exp(
                torch.arange(self.num_frequencies) * -log_timescale_increment
            )
            
            self.register_buffer('frequencies', inv_timescales)
            self.register_buffer('phase_shifts', torch.zeros(self.num_frequencies))
        
        # 填充剩餘維度 (如果有)
        self.linear_projection = None
        remaining_dims = self.dimension - 2 * self.num_frequencies
        
        if remaining_dims > 0:
            self.linear_projection = nn.Linear(1, remaining_dims)
            nn.init.xavier_uniform_(self.linear_projection.weight)
            nn.init.zeros_(self.linear_projection.bias)
    
    def forward(self, t):
        """前向傳播 - 計算傅立葉時間編碼"""
        # 準備輸入
        t = self._prepare_input(t)
        
        batch_size = t.size(0)
        device = t.device
        
        # 計算各頻率的正弦和餘弦
        # 擴展時間和頻率以進行批量計算
        t_expanded = t.expand(batch_size, self.num_frequencies)
        freqs_expanded = self.frequencies.view(1, -1).expand(batch_size, -1)
        phases_expanded = self.phase_shifts.view(1, -1).expand(batch_size, -1)
        
        # 計算角度
        angles = 2 * math.pi * freqs_expanded * t_expanded + phases_expanded
        
        # 計算正弦和餘弦值
        sin_vals = torch.sin(angles)
        cos_vals = torch.cos(angles)
        
        # 將正弦和餘弦值交織到一起
        fourier_features = torch.zeros(batch_size, 2 * self.num_frequencies, device=device)
        fourier_features[:, 0::2] = sin_vals
        fourier_features[:, 1::2] = cos_vals
        
        # 處理可能的剩餘維度
        if self.linear_projection is not None:
            linear_feats = self.linear_projection(t)
            # 連接傅立葉和線性特徵
            encoding = torch.cat([fourier_features, linear_feats], dim=1)
        else:
            encoding = fourier_features
        
        return encoding

class TimeEncodingFactory:
    """時間編碼工廠類
    
    用於創建各種時間編碼實例的工廠類，基於配置提供合適的編碼器
    """
    
    @staticmethod
    def create_time_encoding(config):
        """
        基於配置創建時間編碼實例
        
        參數:
            config (dict): 包含時間編碼配置的字典
            
        返回:
            BaseTimeEncoding: 時間編碼實例
        """
        # 獲取配置或使用默認值
        if not isinstance(config, dict):
            config = {}
            
        time_encoding_config = config.get('time_encoding', {})
        
        # 獲取編碼類型和維度
        encoding_type = time_encoding_config.get('method', 'memory_efficient')
        dimension = time_encoding_config.get('dimension', 64)
        
        # 根據類型創建相應的編碼器
        if encoding_type == 'memory_efficient':
            return MemoryEfficientTimeEncoding(dimension)
        elif encoding_type == 'learnable':
            num_bins = time_encoding_config.get('num_bins', 1000)
            max_period = time_encoding_config.get('max_period', 10000.0)
            return LearnableTimeEmbedding(dimension, num_bins, max_period)
        elif encoding_type == 'time2vec':
            return Time2VecEncoding(dimension)
        elif encoding_type == 'fourier':
            num_frequencies = time_encoding_config.get('fourier_freqs', dimension // 2)
            learnable = time_encoding_config.get('learnable_freqs', True)
            base_freq = time_encoding_config.get('base_freq', 0.1)
            max_freq = time_encoding_config.get('max_freq', None)
            return FourierTimeEncoding(dimension, num_frequencies, learnable, base_freq, max_freq)
        else:
            logger.warning(f"未知的時間編碼類型: {encoding_type}，使用默認記憶體高效編碼")
            return MemoryEfficientTimeEncoding(dimension)
