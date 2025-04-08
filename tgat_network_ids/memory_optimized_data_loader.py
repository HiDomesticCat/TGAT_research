#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
記憶體優化版資料載入與預處理模組

此模組是 data_loader.py 的記憶體優化版本，提供以下功能：
1. 增量式資料加載
2. 預處理資料的保存與加載
3. 記憶體映射大型數據集
4. 資料壓縮
5. DataFrame 記憶體優化
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm
import gc
import time
import pickle
import glob
from datetime import datetime

# 導入記憶體優化工具
from memory_utils import (
    memory_mapped_array, load_memory_mapped_array, save_dataframe_chunked, 
    load_dataframe_chunked, optimize_dataframe_memory, clean_memory, 
    memory_usage_decorator, track_memory_usage, print_memory_usage, 
    get_memory_usage, get_memory_optimization_suggestions, 
    adaptive_batch_size, detect_memory_leaks, limit_gpu_memory
)

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryOptimizedDataLoader:
    """記憶體優化版資料載入與預處理類別"""
    
    def __init__(self, config):
        """
        初始化資料載入器
        
        參數:
            config (dict): 配置字典，包含以下鍵：
                - data.path: 資料集路徑
                - data.test_size: 測試集比例
                - data.random_state: 隨機種子
                - data.batch_size: 批次大小
                - data.use_memory_mapping: 是否使用記憶體映射
                - data.save_preprocessed: 是否保存預處理後的資料
                - data.preprocessed_path: 預處理資料保存路徑
                - data.incremental_loading: 是否使用增量式資料加載
                - data.chunk_size_mb: 增量加載的塊大小 (MB)
                - data.use_compression: 是否使用資料壓縮
                - data.compression_format: 壓縮格式
        """
        # 從配置中提取資料相關設置
        data_config = config.get('data', {})
        self.data_path = data_config.get('path', './data')
        self.test_size = data_config.get('test_size', 0.2)
        self.random_state = data_config.get('random_state', 42)
        self.batch_size = data_config.get('batch_size', 128)
        
        # 記憶體優化相關設置
        self.use_memory_mapping = data_config.get('use_memory_mapping', True)
        self.save_preprocessed = data_config.get('save_preprocessed', True)
        self.preprocessed_path = data_config.get('preprocessed_path', './preprocessed_data')
        self.incremental_loading = data_config.get('incremental_loading', True)
        self.chunk_size_mb = data_config.get('chunk_size_mb', 100)
        self.use_compression = data_config.get('use_compression', True)
        self.compression_format = data_config.get('compression_format', 'gzip')
        
        # 確保預處理資料目錄存在
        if self.save_preprocessed and not os.path.exists(self.preprocessed_path):
            os.makedirs(self.preprocessed_path)
        
        # 初始化預處理工具
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # 初始化資料變數
        self.df = None
        self.features = None
        self.target = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # 記錄預處理時間戳記，用於生成唯一文件名
        self.preprocess_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"初始化記憶體優化版資料載入器: 資料路徑={self.data_path}, 測試集比例={self.test_size}")
        logger.info(f"記憶體優化設置: 使用記憶體映射={self.use_memory_mapping}, 保存預處理資料={self.save_preprocessed}")
        logger.info(f"增量加載設置: 啟用={self.incremental_loading}, 塊大小={self.chunk_size_mb}MB")
    
    def _get_preprocessed_files(self):
        """獲取預處理文件路徑"""
        features_file = os.path.join(self.preprocessed_path, "features.npy")
        target_file = os.path.join(self.preprocessed_path, "target.npy")
        feature_names_file = os.path.join(self.preprocessed_path, "feature_names.pkl")
        scaler_file = os.path.join(self.preprocessed_path, "scaler.pkl")
        label_encoder_file = os.path.join(self.preprocessed_path, "label_encoder.pkl")
        train_indices_file = os.path.join(self.preprocessed_path, "train_indices.npy")
        test_indices_file = os.path.join(self.preprocessed_path, "test_indices.npy")
        
        return {
            'features': features_file,
            'target': target_file,
            'feature_names': feature_names_file,
            'scaler': scaler_file,
            'label_encoder': label_encoder_file,
            'train_indices': train_indices_file,
            'test_indices': test_indices_file
        }
    
    def _check_preprocessed_exists(self):
        """檢查預處理文件是否存在"""
        files = self._get_preprocessed_files()
        return all(os.path.exists(f) for f in files.values())
    
    @memory_usage_decorator
    def load_data(self):
        """載入資料集"""
        logger.info(f"載入資料集從: {self.data_path}")
        
        # 檢查是否有預處理好的資料
        if self.save_preprocessed and self._check_preprocessed_exists():
            logger.info("發現預處理資料，直接加載")
            self._load_preprocessed_data()
            return self.df
        
        # 檢查是csv檔案還是目錄
        if os.path.isdir(self.data_path):
            # 載入目錄中的所有CSV檔案
            all_files = [os.path.join(self.data_path, f) 
                        for f in os.listdir(self.data_path) 
                        if f.endswith('.csv')]
            
            if not all_files:
                raise ValueError("未找到有效的CSV檔案")
            
            if self.incremental_loading:
                # 增量式加載
                self._load_data_incrementally(all_files)
            else:
                # 一次性加載所有文件
                self._load_data_at_once(all_files)
        else:
            # 載入單一CSV檔案
            if self.incremental_loading:
                self._load_single_file_incrementally(self.data_path)
            else:
                self.df = pd.read_csv(self.data_path, low_memory=False)
                logger.info(f"載入資料集形狀: {self.df.shape}")
        
        # 優化 DataFrame 記憶體使用
        if self.df is not None:
            self.df = optimize_dataframe_memory(self.df)
        
        return self.df
    
    def _load_data_incrementally(self, file_list):
        """增量式加載多個文件"""
        logger.info(f"增量式加載 {len(file_list)} 個文件")
        
        # 估計每個文件的大小
        file_sizes = [os.path.getsize(f) / (1024 * 1024) for f in file_list]  # MB
        total_size = sum(file_sizes)
        
        logger.info(f"總資料大小: {total_size:.2f} MB")
        
        # 根據塊大小計算每個文件要讀取的行數
        chunks = []
        
        for i, file in enumerate(tqdm(file_list, desc="增量加載文件")):
            file_size = file_sizes[i]
            
            # 如果文件小於塊大小，直接讀取整個文件
            if file_size <= self.chunk_size_mb:
                try:
                    df_chunk = pd.read_csv(file, low_memory=False, dtype={
                        'Protocol': 'category',
                        'Destination Port': 'int32',
                        'Flow Duration': 'float32'
                    })
                    chunks.append(df_chunk)
                    logger.info(f"加載文件 {i+1}/{len(file_list)}: {file}, 形狀: {df_chunk.shape}")
                except Exception as e:
                    logger.error(f"載入 {file} 時發生錯誤: {str(e)}")
            else:
                # 分塊讀取大文件
                chunk_reader = pd.read_csv(file, chunksize=int(self.chunk_size_mb * 10000 / file_size),
                                         low_memory=False, dtype={
                                             'Protocol': 'category',
                                             'Destination Port': 'int32',
                                             'Flow Duration': 'float32'
                                         })
                
                for j, sub_chunk in enumerate(chunk_reader):
                    chunks.append(sub_chunk)
                    logger.info(f"加載文件 {i+1}/{len(file_list)} 塊 {j+1}: {file}, 形狀: {sub_chunk.shape}")
                    
                    # 定期清理記憶體
                    if (j + 1) % 5 == 0:
                        clean_memory()
            
            # 每處理完一個文件，清理記憶體
            clean_memory()
        
        # 合併所有塊
        logger.info(f"合併 {len(chunks)} 個資料塊")
        self.df = pd.concat(chunks, ignore_index=True)
        logger.info(f"合併資料集形狀: {self.df.shape}")
        
        # 釋放記憶體
        del chunks
        clean_memory()
    
    def _load_single_file_incrementally(self, file_path):
        """增量式加載單個文件"""
        logger.info(f"增量式加載單個文件: {file_path}")
        
        # 估計文件大小
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        logger.info(f"文件大小: {file_size:.2f} MB")
        
        # 如果文件小於塊大小，直接讀取整個文件
        if file_size <= self.chunk_size_mb:
            self.df = pd.read_csv(file_path, low_memory=False)
            logger.info(f"載入資料集形狀: {self.df.shape}")
        else:
            # 計算每塊的行數 (粗略估計)
            rows_per_mb = 10000  # 假設每MB約有10000行
            chunk_size = int(self.chunk_size_mb * rows_per_mb)
            
            # 分塊讀取
            chunks = []
            for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, low_memory=False)):
                chunks.append(chunk)
                logger.info(f"已加載塊 {i+1}, 形狀: {chunk.shape}")
                
                # 定期清理記憶體
                if (i + 1) % 5 == 0:
                    clean_memory()
            
            # 合併所有塊
            logger.info(f"合併 {len(chunks)} 個資料塊")
            self.df = pd.concat(chunks, ignore_index=True)
            logger.info(f"合併資料集形狀: {self.df.shape}")
            
            # 釋放記憶體
            del chunks
            clean_memory()
    
    def _load_data_at_once(self, file_list):
        """一次性加載所有文件"""
        logger.info(f"一次性加載 {len(file_list)} 個文件")
        
        df_list = []
        for file in tqdm(file_list, desc="載入資料集"):
            try:
                # 使用低記憶體載入
                df = pd.read_csv(file, low_memory=False, dtype={
                    'Protocol': 'category',
                    'Destination Port': 'int32',
                    'Flow Duration': 'float32'
                })
                df_list.append(df)
                logger.info(f"成功載入資料: {file}, 形狀: {df.shape}")
            except Exception as e:
                logger.error(f"載入 {file} 時發生錯誤: {str(e)}")
            
            # 及時釋放記憶體
            gc.collect()
        
        if df_list:
            self.df = pd.concat(df_list, ignore_index=True)
            logger.info(f"合併資料集形狀: {self.df.shape}")
        else:
            raise ValueError("未找到有效的CSV檔案")
        
        # 釋放記憶體
        del df_list
        clean_memory()
    
    @memory_usage_decorator
    def preprocess(self):
        """預處理資料集"""
        # 檢查是否有預處理好的資料
        if self.save_preprocessed and self._check_preprocessed_exists():
            logger.info("發現預處理資料，直接加載")
            return self._load_preprocessed_data()
        
        if self.df is None:
            self.load_data()

        logger.info("開始資料預處理...")

        # 清理列名，移除前後空格
        self.df.columns = [col.strip() for col in self.df.columns]

        # 處理缺失值
        logger.info("處理缺失值...")
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # 分別處理不同類型的欄位
        numeric_columns = self.df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns

        # 對數值列填充中位數
        for col in numeric_columns:
            self.df[col].fillna(self.df[col].median())

        # 對分類列填充最常見值
        categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            self.df[col].fillna(self.df[col].mode()[0])

        # 檢查並移除常數特徵
        nunique = self.df.nunique()
        cols_to_drop = nunique[nunique <= 1].index
        self.df.drop(columns=cols_to_drop, inplace=True)
        logger.info(f"移除 {len(cols_to_drop)} 個常數特徵")

        # 自動檢測標籤欄位
        label_column = self._detect_label_column()

        # 預處理 IP 和其他特殊欄位
        ip_columns = [col for col in self.df.columns if 'ip' in col.lower()]
        for col in ip_columns:
            # 將 IP 地址轉換為分類編碼
            self.df[col] = pd.Categorical(self.df[col]).codes

        # 處理時間戳記欄位
        timestamp_columns = [col for col in self.df.columns if 'timestamp' in col.lower()]
        for col in timestamp_columns:
            try:
                self.df[col] = pd.to_datetime(self.df[col]).astype(int) / 10**9
            except:
                # 如果轉換失敗，嘗試編碼
                self.df[col] = pd.Categorical(self.df[col]).codes

        # 處理分類特徵
        categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            if col != label_column:
                self.df[col] = pd.Categorical(self.df[col]).codes

        # 標籤編碼
        logger.info("編碼標籤...")
        self.df[label_column] = self.label_encoder.fit_transform(self.df[label_column])

        # 排除不需要的特徵欄位
        excluded_patterns = ['flow id', 'timestamp', 'unnamed']
        feature_cols = [
            col for col in self.df.columns 
            if not any(pattern in col.lower() for pattern in excluded_patterns)
            and col != label_column 
            and self.df[col].dtype in ['int64', 'float64', 'int32', 'float32']
        ]

        # 準備特徵和標籤
        self.feature_names = feature_cols
        
        # 使用記憶體映射或普通數組
        if self.use_memory_mapping:
            # 創建記憶體映射數組
            features_file = os.path.join(self.preprocessed_path, f"features_{self.preprocess_timestamp}.dat")
            self.features = memory_mapped_array(
                shape=(len(self.df), len(feature_cols)),
                dtype=np.float32,
                filename=features_file
            )
            
            # 填充特徵數據
            for i, col in enumerate(tqdm(feature_cols, desc="填充特徵數據")):
                self.features[:, i] = self.df[col].values
                
                # 定期清理記憶體
                if (i + 1) % 10 == 0:
                    clean_memory()
        else:
            # 使用普通數組
            self.features = self.df[feature_cols].values
        
        self.target = self.df[label_column]

        # 特徵標準化
        logger.info("特徵標準化...")
        
        if self.use_memory_mapping:
            # 分批標準化
            batch_size = min(10000, len(self.features))
            num_batches = (len(self.features) + batch_size - 1) // batch_size
            
            # 先擬合整個數據集
            self.scaler.fit(self.features)
            
            # 分批轉換
            for i in tqdm(range(num_batches), desc="標準化特徵"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(self.features))
                
                # 轉換批次
                self.features[start_idx:end_idx] = self.scaler.transform(self.features[start_idx:end_idx])
                
                # 定期清理記憶體
                if (i + 1) % 10 == 0:
                    clean_memory()
        else:
            self.features = self.scaler.fit_transform(self.features)

        logger.info("資料預處理完成")
        
        # 保存預處理後的資料
        if self.save_preprocessed:
            self._save_preprocessed_data()
        
        # 釋放記憶體
        clean_memory()

        return self.features, self.target
    
    def _save_preprocessed_data(self):
        """保存預處理後的資料"""
        logger.info(f"保存預處理資料到: {self.preprocessed_path}")
        
        files = self._get_preprocessed_files()
        
        # 保存特徵
        if self.use_memory_mapping:
            # 如果使用記憶體映射，特徵已經保存在文件中
            # 創建一個符號鏈接或複製文件
            features_file = files['features']
            if not os.path.exists(features_file):
                os.symlink(self.features.filename, features_file)
        else:
            # 保存為 numpy 數組
            np.save(files['features'], self.features)
        
        # 保存標籤
        np.save(files['target'], self.target.values)
        
        # 保存特徵名稱
        with open(files['feature_names'], 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        # 保存標準化器
        with open(files['scaler'], 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # 保存標籤編碼器
        with open(files['label_encoder'], 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # 如果已經拆分了訓練集和測試集，也保存索引
        if hasattr(self, 'train_indices') and hasattr(self, 'test_indices'):
            np.save(files['train_indices'], self.train_indices)
            np.save(files['test_indices'], self.test_indices)
        
        logger.info("預處理資料保存完成")
    
    def _load_preprocessed_data(self):
        """加載預處理後的資料"""
        with track_memory_usage("加載預處理資料", detailed=True):
            logger.info(f"加載預處理資料從: {self.preprocessed_path}")
            
            files = self._get_preprocessed_files()
            
            # 加載特徵
            if self.use_memory_mapping:
                # 使用記憶體映射加載
                try:
                    # 使用新的加載函數
                    self.features = load_memory_mapped_array(files['features'], mode='r')
                except Exception as e:
                    logger.warning(f"使用新函數加載記憶體映射失敗: {str(e)}，嘗試舊方法")
                    # 回退到舊方法
                    self.features = np.memmap(
                        files['features'],
                        dtype=np.float32,
                        mode='r',
                        shape=tuple(np.load(f"{files['features']}.shape.npy"))
                    )
            else:
                # 直接加載 numpy 數組
                self.features = np.load(files['features'])
        
        # 加載標籤
        target_values = np.load(files['target'])
        self.target = pd.Series(target_values)
        
        # 加載特徵名稱
        with open(files['feature_names'], 'rb') as f:
            self.feature_names = pickle.load(f)
        
        # 加載標準化器
        with open(files['scaler'], 'rb') as f:
            self.scaler = pickle.load(f)
        
        # 加載標籤編碼器
        with open(files['label_encoder'], 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # 如果存在訓練集和測試集索引，也加載它們
        if os.path.exists(files['train_indices']) and os.path.exists(files['test_indices']):
            self.train_indices = np.load(files['train_indices'])
            self.test_indices = np.load(files['test_indices'])
            
            # 使用索引拆分數據
            self.X_train = self.features[self.train_indices]
            self.X_test = self.features[self.test_indices]
            self.y_train = self.target.iloc[self.train_indices]
            self.y_test = self.target.iloc[self.test_indices]
            
            logger.info(f"已加載拆分的訓練集和測試集")
            logger.info(f"  訓練集: X_train {self.X_train.shape}, y_train {self.y_train.shape}")
            logger.info(f"  測試集: X_test {self.X_test.shape}, y_test {self.y_test.shape}")
        
        logger.info(f"預處理資料加載完成: 特徵形狀 {self.features.shape}, 標籤形狀 {self.target.shape}")
        
        return self.features, self.target
    
    def _detect_label_column(self):
        """自動檢測標籤欄位"""
        possible_label_columns = ['Label', 'label', 'attack_type']
        for col in possible_label_columns:
            if col in self.df.columns:
                return col
        
        # 如果找不到標準標籤欄位，使用最後一個欄位
        label_column = self.df.columns[-1]
        logger.warning(f"未找到標準標籤欄位，使用預設欄位: {label_column}")
        return label_column
    
    @memory_usage_decorator
    def split_data(self):
        """拆分訓練和測試資料集"""
        if self.features is None or self.target is None:
            self.preprocess()
        
        # 檢查是否已經加載了拆分的數據
        if self.X_train is not None and self.X_test is not None:
            logger.info("使用已加載的訓練集和測試集拆分")
            return self.X_train, self.X_test, self.y_train, self.y_test
        
        logger.info(f"拆分資料集: 測試集比例 {self.test_size}")
        
        # 生成訓練集和測試集的索引
        indices = np.arange(len(self.target))
        self.train_indices, self.test_indices = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.target  # 確保標籤分布一致
        )
        
        # 使用索引獲取訓練集和測試集
        if self.use_memory_mapping:
            # 對於記憶體映射數組，使用索引切片
            self.X_train = self.features[self.train_indices]
            self.X_test = self.features[self.test_indices]
        else:
            # 對於普通數組，直接索引
            self.X_train = self.features[self.train_indices]
            self.X_test = self.features[self.test_indices]
        
        # 對於標籤，使用 iloc 索引
        self.y_train = self.target.iloc[self.train_indices]
        self.y_test = self.target.iloc[self.test_indices]
        
        # 如果啟用了保存預處理資料，保存訓練集和測試集索引
        if self.save_preprocessed:
            files = self._get_preprocessed_files()
            np.save(files['train_indices'], self.train_indices)
            np.save(files['test_indices'], self.test_indices)
            logger.info("已保存訓練集和測試集索引")
        
        logger.info(f"訓練集: X_train {self.X_train.shape}, y_train {self.y_train.shape}")
        logger.info(f"測試集: X_test {self.X_test.shape}, y_test {self.y_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_attack_types(self):
        """取得攻擊類型對應表"""
        return {i: label for i, label in enumerate(self.label_encoder.classes_)}
    
    @memory_usage_decorator
    def get_temporal_edges(self, max_edges=500000):
        """
        建立時間性邊緣關係
        
        參數:
            max_edges (int): 最大邊數量限制
            
        返回:
            list: 邊列表，每個元素為 (src, dst, time, edge_feat)
        """
        # 重新載入原始資料以獲取 IP 位址和時間戳記資訊
        if self.df is None:
            self.load_data()
        
        edges = []
        
        # 獲取原始 IP 和時間戳記欄位
        ip_src_col = 'Source IP' if 'Source IP' in self.df.columns else 'Src IP'
        ip_dst_col = 'Destination IP' if 'Destination IP' in self.df.columns else 'Dst IP'
        time_col = 'Timestamp' if 'Timestamp' in self.df.columns else 'Flow Start Time'
        
        # 將時間戳記轉換為數值
        if isinstance(self.df[time_col].iloc[0], str):
            self.df[time_col] = pd.to_datetime(self.df[time_col]).astype(int) / 10**9
        
        # 為每個封包分配索引 (節點ID)
        node_indices = {i: i for i in range(len(self.df))}
        
        # 根據 IP 對關係建立邊
        logger.info("分析 IP 對關係...")
        
        # 使用分組操作而不是字典，減少記憶體使用
        ip_pairs = []
        
        # 分批處理以減少記憶體使用
        batch_size = min(100000, len(self.df))
        num_batches = (len(self.df) + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="收集 IP 對"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(self.df))
            
            # 處理當前批次
            batch_df = self.df.iloc[start_idx:end_idx]
            
            # 獲取唯一的 IP 對
            batch_pairs = batch_df.groupby([ip_src_col, ip_dst_col]).indices
            
            # 處理每個 IP 對
            for (src_ip, dst_ip), indices in batch_pairs.items():
                # 限制每個 IP 對處理的封包數量
                max_packets_per_pair = 50
                if len(indices) > max_packets_per_pair:
                    indices = sorted(indices, key=lambda x: batch_df.iloc[x - start_idx][time_col])[:max_packets_per_pair]
                
                # 添加到 IP 對列表
                ip_pairs.append((src_ip, dst_ip, indices))
            
            # 清理記憶體
            del batch_df
            clean_memory()
        
        # 使用閾值定義什麼是"時間接近"
        time_threshold = 0.5  # 縮短時間閾值以減少邊數量
        
        # 處理每個 IP 對，建立時間性邊
        for i, (src_ip, dst_ip, indices) in enumerate(tqdm(ip_pairs, desc="建立時間性邊")):
            # 每 1000 個 IP 對輸出一次日誌
            if i % 1000 == 0:
                logger.info(f"已處理 {i}/{len(ip_pairs)} 個 IP 對，當前邊數: {len(edges)}")
            
            # 如果邊數達到限制則提前結束
            if len(edges) >= max_edges:
                logger.info(f"已達到最大邊數限制 {max_edges}，提前結束邊創建")
                break
            
            # 對於相同 IP 對中的每對連續封包，建立時間性邊
            indices_list = sorted(indices, key=lambda x: self.df.iloc[x][time_col])
            
            for j in range(len(indices_list) - 1):
                idx1 = indices_list[j]
                idx2 = indices_list[j + 1]
                
                time1 = self.df.iloc[idx1][time_col]
                time2 = self.df.iloc[idx2][time_col]
                
                # 時間差
                time_diff = time2 - time1
                
                # 如果時間接近，建立邊
                if time_diff <= time_threshold:
                    # 邊特徵: 時間差、封包大小比例等
                    size1 = self.df.iloc[idx1]['Fwd Packet Length Mean'] if 'Fwd Packet Length Mean' in self.df.columns else 0
                    size2 = self.df.iloc[idx2]['Fwd Packet Length Mean'] if 'Fwd Packet Length Mean' in self.df.columns else 0
                    
                    edge_feat = [time_diff, size2/size1 if size1 > 0 else 0]
                    
                    # 添加有向邊 (前 -> 後)
                    edges.append((node_indices[idx1], node_indices[idx2], time2, edge_feat))
                    
                    # 如果達到限制，提前退出內循環
                    if len(edges) >= max_edges:
                        break
            
            # 定期清理記憶體
            if i % 1000 == 0:
                clean_memory()
        
        logger.info(f"建立了 {len(edges)} 條時間性邊")
        return edges
    
    @memory_usage_decorator
    def get_sample_batch(self, batch_size=None, memory_threshold=0.8):
        """
        獲取樣本批次，用於模擬動態圖更新
        
        參數:
            batch_size (int, optional): 批次大小，如果為 None 則使用配置中的批次大小
            memory_threshold (float): 記憶體使用率閾值，超過此值將減小批次大小
            
        返回:
            tuple: (批次資料, 批次標籤, 批次索引)
        """
        if self.features is None or self.target is None:
            self.preprocess()
        
        # 使用自適應批次大小
        if batch_size is None:
            initial_batch_size = self.batch_size
            batch_size = adaptive_batch_size(
                initial_batch_size=initial_batch_size,
                memory_threshold=memory_threshold,
                min_batch_size=16
            )
            
            if batch_size < initial_batch_size:
                logger.info(f"使用自適應批次大小: {batch_size} (原始: {initial_batch_size})")
        
        total_samples = len(self.target)
        if batch_size > total_samples:
            batch_size = total_samples
        
        # 使用 track_memory_usage 上下文管理器追蹤記憶體使用
        with track_memory_usage("獲取樣本批次"):
            # 隨機抽樣
            indices = np.random.choice(total_samples, batch_size, replace=False)
            
            # 獲取批次資料
            if self.use_memory_mapping:
                batch_features = self.features[indices].copy()  # 複製以避免修改原始數據
            else:
                batch_features = self.features[indices]
            
            batch_labels = self.target.iloc[indices].values if isinstance(self.target, pd.Series) else self.target[indices]
        
        return batch_features, batch_labels, indices
    
    def check_for_memory_leaks(self, iterations=3):
        """
        檢測記憶體洩漏
        
        通過多次執行數據加載和預處理操作，檢測是否存在記憶體洩漏。
        
        參數:
            iterations (int): 執行迭代次數
            
        返回:
            dict: 記憶體洩漏檢測結果
        """
        logger.info("開始檢測記憶體洩漏...")
        
        # 定義測試函數
        def test_data_operations():
            # 加載小批次數據
            if self.df is not None:
                sample_df = self.df.sample(min(1000, len(self.df)))
                # 執行一些典型操作
                sample_df = optimize_dataframe_memory(sample_df)
                # 獲取樣本批次
                if self.features is not None and self.target is not None:
                    self.get_sample_batch(batch_size=100)
            clean_memory()
        
        # 使用記憶體洩漏檢測工具
        result = detect_memory_leaks(iterations=iterations, func=test_data_operations)
        
        # 輸出結果
        if result['has_leak']:
            logger.warning("檢測到可能的記憶體洩漏，請檢查代碼中的資源釋放")
        else:
            logger.info("未檢測到明顯的記憶體洩漏")
        
        return result

# 測試資料載入器
if __name__ == "__main__":
    import yaml
    
    # 載入配置
    with open("memory_optimized_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # 初始化資料載入器
    loader = MemoryOptimizedDataLoader(config)
    
    # 載入和預處理資料
    X, y = loader.preprocess()
    print(f"處理後的特徵形狀: {X.shape}")
    print(f"標籤形狀: {y.shape}")
    
    # 拆分資料集
    X_train, X_test, y_train, y_test = loader.split_data()
    print(f"訓練集: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"測試集: X_test {X_test.shape}, y_test {y_test.shape}")
    
    # 獲取攻擊類型
    attack_types = loader.get_attack_types()
    print(f"攻擊類型: {attack_types}")
    
    # 獲取時間性邊
    edges = loader.get_temporal_edges(max_edges=1000)
    print(f"前 5 條時間性邊: {edges[:5]}")
    
    # 獲取樣本批次
    batch_features, batch_labels, batch_indices = loader.get_sample_batch(batch_size=100)
    print(f"批次特徵形狀: {batch_features.shape}")
    print(f"批次標籤形狀: {batch_labels.shape}")
    print(f"批次索引數量: {len(batch_indices)}")
