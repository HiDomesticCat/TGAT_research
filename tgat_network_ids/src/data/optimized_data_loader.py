#!/usr/bin/env python
# coding: utf-8 -*-

"""
高效記憶體優化版資料載入與預處理模組

根據建議進行升級，重點優化：
1. 增強的分塊處理機制
2. 更積極的資料類型轉換
3. 稀疏表示的統一使用
4. 記憶體映射的進階應用
5. 整合Polars作為Pandas替代方案
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
import shutil

# 導入記憶體優化工具
from ..utils.memory_utils import (
    memory_mapped_array, load_memory_mapped_array, save_dataframe_chunked,
    load_dataframe_chunked, optimize_dataframe_memory, clean_memory,
    memory_usage_decorator, track_memory_usage, print_memory_usage,
    get_memory_usage, print_optimization_suggestions, adaptive_batch_size,
    detect_memory_leaks, limit_gpu_memory
)

# 嘗試導入Polars作為高效率DataFrame替代
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedMemoryOptimizedDataLoader:
    """增強版記憶體優化資料載入與預處理類別"""

    def __init__(self, config):
        """
        初始化資料載入器

        參數:
            config (dict): 配置字典
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

        # 新增的優化設置
        self.use_polars = data_config.get('use_polars', POLARS_AVAILABLE)
        self.use_thread_pool = data_config.get('use_thread_pool', True)
        self.thread_pool_size = data_config.get('thread_pool_size', 4)
        self.use_pyarrow = data_config.get('use_pyarrow', True)
        self.aggressive_dtypes = data_config.get('aggressive_dtypes', True)
        self.aggressive_gc = data_config.get('aggressive_gc', False)
        self.chunk_row_limit = data_config.get('chunk_row_limit', 100000)

        # 資料採樣相關設置
        self.use_sampling = data_config.get('use_sampling', False)
        self.sampling_strategy = data_config.get('sampling_strategy', 'stratified')
        self.sampling_ratio = data_config.get('sampling_ratio', 0.1)
        self.min_samples_per_class = data_config.get('min_samples_per_class', 1000)

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

        # 記錄預處理時間戳記
        self.preprocess_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(f"初始化增強版記憶體優化資料載入器: 資料路徑={self.data_path}")
        logger.info(f"使用Polars={self.use_polars} 線程池={self.use_thread_pool}(大小={self.thread_pool_size})")
        logger.info(f"積極的數據類型轉換={self.aggressive_dtypes} 積極的垃圾回收={self.aggressive_gc}")

    @memory_usage_decorator
    def load_data(self):
        """載入資料集 - 使用進階優化"""
        logger.info(f"載入資料集從: {self.data_path}")

        # 檢查預處理資料
        if self.save_preprocessed and self._check_preprocessed_exists():
            logger.info("發現預處理資料，直接加載")
            success = self._load_preprocessed_data()
            if success:
                logger.info("預處理資料加載成功")
                return self.df
            else:
                logger.warning("預處理資料加載失敗，將重新處理原始資料")

        # 判斷是否為目錄或單一文件
        if os.path.isdir(self.data_path):
            # 載入目錄中的所有CSV檔案
            all_files = [os.path.join(self.data_path, f)
                        for f in os.listdir(self.data_path)
                        if f.endswith('.csv')]

            if not all_files:
                raise ValueError("未找到有效的CSV檔案")

            # 使用Polars加載(如果可用)
            if self.use_polars and POLARS_AVAILABLE:
                self._load_with_polars(all_files)
            elif self.incremental_loading:
                # 增量式加載
                self._load_data_incrementally(all_files)
            else:
                # 一次性加載所有文件
                self._load_data_at_once(all_files)
        else:
            # 載入單一CSV檔案
            if self.use_polars and POLARS_AVAILABLE:
                self._load_single_file_with_polars(self.data_path)
            elif self.incremental_loading:
                self._load_single_file_incrementally(self.data_path)
            else:
                self.df = pd.read_csv(self.data_path, low_memory=False)
                logger.info(f"載入資料集形狀: {self.df.shape}")

        # 優化 DataFrame 記憶體使用 - 使用更積極的優化
        if self.df is not None:
            self.df = self._enhanced_optimize_dataframe_memory(self.df)

        return self.df

    def preprocess(self):
        """預處理資料 - 對數值和分類特徵進行標準化和編碼"""
        logger.info("開始預處理資料...")
        
        # 載入資料
        if self.df is None:
            self.load_data()
            
        if self.df is None or self.df.empty:
            raise ValueError("資料載入失敗，無法進行預處理")
            
        # 進行特徵和目標分離
        # 尋找常見的目標列名
        target_columns = ['label', 'class', 'target', 'attack_type', 'is_attack']
        target_col = None
        
        for col in target_columns:
            if col in self.df.columns:
                target_col = col
                break
                
        if target_col is None:
            # 假設最後一列是目標
            target_col = self.df.columns[-1]
            logger.warning(f"未找到明確的目標列，使用最後一列 '{target_col}' 作為目標")
        
        # 提取特徵和目標
        self.target = self.df[target_col]
        self.features = self.df.drop(columns=[target_col])
        
        # 儲存特徵名稱
        self.feature_names = list(self.features.columns)
        
        # 標準化數值特徵
        num_features = self.features.select_dtypes(include=['int', 'float'])
        if not num_features.empty:
            # 創建副本以避免 SettingWithCopyWarning
            self.features = self.features.copy()
            
            # 標準化數值特徵
            num_cols = num_features.columns
            self.features[num_cols] = self.scaler.fit_transform(num_features)
            
        # 編碼分類目標
        if not pd.api.types.is_numeric_dtype(self.target):
            self.target = pd.Series(self.label_encoder.fit_transform(self.target))
            
        logger.info(f"預處理完成。特徵形狀: {self.features.shape}, 目標形狀: {self.target.shape}")
        
        # 如果啟用保存預處理資料
        if self.save_preprocessed:
            self._save_preprocessed_data()
            
        return self.features, self.target
        
    def split_data(self):
        """拆分資料為訓練集和測試集"""
        logger.info(f"拆分資料為訓練集和測試集 (測試集比例: {self.test_size})")
        
        # 確保特徵和目標已準備好
        if self.features is None or self.target is None:
            self.preprocess()
            
        # 使用 sklearn 進行拆分
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.target  # 分層抽樣以保持類別分佈
        )
        
        # 儲存拆分結果
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        logger.info(f"資料拆分完成。訓練集: {X_train.shape}, 測試集: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
        
    def get_attack_types(self):
        """獲取攻擊類型映射"""
        # 確保目標已編碼
        if self.target is None:
            self.preprocess()
            
        # 如果目標已經被編碼
        if hasattr(self.label_encoder, 'classes_'):
            attack_types = {i: name for i, name in enumerate(self.label_encoder.classes_)}
        else:
            # 如果目標沒有被編碼或者是數值型
            unique_labels = sorted(self.target.unique())
            attack_types = {label: f"Type_{label}" for label in unique_labels}
            
        return attack_types
        
    def get_sample_batch(self, batch_size=None):
        """獲取一個批次的樣本，用於測試或即時檢測"""
        if batch_size is None:
            batch_size = self.batch_size
            
        # 確保測試集已準備好
        if self.X_test is None or self.y_test is None:
            self.split_data()
            
        # 從測試集隨機採樣
        random_indices = np.random.choice(len(self.X_test), size=min(batch_size, len(self.X_test)), replace=False)
        
        batch_features = self.X_test.iloc[random_indices]
        batch_labels = self.y_test.iloc[random_indices]
        
        return batch_features.values, batch_labels.values, random_indices
        
    def get_temporal_edges(self, max_edges=None):
        """獲取時間性邊 - 根據連續封包之間的時間關係生成"""
        logger.info("生成時間性邊...")
        
        # 確保特徵已準備好
        if self.features is None:
            self.preprocess()
            
        # 檢查是否有時間戳列
        time_cols = [col for col in self.feature_names if 'time' in col.lower()]
        
        if not time_cols:
            logger.warning("未找到時間戳列，使用隨機生成的時間性邊")
            # 隨機生成邊
            edges = []
            num_nodes = len(self.features)
            
            # 限制最大邊數
            if max_edges is None:
                max_edges = min(500000, num_nodes * 10)
                
            # 生成隨機邊
            for _ in range(max_edges):
                src = np.random.randint(0, num_nodes)
                dst = np.random.randint(0, num_nodes)
                
                # 避免自環
                while dst == src:
                    dst = np.random.randint(0, num_nodes)
                    
                # 生成隨機時間戳和特徵
                timestamp = np.random.random() * 100  # 隨機時間戳
                edge_feat = [np.random.random() for _ in range(3)]  # 隨機3維邊特徵
                
                edges.append((src, dst, timestamp, edge_feat))
                
            logger.info(f"隨機生成了 {len(edges)} 條時間性邊")
            return edges
            
        # 使用時間戳生成邊
        logger.info(f"使用時間戳列 '{time_cols[0]}' 生成時間性邊")
        time_col = time_cols[0]
        
        # 排序特徵按時間戳
        sorted_indices = np.argsort(self.features[time_col].values)
        
        # 生成邊 - 連接時間上相鄰的節點
        edges = []
        
        for i in range(1, len(sorted_indices)):
            src = sorted_indices[i-1]
            dst = sorted_indices[i]
            
            # 獲取時間戳
            timestamp = self.features[time_col].values[dst]
            
            # 生成簡單的邊特徵
            edge_feat = [np.random.random() for _ in range(3)]
            
            edges.append((src, dst, timestamp, edge_feat))
            
            # 隨機添加額外連接以增加圖密度
            if np.random.random() < 0.3:  # 30%機率添加額外連接
                extra_dst = sorted_indices[max(0, i-2)]  # 連接到前面的節點
                edges.append((src, extra_dst, timestamp, edge_feat))
                
            # 限制最大邊數
            if max_edges is not None and len(edges) >= max_edges:
                break
                
        logger.info(f"基於時間戳生成了 {len(edges)} 條時間性邊")
        return edges
        
    def _check_preprocessed_exists(self):
        """檢查預處理資料是否存在"""
        feature_path = os.path.join(self.preprocessed_path, 'features.parquet')
        target_path = os.path.join(self.preprocessed_path, 'target.parquet')
        
        return os.path.exists(feature_path) and os.path.exists(target_path)
        
    def _save_preprocessed_data(self):
        """保存預處理後的資料"""
        # 確保目錄存在
        if not os.path.exists(self.preprocessed_path):
            os.makedirs(self.preprocessed_path)
            
        # 保存特徵和目標
        feature_path = os.path.join(self.preprocessed_path, 'features.parquet')
        target_path = os.path.join(self.preprocessed_path, 'target.parquet')
        
        self.features.to_parquet(feature_path, index=False)
        pd.DataFrame({'target': self.target}).to_parquet(target_path, index=False)
        
        # 保存編碼器和縮放器
        with open(os.path.join(self.preprocessed_path, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
            
        with open(os.path.join(self.preprocessed_path, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
            
        logger.info(f"預處理資料已保存至: {self.preprocessed_path}")
        
    def _load_preprocessed_data(self):
        """載入預處理後的資料"""
        try:
            feature_path = os.path.join(self.preprocessed_path, 'features.parquet')
            target_path = os.path.join(self.preprocessed_path, 'target.parquet')
            
            self.features = pd.read_parquet(feature_path)
            target_df = pd.read_parquet(target_path)
            self.target = target_df['target']
            
            # 儲存特徵名稱
            self.feature_names = list(self.features.columns)
            
            # 載入編碼器和縮放器
            with open(os.path.join(self.preprocessed_path, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
                
            with open(os.path.join(self.preprocessed_path, 'label_encoder.pkl'), 'rb') as f:
                self.label_encoder = pickle.load(f)
                
            logger.info(f"預處理資料載入成功: 特徵形狀={self.features.shape}, 目標形狀={self.target.shape}")
            return True
        except Exception as e:
            logger.error(f"預處理資料載入失敗: {str(e)}")
            return False
    
    def _enhanced_optimize_dataframe_memory(self, df):
        """增強版DataFrame記憶體優化 - 更積極的類型轉換策略"""
        start_mem = df.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"優化前 DataFrame 記憶體使用: {start_mem:.2f} MB")
        
        # 複製 DataFrame 以避免修改原始數據
        df_optimized = df.copy()
        
        # 處理整數型列
        integer_columns = df_optimized.select_dtypes(include=['int']).columns
        for col in integer_columns:
            # 獲取列的最小值和最大值
            col_min = df_optimized[col].min()
            col_max = df_optimized[col].max()
            
            # 根據數據範圍選擇更緊湊的數據類型
            if col_min >= 0:
                if col_max < 2**8:
                    df_optimized[col] = df_optimized[col].astype(np.uint8)
                elif col_max < 2**16:
                    df_optimized[col] = df_optimized[col].astype(np.uint16)
                elif col_max < 2**32:
                    df_optimized[col] = df_optimized[col].astype(np.uint32)
                else:
                    df_optimized[col] = df_optimized[col].astype(np.uint64)
            else:
                if col_min > -2**7 and col_max < 2**7:
                    df_optimized[col] = df_optimized[col].astype(np.int8)
                elif col_min > -2**15 and col_max < 2**15:
                    df_optimized[col] = df_optimized[col].astype(np.int16)
                elif col_min > -2**31 and col_max < 2**31:
                    df_optimized[col] = df_optimized[col].astype(np.int32)
                else:
                    df_optimized[col] = df_optimized[col].astype(np.int64)
                
        # 處理浮點型列 - 積極使用float16/float32而非float64
        float_columns = df_optimized.select_dtypes(include=['float']).columns
        for col in float_columns:
            # 檢查值範圍來決定是否可以使用float16
            if self.aggressive_dtypes:
                col_min = df_optimized[col].min()
                col_max = df_optimized[col].max()
                # float16範圍約為±65504
                if col_min > -65000 and col_max < 65000:
                    # 對於小範圍值，使用float16
                    df_optimized[col] = df_optimized[col].astype(np.float16)
                else:
                    # 否則使用float32
                    df_optimized[col] = df_optimized[col].astype(np.float32)
            else:
                # 保守策略：一律使用float32
                df_optimized[col] = df_optimized[col].astype(np.float32)
        
        # 處理對象型列和分類型列
        object_columns = df_optimized.select_dtypes(include=['object']).columns
        for col in object_columns:
            # 計算唯一值比例
            unique_ratio = df_optimized[col].nunique() / len(df_optimized)
            
            # 唯一值較少的列使用分類類型 (調整閾值以更積極地使用category)
            if unique_ratio < 0.7:  # 原為0.5，現在更積極使用category
                df_optimized[col] = df_optimized[col].astype('category')
        
        # 計算優化後的記憶體使用
        end_mem = df_optimized.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"優化後 DataFrame 記憶體使用: {end_mem:.2f} MB")
        logger.info(f"記憶體使用減少: {(start_mem - end_mem) / start_mem * 100:.2f}%")
        
        # 積極的垃圾回收
        if self.aggressive_gc:
            gc.collect()
        
        return df_optimized

    def _process_parquet_with_polars(self, parquet_files):
        """使用Polars高效處理Parquet文件"""
        logger.info(f"使用Polars處理 {len(parquet_files)} 個Parquet文件")
        
        if not POLARS_AVAILABLE:
            logger.warning("Polars不可用，回退到PyArrow處理")
            return False
            
        try:
            # 使用Polars的LazyFrame高效讀取Parquet
            dfs = []
            
            for i, parquet_file in enumerate(tqdm(parquet_files, desc="Polars讀取Parquet")):
                try:
                    # 使用scan_parquet比直接read_parquet更節省記憶體
                    lf = pl.scan_parquet(parquet_file)
                    
                    # 應用採樣（如果需要）
                    if self.use_sampling and self.sampling_strategy == 'random':
                        lf = lf.sample(fraction=self.sampling_ratio)
                        
                    # 收集DataFrame - 使用批次以節省記憶體
                    df = lf.collect()
                    dfs.append(df)
                    
                    logger.info(f"Polars成功讀取Parquet: {parquet_file}, 形狀: {df.shape}")
                    
                    # 定期釋放記憶體
                    if (i + 1) % 3 == 0:
                        # 合併已處理的DataFrame
                        if len(dfs) > 1:
                            combined = pl.concat(dfs)
                            dfs = [combined]
                            
                        # 強制垃圾回收
                        gc.collect()
                except Exception as e:
                    logger.error(f"Polars讀取Parquet失敗 {parquet_file}: {str(e)}")
                    
            # 合併所有DataFrame
            if dfs:
                logger.info(f"合併 {len(dfs)} 個Polars DataFrame")
                
                # 使用concat而非較慢的vstack
                if len(dfs) > 1:
                    combined_df = pl.concat(dfs)
                else:
                    combined_df = dfs[0]
                
                # 對於分層採樣，在這裡處理
                if self.use_sampling and self.sampling_strategy == 'stratified' and 'label' in combined_df.columns:
                    # 對每個類別單獨採樣
                    sampled_dfs = []
                    
                    # 獲取唯一標籤
                    labels = combined_df.select('label').unique().to_series()
                    
                    for lbl in labels:
                        # 過濾單一類別的資料
                        class_df = combined_df.filter(pl.col('label') == lbl)
                        class_size = class_df.shape[0]
                        
                        # 計算採樣數量
                        sample_size = max(
                            int(class_size * self.sampling_ratio),
                            min(self.min_samples_per_class, class_size)
                        )
                        
                        # 採樣並添加到結果中
                        if sample_size < class_size:
                            sampled_class = class_df.sample(n=sample_size)
                            sampled_dfs.append(sampled_class)
                        else:
                            sampled_dfs.append(class_df)
                            
                    # 合併所有採樣後的類別
                    if sampled_dfs:
                        combined_df = pl.concat(sampled_dfs)
                        logger.info(f"分層採樣後大小: {combined_df.shape[0]}")
                
                # 轉換為Pandas DataFrame以便與其他代碼兼容
                self.df = combined_df.to_pandas()
                logger.info(f"Polars處理完成，DataFrame形狀: {self.df.shape}")
                
                # 釋放記憶體
                del dfs, combined_df
                gc.collect()
                
                return True
            else:
                logger.warning("Polars處理未產生任何有效資料")
                return False
                
        except Exception as e:
            logger.error(f"Polars處理Parquet文件過程中發生錯誤: {str(e)}")
            return False
        
    def _load_with_polars(self, file_list):
        """使用Polars高效載入多個文件"""
        logger.info(f"使用Polars載入 {len(file_list)} 個文件")
        
        try:
            # 使用Polars的lazy執行模式逐個載入文件
            dfs = []
            for i, file_path in enumerate(tqdm(file_list, desc="使用Polars載入")):
                try:
                    # 使用LazyFrame進行高效讀取和處理
                    lf = pl.scan_csv(file_path)
                    
                    # 應用篩選(如果需要)
                    if self.use_sampling:
                        if self.sampling_strategy == 'random':
                            # 隨機採樣
                            lf = lf.sample(fraction=self.sampling_ratio)
                        # 注意：Polars的lazy API不直接支持分層採樣，需在收集後處理
                    
                    # 收集DataFrame
                    df = lf.collect()
                    dfs.append(df)
                    
                    logger.info(f"Polars成功載入: {file_path}, 形狀: {df.shape}")
                    
                    # 定期釋放記憶體
                    if (i + 1) % 3 == 0:
                        gc.collect()
                except Exception as e:
                    logger.error(f"Polars載入 {file_path} 失敗: {str(e)}")
            
            # 合併所有DataFrame
            if dfs:
                logger.info(f"合併 {len(dfs)} 個Polars DataFrame")
                combined_df = pl.concat(dfs)
                
                # 將Polars DataFrame轉換為Pandas DataFrame(保持與其他代碼兼容)
                self.df = combined_df.to_pandas()
                logger.info(f"合併完成，DataFrame形狀: {self.df.shape}")
                
                # 釋放記憶體
                del dfs, combined_df
                gc.collect()
            else:
                raise ValueError("所有文件均載入失敗")
                
        except Exception as e:
            logger.error(f"Polars載入過程中發生錯誤: {str(e)}")
            logger.info("回退到標準載入方法")
            
            if self.incremental_loading:
                self._load_data_incrementally(file_list)
            else:
                self._load_data_at_once(file_list)

    def _load_single_file_incrementally(self, file_path):
        """增量式加載單個CSV文件 - 使用分塊處理"""
        logger.info(f"增量式加載單個文件: {file_path}")
        
        # 檢查文件大小
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"文件大小: {file_size_mb:.2f} MB")
        
        # 對於大文件使用分塊處理
        if file_size_mb > 500:  # 大於500MB
            # 估計合理的分塊大小
            sample_size = 1000
            sample_df = pd.read_csv(file_path, nrows=sample_size)
            avg_row_size_bytes = sample_df.memory_usage(deep=True).sum() / len(sample_df)
            
            # 算出每個批次的行數 (目標每批次100MB)
            rows_per_chunk = int((100 * 1024 * 1024) / avg_row_size_bytes)
            logger.info(f"大文件分塊處理，每批次約 {rows_per_chunk} 行")
            
            # 分塊讀取
            dfs = []
            for i, chunk in enumerate(tqdm(pd.read_csv(file_path, chunksize=rows_per_chunk, low_memory=False), desc="分塊處理")):
                # 優化記憶體使用
                chunk = self._enhanced_optimize_dataframe_memory(chunk)
                dfs.append(chunk)
                
                # 定期合併以釋放記憶體
                if len(dfs) >= 5 or (i + 1) * rows_per_chunk >= file_size_mb * 1000000 / avg_row_size_bytes:
                    df_merged = pd.concat(dfs, ignore_index=True)
                    if not hasattr(self, 'df') or self.df is None:
                        self.df = df_merged
                    else:
                        self.df = pd.concat([self.df, df_merged], ignore_index=True)
                    
                    # 清理中間df
                    dfs = []
                    gc.collect()
            
            # 處理剩餘的塊
            if dfs:
                df_merged = pd.concat(dfs, ignore_index=True)
                if not hasattr(self, 'df') or self.df is None:
                    self.df = df_merged
                else:
                    self.df = pd.concat([self.df, df_merged], ignore_index=True)
        else:
            # 小文件直接讀取
            self.df = pd.read_csv(file_path, low_memory=False)
            self.df = self._enhanced_optimize_dataframe_memory(self.df)
            
        logger.info(f"載入完成，DataFrame形狀: {self.df.shape}")
        
    def _load_data_at_once(self, file_list):
        """一次性加載所有CSV文件"""
        logger.info(f"一次性加載 {len(file_list)} 個文件")
        
        # 讀取所有文件並合併
        dfs = []
        for file_path in tqdm(file_list, desc="載入文件"):
            try:
                df = pd.read_csv(file_path, low_memory=False)
                df = optimize_dataframe_memory(df)  # 使用全局函數避免self引用
                dfs.append(df)
            except Exception as e:
                logger.error(f"讀取文件失敗 {file_path}: {str(e)}")
                
        # 合併所有DataFrame
        if dfs:
            self.df = pd.concat(dfs, ignore_index=True)
            logger.info(f"合併完成，DataFrame形狀: {self.df.shape}")
            
            # 釋放記憶體
            del dfs
            gc.collect()
        else:
            raise ValueError("所有文件均載入失敗")
    
    def _load_single_file_with_polars(self, file_path):
        """使用Polars高效載入單個文件"""
        logger.info(f"使用Polars載入單個文件: {file_path}")
        
        try:
            # 檢查文件大小
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            logger.info(f"文件大小: {file_size_mb:.2f} MB")
            
            # 對於大文件使用LazyFrame和分塊處理
            if file_size_mb > 1000:  # 大於1GB的文件
                # 使用LazyFrame提供更好的記憶體效率
                logger.info("文件較大，使用LazyFrame分塊處理")
                lf = pl.scan_csv(file_path)
                
                # 如果啟用採樣，在LazyFrame層面應用
                if self.use_sampling:
                    lf = lf.sample(fraction=self.sampling_ratio)
                
                # 適用於極大文件的分塊收集策略
                chunks = []
                
                # 估計總行數
                total_rows = 0
                with open(file_path, 'r') as f:
                    for _ in range(100):  # 讀取前100行來估計
                        if f.readline():
                            total_rows += 1
                
                # 根據文件大小估計總行數
                estimated_rows = int(total_rows * (file_size_mb / 0.01))  # 假設前100行占約0.01MB
                logger.info(f"估計總行數: ~{estimated_rows}")
                
                # 計算每個塊的大小
                chunk_size = min(self.chunk_row_limit, max(1000, int(estimated_rows / 10)))
                
                # 分塊收集
                for i, chunk_df in enumerate(lf.collect(streaming=True)):
                    chunks.append(chunk_df)
                    logger.info(f"已處理塊 {i+1}，大小: {len(chunk_df)}")
                    
                    # 如果積累了足夠多的塊，合併它們
                    if len(chunks) >= 5:
                        intermediate_df = pl.concat(chunks)
                        chunks = [intermediate_df]
                        gc.collect()
                
                # 合併所有塊
                if chunks:
                    polars_df = pl.concat(chunks)
                    self.df = polars_df.to_pandas()
                    logger.info(f"大文件載入完成，形狀: {self.df.shape}")
                else:
                    raise ValueError("分塊收集失敗")
            else:
                # 對於較小文件直接使用collect
                df = pl.read_csv(file_path).collect()
                self.df = df.to_pandas()
                logger.info(f"文件載入完成，形狀: {self.df.shape}")
            
            # 保存為parquet以加速未來載入
            if self.save_preprocessed:
                parquet_path = file_path.replace('.csv', '.parquet')
                self.df.to_parquet(parquet_path, index=False)
                logger.info(f"已保存為Parquet格式: {parquet_path}")
                
        except Exception as e:
            logger.error(f"Polars載入失敗: {str(e)}")
            logger.info("回退到標準載入方法")
            
            if self.incremental_loading:
                self._load_single_file_incrementally(file_path)
            else:
                self.df = pd.read_csv(file_path, low_memory=False)
                logger.info(f"標準方法載入完成，形狀: {self.df.shape}")

    def _load_data_incrementally(self, file_list):
        """增強版增量式加載多個文件 - 使用更高效的串流處理"""
        logger.info(f"增量式加載 {len(file_list)} 個文件 (增強串流處理)")

        # 轉換所有文件為Parquet格式（如果需要）
        self._ensure_parquet_files(file_list)

        # 使用Parquet格式直接載入和處理資料
        self._process_parquet_files(file_list)

    def _convert_to_parquet(self, file_path):
        """將CSV文件轉換為Parquet格式 - 分離為獨立函數以支持多進程"""
        parquet_path = file_path.replace('.csv', '.parquet')
        try:
            # 優先使用PyArrow直接轉換（更快且記憶體效率更高）
            if self.use_pyarrow:
                try:
                    import pyarrow as pa
                    import pyarrow.csv as pc
                    import pyarrow.parquet as pq

                    # 使用進階優化設置
                    parse_options = pc.ParseOptions(delimiter=',')
                    convert_options = pc.ConvertOptions(
                        strings_to_categorical=True,
                        include_columns=None,
                    )
                    read_options = pc.ReadOptions(
                        block_size=32 * 1024 * 1024,  # 增加到32MB塊大小提高效率
                        use_threads=True
                    )

                    # 直接轉換並寫入Parquet
                    table = pc.read_csv(
                        file_path,
                        parse_options=parse_options,
                        convert_options=convert_options,
                        read_options=read_options
                    )

                    # 使用更佳的壓縮選項
                    pq.write_table(
                        table, 
                        parquet_path, 
                        compression='snappy',
                        use_dictionary=True,
                        write_statistics=True
                    )
                    return f"成功轉換: {file_path} -> {parquet_path}"

                except ImportError:
                    logger.warning("PyArrow不可用，使用pandas備用方法")

            # 使用pandas轉換（較慢但兼容性更好）
            # 對大文件使用分塊處理
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB

            if file_size > 1000:  # 大於1GB的文件使用分塊處理
                # 估計合理的分塊大小
                sample_size = 1000
                sample_df = pd.read_csv(file_path, nrows=sample_size)
                avg_row_size_bytes = sample_df.memory_usage(deep=True).sum() / len(sample_df)
                rows_per_chunk = int((300 * 1024 * 1024) / avg_row_size_bytes)

                # 分塊處理大文件
                for i, chunk in enumerate(pd.read_csv(file_path, chunksize=rows_per_chunk, low_memory=False)):
                    # 優化記憶體使用
                    chunk = optimize_dataframe_memory(chunk)  # 使用全局函數，避免self引用

                    if i == 0:
                        # 第一個塊，創建Parquet文件
                        chunk.to_parquet(parquet_path, engine='pyarrow', compression='snappy', index=False)
                    else:
                        # 後續塊，追加到Parquet文件
                        chunk.to_parquet(parquet_path, engine='pyarrow', compression='snappy',
                                       index=False, append=True)

                    # 釋放記憶體
                    del chunk
                    gc.collect()

                return f"使用分塊處理成功轉換: {file_path} -> {parquet_path}"
            else:
                # 小文件一次性處理
                try:
                    # 嘗試使用Polars (如果可用)
                    if 'pl' in globals() and self.use_polars:
                        df = pl.read_csv(file_path).collect()
                        df.write_parquet(parquet_path, compression="snappy")
                    else:
                        # 使用pandas
                        df = pd.read_csv(file_path, low_memory=False)
                        df = optimize_dataframe_memory(df)  # 使用全局函數，避免self引用
                        df.to_parquet(parquet_path, compression='snappy', index=False)
                except Exception as e:
                    logger.error(f"轉換時出錯: {str(e)}, 嘗試備用方法")
                    df = pd.read_csv(file_path, low_memory=False)
                    df.to_parquet(parquet_path, compression='snappy', index=False)
                    
                return f"成功轉換: {file_path} -> {parquet_path}"

        except Exception as e:
            return f"轉換失敗 {file_path}: {str(e)}"

    def _ensure_parquet_files(self, file_list):
        """增強版確保CSV文件轉換為Parquet格式 - 使用順序處理避免多進程問題"""
        # 獲取需要轉換的文件列表
        conversion_needed = []
        for csv_file in file_list:
            parquet_file = csv_file.replace('.csv', '.parquet')
            if not os.path.exists(parquet_file) and self.save_preprocessed:
                conversion_needed.append(csv_file)

        if not conversion_needed:
            logger.info("所有文件已有Parquet格式，無需轉換")
            return

        # 降級到順序處理以避免多進程問題
        logger.info(f"順序轉換 {len(conversion_needed)} 個文件為Parquet格式")
        
        for i, file_path in enumerate(tqdm(conversion_needed, desc="轉換CSV至Parquet")):
            try:
                result = self._convert_to_parquet(file_path)
                logger.info(f"[{i+1}/{len(conversion_needed)}] {result}")
            except Exception as e:
                logger.error(f"轉換過程中發生錯誤: {str(e)}")

    def _process_parquet_files(self, file_list):
        """增強版串流處理Parquet文件 - 使用更佳的批次處理和記憶體管理"""
        # 獲取所有文件的Parquet路徑
        parquet_files = [f.replace('.csv', '.parquet') if f.endswith('.csv') else f for f in file_list]
        available_parquet = [f for f in parquet_files if os.path.exists(f)]

        if not available_parquet:
            raise ValueError("無法找到可用的Parquet文件")

        logger.info(f"開始串流處理{len(available_parquet)}個Parquet文件")

        # 使用串流處理避免記憶體溢出
        try:
            # 優先使用Polars進行高效率處理
            if POLARS_AVAILABLE and self.use_polars:
                self._process_parquet_with_polars(available_parquet)
                return
                
            # 備用：使用PyArrow直接讀取
            try:
                import pyarrow.parquet as pq
                
                # 估計總行數
                total_rows = 0
                for pq_file in available_parquet:
                    metadata = pq.read_metadata(pq_file)
                    total_rows += metadata.num_rows

                logger.info(f"估計總行數: {total_rows}")

                # 初始化DataFrame
                self.df = pd.DataFrame()

                # 優化的數據類型映射
                dtype_mapping = {
                    'string': 'category',
                    'double': 'float32',
                    'int64': 'int32'
                }

                # 分塊處理每個文件
                processed_rows = 0

                for file_idx, pq_file in enumerate(available_parquet):
                    logger.info(f"處理Parquet文件 {file_idx+1}/{len(available_parquet)}: {pq_file}")

                    # 檢查是否為超大文件（大於2GB）
                    file_size_mb = os.path.getsize(pq_file) / (1024 * 1024)

                    if file_size_mb > 2048:  # 大於2GB的文件
                        # 使用分批讀取避免記憶體問題
                        parquet_file = pq.ParquetFile(pq_file)
                        num_row_groups = parquet_file.num_row_groups

                        logger.info(f"大型Parquet文件，分批讀取 {num_row_groups} 個行組")

                        # 收集所有批次的DataFrames，然後一次性合併
                        batch_dfs = []
                        
                        # 每次讀取一個行組
                        for i in range(num_row_groups):
                            # 讀取一個行組並轉換為pandas DataFrame
                            batch_df = parquet_file.read_row_group(i).to_pandas()
                            batch_df = self._enhanced_optimize_dataframe_memory(batch_df)

                            # 收集批次DataFrame
                            batch_dfs.append(batch_df)
                            
                            # 定期合併並釋放記憶體
                            if len(batch_dfs) >= 5 or i == num_row_groups - 1:
                                combined_batch = pd.concat(batch_dfs, ignore_index=True)
                                
                                if self.df.empty:
                                    self.df = combined_batch
                                else:
                                    self.df = pd.concat([self.df, combined_batch], ignore_index=True)
                                    
                                # 清理批次
                                batch_dfs = []
                                gc.collect()

                            processed_rows += len(batch_df)
                    else:
                        # 較小文件一次性讀取
                        try:
                            batch_df = pd.read_parquet(pq_file)
                            batch_df = self._enhanced_optimize_dataframe_memory(batch_df)
                            
                            if self.df.empty:
                                self.df = batch_df
                            else:
                                self.df = pd.concat([self.df, batch_df], ignore_index=True)
                                
                            processed_rows += len(batch_df)
                            
                            # 定期進行垃圾收集
                            if self.aggressive_gc or file_idx % 3 == 0:
                                gc.collect()
                                
                        except Exception as e:
                            logger.error(f"讀取Parquet文件失敗 {pq_file}: {str(e)}")
                            
                logger.info(f"完成Parquet文件處理: 總計 {processed_rows} 行")
                
                # 如果啟用採樣但未在Polars層面實現，在這裡進行採樣
                if self.use_sampling and self.df is not None and not self.df.empty:
                    if self.sampling_strategy == 'random':
                        # 隨機採樣
                        sample_size = max(int(len(self.df) * self.sampling_ratio), 1000)
                        self.df = self.df.sample(n=sample_size, random_state=self.random_state)
                        logger.info(f"隨機採樣後大小: {len(self.df)}")
                    elif self.sampling_strategy == 'stratified':
                        # 分層採樣 (如果有標籤列)
                        if 'label' in self.df.columns:
                            from sklearn.model_selection import train_test_split
                            
                            # 獲取每個類別的最小樣本數
                            classes = self.df['label'].unique()
                            sampled_indices = []
                            
                            for cls in classes:
                                cls_indices = self.df[self.df['label'] == cls].index
                                cls_sample_size = max(
                                    int(len(cls_indices) * self.sampling_ratio),
                                    min(self.min_samples_per_class, len(cls_indices))
                                )
                                sampled_cls_indices = np.random.choice(
                                    cls_indices, size=cls_sample_size, replace=False
                                )
                                sampled_indices.extend(sampled_cls_indices)
                                
                            self.df = self.df.loc[sampled_indices]
                            logger.info(f"分層採樣後大小: {len(self.df)}")
                        else:
                            logger.warning("無法進行分層採樣: 未找到'label'列")
            except ImportError as e:
                logger.error(f"PyArrow導入失敗: {str(e)}")
                logger.info("使用pandas備用方法讀取Parquet")
                
                # 備用：使用pandas直接讀取
                self.df = pd.DataFrame()
                for file_idx, pq_file in enumerate(tqdm(available_parquet, desc="讀取Parquet文件")):
                    try:
                        batch_df = pd.read_parquet(pq_file)
                        
                        if self.df.empty:
                            self.df = batch_df
                        else:
                            self.df = pd.concat([self.df, batch_df], ignore_index=True)
                            
                        # 定期進行垃圾收集
                        if (file_idx + 1) % 3 == 0:
                            gc.collect()
                            
                    except Exception as e:
                        logger.error(f"讀取Parquet文件失敗 {pq_file}: {str(e)}")
                        
                # 如果啟用採樣，在最終數據上執行
                if self.use_sampling and self.df is not None and not self.df.empty:
                    sample_size = max(int(len(self.df) * self.sampling_ratio), 1000)
                    self.df = self.df.sample(n=sample_size, random_state=self.random_state)
                    logger.info(f"採樣後大小: {len(self.df)}")
            except Exception as e:
                logger.error(f"處理Parquet文件時發生錯誤: {str(e)}")
                # 重新引發異常但提供更明確的訊息
                raise ValueError(f"Parquet處理失敗: {str(e)}") from e
        except Exception as e:
            logger.error(f"Parquet文件處理失敗: {str(e)}")
            logger.info("使用pandas備用方法")
            
            # 備用：使用pandas直接加載CSV
            self.df = pd.DataFrame()
            csv_files = [f for f in file_list if f.endswith('.csv')]
            
            if not csv_files:
                raise ValueError("找不到可用的CSV文件，且Parquet處理失敗")
                
            for file_idx, csv_file in enumerate(tqdm(csv_files, desc="備用CSV載入")):
                try:
                    # 分塊加載大文件
                    file_size = os.path.getsize(csv_file) / (1024 * 1024)  # MB
                    
                    if file_size > 1000:  # 大於1GB
                        chunk_dfs = []
                        for chunk in pd.read_csv(csv_file, chunksize=100000, low_memory=False):
                            chunk = self._enhanced_optimize_dataframe_memory(chunk)
                            chunk_dfs.append(chunk)
                            
                            if len(chunk_dfs) >= 5:
                                merged = pd.concat(chunk_dfs, ignore_index=True)
                                if self.df.empty:
                                    self.df = merged
                                else:
                                    self.df = pd.concat([self.df, merged], ignore_index=True)
                                chunk_dfs = []
                                gc.collect()
                                
                        # 處理剩餘的塊
                        if chunk_dfs:
                            merged = pd.concat(chunk_dfs, ignore_index=True)
                            if self.df.empty:
                                self.df = merged
                            else:
                                self.df = pd.concat([self.df, merged], ignore_index=True)
                    else:
                        # 小文件一次加載
                        chunk = pd.read_csv(csv_file, low_memory=False)
                        chunk = self._enhanced_optimize_dataframe_memory(chunk)
                        
                        if self.df.empty:
                            self.df = chunk
                        else:
                            self.df = pd.concat([self.df, chunk], ignore_index=True)
                            
                    # 定期清理記憶體
                    if (file_idx + 1) % 3 == 0:
                        gc.collect()
                        
                except Exception as e:
                    logger.error(f"備用CSV加載失敗 {csv_file}: {str(e)}")
