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
                self._load_with_incremental_processing(all_files)
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
            
            # 檢測並處理無限值和極端值
            num_features_cleaned = num_features.copy()
            
            # 替換無限值和NaN為該列的中位數
            for col in num_features_cleaned.columns:
                # 獲取有限值的掩碼
                mask_inf = np.isinf(num_features_cleaned[col])
                mask_nan = np.isnan(num_features_cleaned[col])
                mask_large = np.abs(num_features_cleaned[col]) > 1e30  # 過大的值
                
                if mask_inf.any() or mask_nan.any() or mask_large.any():
                    # 組合所有問題值的掩碼
                    mask_invalid = mask_inf | mask_nan | mask_large
                    
                    # 計算有效值的中位數
                    valid_values = num_features_cleaned.loc[~mask_invalid, col]
                    if len(valid_values) > 0:
                        median_value = valid_values.median()
                    else:
                        median_value = 0  # 如果沒有有效值，使用0
                    
                    # 替換無效值
                    num_features_cleaned.loc[mask_invalid, col] = median_value
                    
                    logger.warning(f"列 '{col}' 中發現 {mask_invalid.sum()} 個無效值 (inf/NaN/極大值)，已替換為中位數 {median_value}")
                    
                # 將值限制在合理範圍內
                upper_limit = num_features_cleaned[col].quantile(0.99) * 1.5
                lower_limit = num_features_cleaned[col].quantile(0.01) * 0.5
                
                # 處理極端值的邊界情況
                if upper_limit == lower_limit:
                    # 如果上下限相同，擴大範圍
                    if upper_limit == 0:
                        upper_limit = 1
                        lower_limit = -1
                    else:
                        upper_limit = upper_limit * 2
                        lower_limit = lower_limit * 0.5
                
                # 截斷極端值
                num_features_cleaned[col] = num_features_cleaned[col].clip(lower_limit, upper_limit)
            
            # 標準化數值特徵
            num_cols = num_features_cleaned.columns
            try:
                # 嘗試標準化
                scaled_features = self.scaler.fit_transform(num_features_cleaned)
                self.features[num_cols] = scaled_features
            except Exception as e:
                logger.error(f"標準化特徵時發生錯誤: {str(e)}")
                # 回退策略：如果標準化失敗，使用最小最大縮放
                for i, col in enumerate(num_cols):
                    col_min = num_features_cleaned[col].min()
                    col_max = num_features_cleaned[col].max()
                    # 避免除以零
                    if col_max > col_min:
                        self.features[col] = (num_features_cleaned[col] - col_min) / (col_max - col_min)
                    else:
                        self.features[col] = 0  # 如果所有值相同，設為0
                
                logger.warning("使用最小最大縮放替代標準化")
            
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
        feature_csv_path = os.path.join(self.preprocessed_path, 'features.csv')
        target_path = os.path.join(self.preprocessed_path, 'target.parquet')
        target_csv_path = os.path.join(self.preprocessed_path, 'target.csv')
        
        # 檢查 parquet 或 csv 格式檔案是否存在
        features_exist = os.path.exists(feature_path) or os.path.exists(feature_csv_path)
        target_exist = os.path.exists(target_path) or os.path.exists(target_csv_path)
        
        return features_exist and target_exist
        
    def _save_preprocessed_data(self):
        """保存預處理後的資料，使用多檔案分片儲存策略避免單一大型檔案"""
        # 確保目錄存在
        if not os.path.exists(self.preprocessed_path):
            os.makedirs(self.preprocessed_path)
            
        # 建立分片目錄
        features_dir = os.path.join(self.preprocessed_path, 'features_chunks')
        if not os.path.exists(features_dir):
            os.makedirs(features_dir)
            
        # 創建一個淺拷貝用於保存，避免修改原始資料
        logger.info("準備保存預處理資料 (使用分片儲存策略)...")
        
        # 創建檢查點檔案路徑與其他必要的檔案
        checkpoint_path = os.path.join(self.preprocessed_path, 'checkpoint.json')
        target_csv_path = os.path.join(self.preprocessed_path, 'target.csv')
        metadata_path = os.path.join(self.preprocessed_path, 'features_metadata.json')
        
        # 檢查是否已存在處理檢查點
        current_chunk = 0
        if os.path.exists(checkpoint_path):
            try:
                import json
                with open(checkpoint_path, 'r') as f:
                    checkpoint_data = json.load(f)
                    current_chunk = checkpoint_data.get('current_chunk', 0)
                    logger.info(f"發現處理檢查點，將從分塊 {current_chunk} 繼續處理")
            except Exception as e:
                logger.warning(f"讀取檢查點文件失敗: {str(e)}，將從頭開始處理")
                current_chunk = 0
        
        # 分塊保存策略，避免記憶體溢出
        chunk_size = 500000  # 減小每個分塊的行數，避免後續讀取時的記憶體問題
        total_rows = len(self.features)
        chunks = list(range(0, total_rows, chunk_size)) + [total_rows]
        total_chunks = len(chunks) - 1
        logger.info(f"將分 {total_chunks} 批次保存 {total_rows} 筆資料，從第 {current_chunk} 批次開始")
        
        # 特別處理IP地址和其他大量唯一值的欄位
        ip_cols = [col for col in self.features.columns if 'IP' in col or 'HTTP' in col or 'Simillar' in col]
        if ip_cols:
            logger.info(f"檢測到可能包含大量唯一值的欄位: {ip_cols}")
        
        # 儲存列名和資料類型資訊，以便後續載入重建
        if current_chunk == 0:
            # 儲存特徵元數據
            columns = list(self.features.columns)
            dtypes = {col: str(self.features[col].dtype) for col in columns}
            
            # 記錄重要的中繼資料
            metadata = {
                'columns': columns,
                'dtypes': dtypes,
                'total_rows': total_rows,
                'chunk_size': chunk_size,
                'total_chunks': total_chunks,
                'high_cardinality_columns': ip_cols,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 儲存元數據
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"已儲存特徵元數據: {metadata_path}")
            
            # 保存目標變數 (通常體積較小，可一次處理)
            pd.DataFrame({'target': self.target}).to_csv(target_csv_path, index=False)
            logger.info(f"目標資料已保存至CSV: {target_csv_path}")
        
        # 分塊保存特徵資料 - 每個分塊儲存為獨立檔案
        import json
        for i in range(current_chunk, total_chunks):
            start_idx = chunks[i]
            end_idx = chunks[i+1]
            
            try:
                # 定義此分塊的檔案路徑
                chunk_path = os.path.join(features_dir, f'chunk_{i:05d}.parquet')
                
                # 檢查是否已存在此分塊檔案（支援續傳）
                if os.path.exists(chunk_path):
                    logger.info(f"分塊檔案已存在，跳過：{chunk_path}")
                    continue
                
                # 取出當前分塊
                chunk = self.features.iloc[start_idx:end_idx].copy()
                
                # 針對類別列的特殊處理，避免記憶體暴增
                for col in chunk.select_dtypes(include=['category']).columns:
                    if col in ip_cols:
                        # 對於 IP 地址或大量唯一值的列，使用較節省記憶體的方法
                        logger.info(f"使用記憶體優化方法處理大量唯一值欄位: '{col}'")
                        # 維持原始編碼，不轉字串
                        pass
                    else:
                        # 其他類別列可以安全地轉換為字串
                        try:
                            chunk[col] = chunk[col].astype(str)
                        except Exception as e:
                            logger.warning(f"轉換列 '{col}' 失敗: {str(e)}，保持原狀")
                
                # 儲存為獨立的 Parquet 檔案
                if POLARS_AVAILABLE:
                    # 使用 Polars 高效保存 Parquet (更節省記憶體)
                    pl.from_pandas(chunk).write_parquet(chunk_path)
                else:
                    # 回退到 Pandas
                    chunk.to_parquet(chunk_path, index=False)
                    
                logger.info(f"已保存分塊 {i+1}/{total_chunks} 至獨立檔案: {chunk_path}")
                
                # 更新檢查點
                with open(checkpoint_path, 'w') as f:
                    json.dump({
                        'current_chunk': i + 1,
                        'total_chunks': total_chunks,
                        'last_processed_row': end_idx,
                        'total_rows': total_rows,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }, f)
                    
                logger.info(f"已更新檢查點: 批次 {i+1}/{total_chunks}")
                
                # 強制釋放記憶體
                del chunk
                gc.collect()
                print_memory_usage()
                
            except Exception as e:
                logger.error(f"處理分塊 {i+1} 時發生錯誤: {str(e)}")
                logger.info(f"已保存檢查點，可在程式重啟後從批次 {i} 繼續處理")
                raise  # 重新引發異常，讓上層捕獲
        
        logger.info(f"特徵資料已全部保存為分片檔案: {features_dir}")
        
        # 嘗試使用 Polars 進行高效率 Parquet 轉換（低記憶體方式）
        try:
            # 檢查 Polars 是否可用
            if POLARS_AVAILABLE:
                logger.info("使用 Polars 進行低記憶體 Parquet 轉換...")
                feature_path = os.path.join(self.preprocessed_path, 'features.parquet')
                target_path = os.path.join(self.preprocessed_path, 'target.parquet')
                
                # 使用 Polars 分批讀取 CSV 並保存為 Parquet
                logger.info(f"使用 Polars 從 CSV 轉換為 Parquet: {feature_csv_path} -> {feature_path}")
                
                # 分批處理以減少記憶體使用
                batch_size = 100000  # 每批處理的行數
                
                # 首先嘗試獲取總行數
                try:
                    # 使用 wc -l 快速計算行數
                    import subprocess
                    result = subprocess.run(['wc', '-l', feature_csv_path], capture_output=True, text=True)
                    total_rows = int(result.stdout.split()[0]) - 1  # 減去標題行
                    logger.info(f"CSV 文件總行數: {total_rows}")
                except Exception as e:
                    logger.warning(f"無法獲取行數: {str(e)}，將使用預設批次大小")
                    total_rows = None
                
                # 使用 Polars 進行分批處理
                if total_rows:
                    # 計算批次數
                    num_batches = (total_rows + batch_size - 1) // batch_size
                    logger.info(f"將分 {num_batches} 批次進行 Parquet 轉換")
                    
                    # 獲取 CSV 標題
                    df_schema = pl.read_csv(feature_csv_path, n_rows=1)
                    columns = df_schema.columns
                    
                    # 初始化空的 Parquet 檔案
                    first_batch = pl.read_csv(feature_csv_path, n_rows=1)
                    first_batch.write_parquet(feature_path)
                    
                    # 分批讀取並追加
                    for i in range(num_batches):
                        offset = i * batch_size + 1  # 加1跳過標題
                        # 使用「掃描」模式，而不是一次性載入整個資料
                        batch_df = pl.scan_csv(
                            feature_csv_path, 
                            skip_rows=offset,
                            n_rows=batch_size,
                            has_header=False,
                            new_columns=columns
                        ).collect()
                        
                        # 追加到 Parquet 文件
                        if i == 0:
                            batch_df.write_parquet(feature_path)
                        else:
                            batch_df.write_parquet(feature_path, mode="append")
                            
                        logger.info(f"已處理批次 {i+1}/{num_batches}")
                        
                        # 強制釋放記憶體
                        del batch_df
                        gc.collect()
                        print_memory_usage()
                else:
                    # 如果無法獲取行數，使用最保守的方法
                    logger.info("使用 Polars LazyFrame 掃描 CSV 並轉換為 Parquet")
                    pl.scan_csv(feature_csv_path).collect().write_parquet(feature_path)
                
                # 處理目標變數 (通常比較小)
                logger.info(f"處理目標變數: {target_csv_path} -> {target_path}")
                pl.read_csv(target_csv_path).write_parquet(target_path)
                
                logger.info(f"Polars 成功將資料轉換為 Parquet 格式: {feature_path}")
                
            else:
                # Polars 不可用，使用 CSV 格式
                logger.warning("Polars 不可用，僅使用 CSV 格式儲存")
                feature_path = os.path.join(self.preprocessed_path, 'features.csv')
                target_path = os.path.join(self.preprocessed_path, 'target.csv')
                
                # 創建標記文件，表示 Parquet 格式不可用
                feature_parquet_path = os.path.join(self.preprocessed_path, 'features.parquet.unavailable')
                target_parquet_path = os.path.join(self.preprocessed_path, 'target.parquet.unavailable')
                
                with open(feature_parquet_path, 'w') as f:
                    f.write("Parquet format unavailable because Polars is not installed. Use CSV format instead.")
                    
                with open(target_parquet_path, 'w') as f:
                    f.write("Parquet format unavailable because Polars is not installed. Use CSV format instead.")
                    
                logger.info(f"已創建標記文件，標示 Parquet 格式不可用: {feature_parquet_path}")
        except Exception as e:
            logger.warning(f"使用 Polars 保存 Parquet 格式失敗: {str(e)}，僅使用 CSV 格式")
            feature_path = os.path.join(self.preprocessed_path, 'features.csv')
            target_path = os.path.join(self.preprocessed_path, 'target.csv')
        
        # 保存編碼器和縮放器
        with open(os.path.join(self.preprocessed_path, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
            
        with open(os.path.join(self.preprocessed_path, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # 全部完成後，移除檢查點檔案
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            
        logger.info(f"預處理資料已全部保存至: {self.preprocessed_path}")
        logger.info("保存過程已完成，檢查點已清理")
        
    def _load_preprocessed_data(self):
        """載入預處理後的資料，並嘗試恢復任何缺少的預處理工具"""
        # 檢查必要的檔案路徑
        feature_path = os.path.join(self.preprocessed_path, 'features.parquet')
        target_path = os.path.join(self.preprocessed_path, 'target.parquet')
        feature_csv_path = os.path.join(self.preprocessed_path, 'features.csv')
        target_csv_path = os.path.join(self.preprocessed_path, 'target.csv')
        scaler_path = os.path.join(self.preprocessed_path, 'scaler.pkl')
        encoder_path = os.path.join(self.preprocessed_path, 'label_encoder.pkl')
        
        # 追蹤加載進度
        features_loaded = False
        target_loaded = False
        scaler_loaded = False
        encoder_loaded = False
        
        # 嘗試載入特徵數據
        try:
            if os.path.exists(feature_path):
                self.features = pd.read_parquet(feature_path)
                logger.info(f"從Parquet格式載入特徵: {feature_path}")
                features_loaded = True
            elif os.path.exists(feature_csv_path):
                self.features = pd.read_csv(feature_csv_path)
                logger.info(f"從CSV格式載入特徵: {feature_csv_path}")
                features_loaded = True
            else:
                logger.warning("未找到特徵資料文件")
        except Exception as e:
            logger.warning(f"載入特徵資料時出錯: {str(e)}")
            
        # 嘗試載入目標數據
        try:
            if os.path.exists(target_path):
                target_df = pd.read_parquet(target_path)
                logger.info(f"從Parquet格式載入目標: {target_path}")
                self.target = target_df['target']
                target_loaded = True
            elif os.path.exists(target_csv_path):
                target_df = pd.read_csv(target_csv_path)
                logger.info(f"從CSV格式載入目標: {target_csv_path}")
                self.target = target_df['target']
                target_loaded = True
            else:
                logger.warning("未找到目標資料文件")
        except Exception as e:
            logger.warning(f"載入目標資料時出錯: {str(e)}")
            
        # 如果特徵和目標都成功載入，則繼續處理
        if features_loaded and target_loaded:
            # 儲存特徵名稱
            self.feature_names = list(self.features.columns)
            
            # 嘗試載入縮放器
            try:
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                    logger.info(f"成功載入縮放器: {scaler_path}")
                    scaler_loaded = True
                else:
                    logger.warning(f"找不到縮放器文件: {scaler_path}，將重新建立")
            except Exception as e:
                logger.warning(f"載入縮放器時出錯: {str(e)}，將重新建立")
                
            # 嘗試載入標籤編碼器
            try:
                if os.path.exists(encoder_path):
                    with open(encoder_path, 'rb') as f:
                        self.label_encoder = pickle.load(f)
                    logger.info(f"成功載入標籤編碼器: {encoder_path}")
                    encoder_loaded = True
                else:
                    logger.warning(f"找不到標籤編碼器文件: {encoder_path}，將重新建立")
            except Exception as e:
                logger.warning(f"載入標籤編碼器時出錯: {str(e)}，將重新建立")
            
            # 如果缺少縮放器，創建一個默認的而不是嘗試從所有數據重建（節省記憶體）
            if not scaler_loaded:
                try:
                    logger.info("創建默認縮放器（避免記憶體密集型重建）...")
                    # 直接創建標準縮放器，無需在全數據集上訓練
                    self.scaler = StandardScaler()
                    # 將mean_和scale_設定為合理的默認值
                    # 註：這裡假設數據已經被標準化過，所以使用簡單的标识转换
                    # 由於數據已經標準化，所以我們可以設置mean_=0, scale_=1
                    
                    # 簡單檢查特徵中的数值列
                    num_cols = self.features.select_dtypes(include=['int', 'float']).columns
                    
                    if len(num_cols) > 0:
                        logger.info(f"檢測到 {len(num_cols)} 個數值特徵欄位，為縮放器設置默認參數")
                        # 創建並初始化標準化器參數
                        self.scaler.mean_ = np.zeros(len(num_cols))
                        self.scaler.scale_ = np.ones(len(num_cols))
                        self.scaler.var_ = np.ones(len(num_cols))
                        self.scaler.n_features_in_ = len(num_cols)
                        self.scaler.n_samples_seen_ = 1
                        self.scaler.feature_names_in_ = np.array(num_cols)
                    
                    logger.info("成功創建默認縮放器")
                    
                    # 保存默認縮放器
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(self.scaler, f)
                    logger.info(f"默認縮放器已保存至: {scaler_path}")
                except Exception as e:
                    logger.warning(f"創建默認縮放器時出錯: {str(e)}")
                    # 創建最簡單的縮放器物件，不設置任何參數
                    self.scaler = StandardScaler()
            
            # 如果缺少編碼器，嘗試從數據重建
            if not encoder_loaded:
                try:
                    logger.info("從載入的數據重建標籤編碼器...")
                    if isinstance(self.target, pd.Series):
                        self.label_encoder = LabelEncoder()
                        # 如果目標已經是數值型，只需創建一個簡單的映射
                        if pd.api.types.is_numeric_dtype(self.target):
                            unique_values = sorted(self.target.unique())
                            self.label_encoder.classes_ = np.array(unique_values)
                        else:
                            # 否則重新編碼
                            self.label_encoder.fit(self.target)
                        logger.info("成功從現有數據重建標籤編碼器")
                        # 保存重建的編碼器
                        with open(encoder_path, 'wb') as f:
                            pickle.dump(self.label_encoder, f)
                        logger.info(f"重建的標籤編碼器已保存至: {encoder_path}")
                    else:
                        logger.warning("無法重建標籤編碼器：目標變數格式不正確")
                except Exception as e:
                    logger.warning(f"重建標籤編碼器時出錯: {str(e)}")
            
            logger.info(f"預處理資料載入完成: 特徵形狀={self.features.shape}, 目標形狀={self.target.shape}")
            logger.info(f"加載狀態: 特徵={features_loaded}, 目標={target_loaded}, 縮放器={scaler_loaded}, 編碼器={encoder_loaded}")
            return True
        else:
            # 如果關鍵數據沒有成功載入，則返回失敗
            missing = []
            if not features_loaded: missing.append("特徵")
            if not target_loaded: missing.append("目標")
            logger.error(f"預處理資料載入失敗：缺少必要的數據文件 ({', '.join(missing)})")
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
        
    def _load_with_incremental_processing(self, file_list):
        """增強版增量式加載多個文件 - 使用更高效的串流處理"""
        logger.info(f"增量式加載 {len(file_list)} 個文件 (增強串流處理)")

        # 初始化 DataFrame
        self.df = pd.DataFrame()
        
        # 逐個文件處理
        for file_idx, file_path in enumerate(tqdm(file_list, desc="增量式加載CSV")):
            try:
                # 對於大文件使用分塊處理
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                
                if file_size_mb > 500:  # 大於500MB的文件
                    # 估計合理的分塊大小
                    sample_size = 1000
                    sample_df = pd.read_csv(file_path, nrows=sample_size)
                    avg_row_size_bytes = sample_df.memory_usage(deep=True).sum() / len(sample_df)
                    
                    # 算出每個批次的行數 (目標每批次100MB)
                    rows_per_chunk = int((100 * 1024 * 1024) / avg_row_size_bytes)
                    logger.info(f"大文件分塊處理，每批次約 {rows_per_chunk} 行")
                    
                    # 分塊讀取
                    dfs = []
                    for chunk in pd.read_csv(file_path, chunksize=rows_per_chunk, low_memory=False):
                        # 優化記憶體使用
                        chunk = self._enhanced_optimize_dataframe_memory(chunk)
                        dfs.append(chunk)
                        
                        # 定期合併以釋放記憶體
                        if len(dfs) >= 5:
                            df_merged = pd.concat(dfs, ignore_index=True)
                            if self.df.empty:
                                self.df = df_merged
                            else:
                                self.df = pd.concat([self.df, df_merged], ignore_index=True)
                            
                            # 清理中間df
                            dfs = []
                            gc.collect()
                    
                    # 處理剩餘的塊
                    if dfs:
                        df_merged = pd.concat(dfs, ignore_index=True)
                        if self.df.empty:
                            self.df = df_merged
                        else:
                            self.df = pd.concat([self.df, df_merged], ignore_index=True)
                else:
                    # 小文件直接讀取
                    df = pd.read_csv(file_path, low_memory=False)
                    df = self._enhanced_optimize_dataframe_memory(df)
                    
                    if self.df.empty:
                        self.df = df
                    else:
                        self.df = pd.concat([self.df, df], ignore_index=True)
                
                logger.info(f"已處理文件 {file_idx+1}/{len(file_list)}: {file_path}")
                
                # 定期垃圾回收
                if (file_idx + 1) % 3 == 0:
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"處理文件失敗 {file_path}: {str(e)}")
        
        if self.df.empty:
            raise ValueError("所有文件均處理失敗")
            
        logger.info(f"增量式加載完成，DataFrame形狀: {self.df.shape}")
    
    def _load_single_file_incrementally(self, file_path):
        """增量式加載單個CSV文件 - 使用分塊處理"""
        logger.info(f"增量式加載單個文件: {file_path}")
        
        # 檢查文件大小
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"文件大小: {file_size_mb:.2f} MB")
        
        # 對於大文件使用分塊處理
        if file_size_mb > 300:  # 大於300MB
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
                if len(dfs) >= 5:
                    self.df = pd.concat(dfs, ignore_index=True)
                    
                    # 清理中間df
                    dfs = []
                    gc.collect()
            
            # 處理剩餘的塊
            if dfs:
                if self.df is None:
                    self.df = pd.concat(dfs, ignore_index=True)
                else:
                    self.df = pd.concat([self.df] + dfs, ignore_index=True)
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
                
                # 檢查是否有記憶體壓力
                if get_memory_usage()['percent'] > 80:
                    logger.warning("記憶體使用率過高，將提前合併並清理")
                    temp_df = pd.concat(dfs, ignore_index=True)
                    dfs = [temp_df]
                    gc.collect()
                    
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
        
        if not POLARS_AVAILABLE:
            logger.warning("Polars不可用，回退到標準載入方法")
            self._load_single_file_incrementally(file_path)
            return
            
        try:
            # 使用LazyFrame高效載入
            lf = pl.scan_csv(file_path)
            
            # 應用採樣(如果需要)
            if self.use_sampling and self.sampling_strategy == 'random':
                lf = lf.sample(fraction=self.sampling_ratio)
                
            # 收集DataFrame
            pf = lf.collect()
            logger.info(f"Polars成功載入，形狀: {pf.shape}")
            
            # 轉換為Pandas DataFrame
            self.df = pf.to_pandas()
            
            # 釋放記憶體
            del pf
            gc.collect()
        except Exception as e:
            logger.error(f"Polars載入失敗: {str(e)}")
            logger.info("回退到標準載入方法")
            self._load_single_file_incrementally(file_path)
            
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
                self._load_with_incremental_processing(file_list)
            else:
                self._load_data_at_once(file_list)
                
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
