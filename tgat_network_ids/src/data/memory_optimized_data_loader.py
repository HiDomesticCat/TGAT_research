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
from ..utils.memory_utils import (
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
                - data.use_sampling: 是否使用資料採樣
                - data.sampling_strategy: 採樣策略 ('random', 'stratified')
                - data.sampling_ratio: 採樣比例 (0-1)
                - data.min_samples_per_class: 每個類別的最小樣本數
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
        
        # 資料採樣相關設置
        self.use_sampling = data_config.get('use_sampling', False)
        self.sampling_strategy = data_config.get('sampling_strategy', 'stratified')
        self.sampling_ratio = data_config.get('sampling_ratio', 0.1)  # 默認採樣 10%
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
        """增量式加載多個文件 - 串流處理版本"""
        logger.info(f"增量式加載 {len(file_list)} 個文件 (串流處理版本)")
        
        # 轉換所有文件為Parquet格式（如果需要）
        self._ensure_parquet_files(file_list)
        
        # 使用Parquet格式直接載入和處理資料
        self._process_parquet_files(file_list)
    
    def _ensure_parquet_files(self, file_list):
        """確保所有CSV文件都有對應的Parquet版本"""
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        # 獲取需要轉換的文件列表（沒有對應parquet文件的csv文件）
        conversion_needed = []
        for csv_file in file_list:
            parquet_file = csv_file.replace('.csv', '.parquet')
            if not os.path.exists(parquet_file) and self.save_preprocessed:
                conversion_needed.append(csv_file)
        
        if not conversion_needed:
            logger.info("所有文件已有Parquet格式，無需轉換")
            return
            
        # 獲取可用的CPU核心數，保留2核心給系統
        num_cores = max(1, mp.cpu_count() - 2)
        logger.info(f"使用{num_cores}個核心轉換{len(conversion_needed)}個文件為Parquet格式")
        
        # 轉換函數：將CSV轉為Parquet
        def convert_to_parquet(file_path):
            parquet_path = file_path.replace('.csv', '.parquet')
            try:
                # 使用PyArrow直接轉換（更快且記憶體效率更高）
                try:
                    import pyarrow as pa
                    import pyarrow.csv as pc
                    import pyarrow.parquet as pq
                    
                    # 使用優化設置
                    parse_options = pc.ParseOptions(delimiter=',')
                    convert_options = pc.ConvertOptions(
                        strings_to_categorical=True,
                        include_columns=None
                    )
                    read_options = pc.ReadOptions(
                        block_size=16 * 1024 * 1024,  # 16MB塊大小
                        use_threads=True
                    )
                    
                    # 直接轉換並寫入Parquet
                    table = pc.read_csv(file_path, 
                                       parse_options=parse_options,
                                       convert_options=convert_options,
                                       read_options=read_options)
                    
                    pq.write_table(table, parquet_path, compression='snappy')
                    return f"成功轉換: {file_path} -> {parquet_path}"
                    
                except ImportError:
                    # 使用pandas轉換（較慢但兼容性更好）
                    # 對大文件使用分塊處理
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    
                    if file_size > 1000:  # 大於1GB的文件使用分塊處理
                        # 估計合理的分塊大小（每塊約300MB內存）
                        sample_size = 1000
                        sample_df = pd.read_csv(file_path, nrows=sample_size)
                        avg_row_size_bytes = sample_df.memory_usage(deep=True).sum() / len(sample_df)
                        rows_per_chunk = int((300 * 1024 * 1024) / avg_row_size_bytes)
                        
                        # 分塊處理大文件
                        writer = None
                        for i, chunk in enumerate(pd.read_csv(file_path, chunksize=rows_per_chunk, low_memory=False)):
                            # 優化記憶體使用
                            chunk = optimize_dataframe_memory(chunk)
                            
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
                        df = pd.read_csv(file_path, low_memory=False)
                        df = optimize_dataframe_memory(df)
                        df.to_parquet(parquet_path, compression='snappy', index=False)
                        return f"成功轉換: {file_path} -> {parquet_path}"
                
            except Exception as e:
                return f"轉換失敗 {file_path}: {str(e)}"
        
        # 並行轉換文件
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = [executor.submit(convert_to_parquet, file) for file in conversion_needed]
            
            for future in tqdm(as_completed(futures), total=len(conversion_needed), desc="轉換CSV至Parquet"):
                try:
                    result = future.result()
                    logger.info(result)
                except Exception as e:
                    logger.error(f"轉換過程中發生錯誤: {str(e)}")
    
    def _process_parquet_files(self, file_list):
        """串流處理Parquet文件而不是全部載入記憶體"""
        # 獲取所有文件的Parquet路徑
        parquet_files = [f.replace('.csv', '.parquet') if f.endswith('.csv') else f for f in file_list]
        available_parquet = [f for f in parquet_files if os.path.exists(f)]
        
        if not available_parquet:
            raise ValueError("無法找到可用的Parquet文件")
            
        logger.info(f"開始串流處理{len(available_parquet)}個Parquet文件")
        
        # 使用串流處理避免記憶體溢出
        # 首先獲取整合後的模式和列數據類型
        try:
            import pyarrow.parquet as pq
            sample_schema = pq.read_schema(available_parquet[0])
            
            # 獲取總行數估計
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
                    
                    # 每次讀取一個行組
                    for i in range(num_row_groups):
                        batch_df = parquet_file.read_row_group(i).to_pandas()
                        batch_df = optimize_dataframe_memory(batch_df)
                        
                        # 處理數據分析或採樣
                        if self.use_sampling and self.df.empty:
                            # 對於第一批數據，儲存樣本用於後續的採樣決策
                            if i == 0:
                                self.df = batch_df
                            else:
                                # 採樣策略 - 每n行取一行
                                sample_rate = int(1.0 / self.sampling_ratio)
                                sampled_rows = batch_df.iloc[::sample_rate]
                                self.df = pd.concat([self.df, sampled_rows], ignore_index=True)
                        else:
                            # 如果不採樣或已有數據，直接連接
                            if self.df.empty:
                                self.df = batch_df
                            else:
                                self.df = pd.concat([self.df, batch_df], ignore_index=True)
                        
                        processed_rows += len(batch_df)
                        logger.info(f"已處理 {processed_rows}/{total_rows} 行 ({processed_rows/total_rows:.1%})")
                        
                        # 定期釋放記憶體
                        del batch_df
                        gc.collect()
                else:
                    # 小於2GB的文件，一次性讀取
                    file_df = pd.read_parquet(pq_file)
                    file_df = optimize_dataframe_memory(file_df)
                    
                    if self.df.empty:
                        self.df = file_df
                    else:
                        self.df = pd.concat([self.df, file_df], ignore_index=True)
                    
                    processed_rows += len(file_df)
                    logger.info(f"已處理 {processed_rows}/{total_rows} 行 ({processed_rows/total_rows:.1%})")
                    
                    # 釋放記憶體
                    del file_df
                    gc.collect()
            
            # 對整合後的數據進行全局優化
            self.df = optimize_dataframe_memory(self.df)
            logger.info(f"完成數據加載: 形狀 {self.df.shape}")
            
        except Exception as e:
            logger.error(f"處理Parquet文件時發生錯誤: {str(e)}")
            # 備用方法：如果PyArrow失敗，使用pandas直接加載
            self._fallback_load_parquet(available_parquet)
    
    def _fallback_load_parquet(self, parquet_files):
        """備用加載方法，使用pandas直接加載Parquet"""
        logger.warning("使用備用方法加載Parquet文件")
        dfs = []
        
        for file_idx, pq_file in enumerate(parquet_files):
            try:
                logger.info(f"加載文件 {file_idx+1}/{len(parquet_files)}: {pq_file}")
                df = pd.read_parquet(pq_file)
                df = optimize_dataframe_memory(df)
                dfs.append(df)
                
                # 定期釋放記憶體
                if (file_idx + 1) % 3 == 0:
                    gc.collect()
            except Exception as e:
                logger.error(f"加載 {pq_file} 失敗: {str(e)}")
        
        if dfs:
            # 分批次合併以減少記憶體峰值
            self.df = None
            batch_size = 5  # 每次合併5個DataFrame
            for i in range(0, len(dfs), batch_size):
                batch = dfs[i:i+batch_size]
                batch_df = pd.concat(batch, ignore_index=True)
                
                if self.df is None:
                    self.df = batch_df
                else:
                    self.df = pd.concat([self.df, batch_df], ignore_index=True)
                
                # 釋放批次記憶體
                del batch
                del batch_df
                gc.collect()
            
            logger.info(f"完成數據加載: 形狀 {self.df.shape}")
        else:
            raise ValueError("所有文件加載失敗")
    
    def _load_single_file_incrementally(self, file_path):
        """增量式加載單個文件 - 優化效能版本"""
        logger.info(f"加載文件: {file_path}")
        
        # 檢查是否有預處理的 Parquet 文件
        binary_path = file_path.replace('.csv', '.parquet')
        if os.path.exists(binary_path) and self.save_preprocessed:
            logger.info(f"使用現有 Parquet 文件: {binary_path}")
            
            # 優先使用 pyarrow 加載 Parquet 文件
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(binary_path, memory_map=True)
                self.df = table.to_pandas()
                logger.info(f"Parquet 加載完成: {self.df.shape}")
                return
            except Exception as e:
                # 如果 pyarrow 失敗，使用 pandas
                try:
                    self.df = pd.read_parquet(binary_path)
                    logger.info(f"Pandas Parquet 加載完成: {self.df.shape}")
                    return
                except Exception as e2:
                    logger.warning(f"Parquet 加載失敗: {e2}，將使用 CSV")
        
        # 獲取文件大小
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        
        # 對於小型文件或沒有 PyArrow 的情況，直接使用 Pandas
        if file_size <= self.chunk_size_mb or not self._has_pyarrow():
            try:
                # 使用效能最佳的類型加載一次性 CSV
                dtype_dict = {
                    'Protocol': 'category',
                    'Destination Port': 'int32',
                    'Flow Duration': 'float32'
                }
                self.df = pd.read_csv(file_path, dtype=dtype_dict, low_memory=False)
                logger.info(f"一次性加載 CSV 完成: {self.df.shape}")
                
                # 保存為 Parquet 格式以加速未來加載
                if self.save_preprocessed:
                    self.df.to_parquet(binary_path, index=False)
                    logger.info(f"已保存為 Parquet 格式: {binary_path}")
                
                return
            except Exception as e:
                logger.warning(f"Pandas 加載失敗: {e}，嘗試分塊加載")
        
        # 使用 PyArrow 加載大型文件 (如果可用)
        if self._has_pyarrow():
            try:
                import pyarrow.csv as pc
                import pyarrow.parquet as pq
                
                # 優化的 CSV 解析設置
                parse_options = pc.ParseOptions(delimiter=',')
                convert_options = pc.ConvertOptions(strings_to_categorical=True)
                read_options = pc.ReadOptions(
                    block_size=16 * 1024 * 1024,  # 16MB 塊大小
                    use_threads=True
                )
                
                # 直接轉換並加載
                table = pc.read_csv(file_path, 
                                   parse_options=parse_options,
                                   convert_options=convert_options,
                                   read_options=read_options)
                
                # 保存 Parquet 文件
                if self.save_preprocessed:
                    pq.write_table(table, binary_path)
                    logger.info(f"已保存為 Parquet 格式: {binary_path}")
                
                # 轉換為 DataFrame
                self.df = table.to_pandas()
                logger.info(f"PyArrow 加載完成: {self.df.shape}")
                
                # 釋放資源
                del table
                clean_memory()
                
                return
            except Exception as e:
                logger.warning(f"PyArrow 加載失敗: {e}，使用 Pandas 分塊加載")
        
        # 最後備用方案：Pandas 分塊加載
        try:
            # 優化分塊大小 - 使用固定的大塊
            chunk_size = 2000000  # 使用固定的 200 萬行塊大小
            
            # 一次讀取多個塊，然後連接
            chunks = []
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
                chunks.append(chunk)
            
            # 合併所有塊
            self.df = pd.concat(chunks, ignore_index=True)
            logger.info(f"Pandas 分塊加載完成: {self.df.shape}")
            
            # 保存為 Parquet 格式
            if self.save_preprocessed:
                self.df.to_parquet(binary_path, index=False)
                logger.info(f"已保存為 Parquet 格式: {binary_path}")
            
            # 釋放資源
            del chunks
            clean_memory()
            
        except Exception as e:
            logger.error(f"所有加載方法均失敗: {e}")
            raise
    
    def _has_pyarrow(self):
        """檢查是否有 PyArrow 可用"""
        try:
            import pyarrow
            return True
        except ImportError:
            return False
    
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
        
        # 如果啟用了資料採樣，先進行採樣
        if self.use_sampling and self.df is not None:
            self._sample_data()

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
        with track_memory_usage("加載預處理資料"):
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
    
    def _sample_data(self):
        """
        對資料進行採樣，減少資料量
        
        支持兩種採樣策略：
        1. random: 隨機採樣
        2. stratified: 分層採樣，確保各類別的分布比例一致
        """
        if self.df is None or len(self.df) == 0:
            logger.warning("資料為空，無法進行採樣")
            return
        
        # 自動檢測標籤欄位
        label_column = self._detect_label_column()
        
        # 獲取原始資料大小
        original_size = len(self.df)
        logger.info(f"原始資料大小: {original_size} 行")
        
        # 計算採樣後的目標大小
        target_size = int(original_size * self.sampling_ratio)
        logger.info(f"採樣比例: {self.sampling_ratio}, 目標大小: {target_size} 行")
        
        # 獲取類別分布
        class_counts = self.df[label_column].value_counts()
        logger.info(f"原始類別分布:\n{class_counts}")
        
        # 根據採樣策略進行採樣
        if self.sampling_strategy == 'random':
            # 隨機採樣
            sampled_df = self.df.sample(n=target_size, random_state=self.random_state)
            logger.info(f"隨機採樣完成，採樣後大小: {len(sampled_df)} 行")
        
        elif self.sampling_strategy == 'stratified':
            # 分層採樣
            logger.info("使用分層採樣策略")
            
            # 計算每個類別的採樣數量，確保至少有 min_samples_per_class 個樣本
            class_sample_counts = {}
            remaining_samples = target_size
            
            for class_label, count in class_counts.items():
                # 計算比例採樣數量
                proportional_count = int(count * self.sampling_ratio)
                
                # 確保至少有 min_samples_per_class 個樣本
                sample_count = max(proportional_count, min(self.min_samples_per_class, count))
                
                # 如果原始數量小於最小樣本數，則全部保留
                if count <= self.min_samples_per_class:
                    sample_count = count
                
                class_sample_counts[class_label] = sample_count
                remaining_samples -= sample_count
            
            # 如果還有剩餘樣本，按比例分配給各類別
            if remaining_samples > 0:
                total_remaining_count = sum([count for label, count in class_counts.items() 
                                          if count > class_sample_counts[label]])
                
                if total_remaining_count > 0:
                    for class_label, count in class_counts.items():
                        if count > class_sample_counts[class_label]:
                            # 按比例分配剩餘樣本
                            extra_samples = int(remaining_samples * (count / total_remaining_count))
                            class_sample_counts[class_label] += extra_samples
                            remaining_samples -= extra_samples
            
            # 如果還有剩餘樣本，分配給最大的類別
            if remaining_samples > 0:
                largest_class = class_counts.idxmax()
                class_sample_counts[largest_class] += remaining_samples
            
            # 進行分層採樣
            sampled_dfs = []
            for class_label, sample_count in class_sample_counts.items():
                class_df = self.df[self.df[label_column] == class_label]
                
                # 如果樣本數大於類別總數，則全部保留
                if sample_count >= len(class_df):
                    sampled_class_df = class_df
                else:
                    sampled_class_df = class_df.sample(n=sample_count, random_state=self.random_state)
                
                sampled_dfs.append(sampled_class_df)
                logger.info(f"類別 {class_label}: 原始 {len(class_df)} 行，採樣 {len(sampled_class_df)} 行")
            
            # 合併所有採樣後的資料
            sampled_df = pd.concat(sampled_dfs, ignore_index=True)
            logger.info(f"分層採樣完成，採樣後大小: {len(sampled_df)} 行")
        
        else:
            logger.warning(f"不支持的採樣策略: {self.sampling_strategy}，使用隨機採樣")
            sampled_df = self.df.sample(n=target_size, random_state=self.random_state)
        
        # 更新資料集
        self.df = sampled_df
        
        # 顯示採樣後的類別分布
        new_class_counts = self.df[label_column].value_counts()
        logger.info(f"採樣後類別分布:\n{new_class_counts}")
        
        # 清理記憶體
        clean_memory()
    
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
        Build temporal edge relationships
        
        Parameters:
            max_edges (int): Maximum number of edges limit
            
        Returns:
            list: Edge list, each element is (src, dst, time, edge_feat)
        """
        # Skip real data processing and directly create synthetic edges
        # This is much faster and avoids memory issues
        return self._create_synthetic_edges(max_edges)
    
    def _create_synthetic_edges(self, max_edges=500000):
        """
        Create synthetic edges when real data is not available or for faster processing
        
        Parameters:
            max_edges (int): Maximum number of edges limit
            
        Returns:
            list: Edge list, each element is (src, dst, time, edge_feat)
        """
        import random
        random.seed(42)  # Set random seed for reproducibility
        
        edges = []
        logger.info("Creating synthetic edges")
        
        # Determine number of nodes
        if self.df is not None:
            num_nodes = len(self.df)
        elif self.features is not None:
            num_nodes = len(self.features)
        else:
            # Default to a reasonable number if no data is available
            num_nodes = 1000
            logger.warning(f"No data available, using default node count: {num_nodes}")
        
        # Limit the number of nodes to process to improve performance
        # This is especially important for large datasets
        max_nodes = min(num_nodes, 10000)  # Process at most 10,000 nodes
        
        # Create a connected graph structure (tree)
        # First, ensure all nodes are connected in a tree structure
        for i in range(1, max_nodes):
            if len(edges) >= max_edges:
                break
                
            # Connect to a random previous node
            parent = random.randint(0, i-1)
            timestamp = float(i)  # Use index as timestamp
            
            # Edge features: random values
            edge_feat = [random.random(), random.random(), random.random()]
            
            # Add edge
            edges.append((parent, i, timestamp, edge_feat))
        
        # Add additional random edges to increase connectivity
        remaining_edges = max_edges - len(edges)
        if remaining_edges > 0:
            additional_edges = min(remaining_edges, max_nodes * 2)  # Average 2 additional edges per node
            
            for _ in range(additional_edges):
                src_idx = random.randint(0, max_nodes - 1)
                dst_idx = random.randint(0, max_nodes - 1)
                
                if src_idx != dst_idx:  # Avoid self-loops
                    # Use random timestamp
                    timestamp = float(random.randint(0, max_nodes))
                    
                    # Edge features: random values
                    edge_feat = [random.random(), random.random(), random.random()]
                    
                    # Add edge
                    edges.append((src_idx, dst_idx, timestamp, edge_feat))
        
        logger.info(f"Created {len(edges)} synthetic edges")
        return edges
    
    @memory_usage_decorator
    def get_sample_batch(self, batch_size=1000):
        """
        Get a sample batch for simulating dynamic graph updates
        
        Parameters:
            batch_size (int): Batch size
            
        Returns:
            tuple: (batch_features, batch_labels, batch_indices)
        """
        if self.features is None or self.target is None:
            self.preprocess()
        
        total_samples = len(self.target)
        if batch_size > total_samples:
            batch_size = total_samples
        
        # Random sampling
        indices = np.random.choice(total_samples, batch_size, replace=False)
        
        # Get features and labels for the selected indices
        if self.use_memory_mapping:
            # For memory-mapped arrays, use direct indexing
            batch_features = self.features[indices]
        else:
            # For regular arrays
            batch_features = self.features[indices]
        
        # Get labels
        batch_labels = self.target.iloc[indices].values if isinstance(self.target, pd.Series) else self.target[indices]
        
        return batch_features, batch_labels, indices
