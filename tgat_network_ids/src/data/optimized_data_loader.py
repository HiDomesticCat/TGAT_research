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

    def _ensure_parquet_files(self, file_list):
        """增強版確保CSV文件轉換為Parquet格式 - 使用並行處理和PyArrow優化"""
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # 獲取需要轉換的文件列表
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

        # 轉換函數：將CSV轉為Parquet - 使用PyArrow優化
        def convert_to_parquet(file_path):
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
                        chunk = self._enhanced_optimize_dataframe_memory(chunk)

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
                        if POLARS_AVAILABLE and self.use_polars:
                            df = pl.read_csv(file_path).collect()
                            df.write_parquet(parquet_path, compression="snappy")
                        else:
                            # 使用pandas
                            df = pd.read_csv(file_path, low_memory=False)
                            df = self._enhanced_optimize_dataframe_memory(df)
                            df.to_parquet(parquet_path, compression='snappy', index=False)
                    except Exception as e:
                        logger.error(f"轉換時出錯: {str(e)}, 嘗試備用方法")
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
