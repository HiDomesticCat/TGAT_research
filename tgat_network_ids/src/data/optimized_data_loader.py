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
from sklearn.impute import SimpleImputer # 用於處理 NaN/inf
from sklearn.feature_selection import VarianceThreshold, f_classif, mutual_info_classif
import logging
from tqdm import tqdm
import gc
import time
import pickle
import glob
from datetime import datetime
import shutil
import ipaddress # 確保導入
import warnings # 確保導入
import dgl # 為 build_graph 示例導入
import torch # 為 build_graph 示例導入
from typing import Dict, List, Tuple, Set, Union, Optional, Any

# 導入記憶體優化工具
# 假設 utils 在上層目錄的 src/utils
# 注意：導入路徑可能需要根據您的實際專案結構調整
try:
    from ..utils.memory_utils import (
        memory_mapped_array, load_memory_mapped_array, save_dataframe_chunked,
        load_dataframe_chunked, optimize_dataframe_memory, clean_memory,
        memory_usage_decorator, track_memory_usage, print_memory_usage,
        get_memory_usage, print_optimization_suggestions, adaptive_batch_size,
        detect_memory_leaks, limit_gpu_memory
    )
except ImportError:
     print("警告：資料載入器無法從相對路徑導入記憶體工具。嘗試直接導入。")
     # 如果直接執行此文件或結構不同，則回退
     try:
         from utils.memory_utils import (
             memory_mapped_array, load_memory_mapped_array, save_dataframe_chunked,
             load_dataframe_chunked, optimize_dataframe_memory, clean_memory,
             memory_usage_decorator, track_memory_usage, print_memory_usage,
             get_memory_usage, print_optimization_suggestions, adaptive_batch_size,
             detect_memory_leaks, limit_gpu_memory
         )
     except ImportError as ie:
          print(f"直接導入 memory_utils 失敗: {ie}。將使用虛設函數。")
          def memory_mapped_array(*args, **kwargs): raise NotImplementedError
          def load_memory_mapped_array(*args, **kwargs): raise NotImplementedError
          def save_dataframe_chunked(*args, **kwargs): raise NotImplementedError
          def load_dataframe_chunked(*args, **kwargs): raise NotImplementedError
          def optimize_dataframe_memory(df, *args, **kwargs): return df # 返回原樣
          def clean_memory(*args, **kwargs): pass
          def memory_usage_decorator(func): return func
          def track_memory_usage(*args, **kwargs):
                def decorator(func):
                     return func
                return decorator
          def print_memory_usage(*args, **kwargs): pass
          def get_memory_usage(*args, **kwargs): return {}
          def print_optimization_suggestions(*args, **kwargs): pass
          def adaptive_batch_size(*args, **kwargs): return args[0] if args else 128
          def detect_memory_leaks(*args, **kwargs): pass
          def limit_gpu_memory(*args, **kwargs): pass

# 嘗試導入 Polars 和 PyArrow
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.csv as pc
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

# 配置日誌記錄器
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedMemoryOptimizedDataLoader:
    """增強版記憶體優化資料載入與預處理類別 (已修正)"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化資料載入器

        參數:
            config (dict): 配置字典
        """
        # --- 配置讀取與初始化 ---
        data_config = config.get('data', {})
        self.data_path = data_config.get('path', './data')
        self.test_size = float(data_config.get('test_size', 0.2))
        self.random_state = int(data_config.get('random_state', 42))
        self.batch_size = int(data_config.get('batch_size', 128))
        self.target_column_name = data_config.get('target_column', None) # 允許配置中指定目標列

        # 記憶體優化相關設置
        self.use_memory_mapping = bool(data_config.get('use_memory_mapping', False))
        self.save_preprocessed = bool(data_config.get('save_preprocessed', True))
        self.preprocessed_path = data_config.get('preprocessed_path', './preprocessed_data')
        self.incremental_loading = bool(data_config.get('incremental_loading', True))
        self.chunk_size_mb = int(data_config.get('chunk_size_mb', 200))
        self.use_compression = bool(data_config.get('use_compression', True))
        # 推薦使用 snappy 或 zstd (如果已安裝) 以獲得更好的性能
        self.compression_format = data_config.get('compression_format', 'snappy')

        # 新增的優化設置
        self.use_polars = bool(data_config.get('use_polars', POLARS_AVAILABLE))
        self.use_pyarrow = bool(data_config.get('use_pyarrow', PYARROW_AVAILABLE))
        # 警告：積極轉換為 float16 可能損失精度
        self.aggressive_dtypes = bool(data_config.get('aggressive_dtypes', False))
        self.aggressive_gc = bool(data_config.get('aggressive_gc', True))
        self.chunk_row_limit = int(data_config.get('chunk_row_limit', 200000))

        # 資料採樣相關設置
        self.use_sampling = bool(data_config.get('use_sampling', False))
        self.sampling_strategy = data_config.get('sampling_strategy', 'stratified') # 'stratified' 或 'random'
        self.sampling_ratio = float(data_config.get('sampling_ratio', 0.1))
        self.min_samples_per_class = int(data_config.get('min_samples_per_class', 1000)) # 分層採樣時，每個類別的最小樣本數

        # 確保預處理資料目錄存在
        if self.save_preprocessed:
             try:
                  os.makedirs(self.preprocessed_path, exist_ok=True)
                  logger.info(f"預處理目錄確保存在: {self.preprocessed_path}")
             except OSError as e:
                  logger.error(f"無法創建預處理目錄 {self.preprocessed_path}: {e}")
                  self.save_preprocessed = False # 禁用保存

        # 初始化預處理工具
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # 初始化資料變數
        self.df: Optional[pd.DataFrame] = None # 原始載入的 DataFrame
        self.features: Optional[np.ndarray] = None # 預處理後的特徵 (Numpy Array)
        self.target: Optional[np.ndarray] = None   # 預處理後的目標 (Numpy Array)
        self.feature_names: Optional[List[str]] = None # 特徵名稱列表
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.train_indices: Optional[np.ndarray] = None # 訓練集索引
        self.test_indices: Optional[np.ndarray] = None  # 測試集索引

        # 創建一個時間戳，可用於標記本次處理的文件
        self.process_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(f"初始化資料載入器: 資料路徑='{self.data_path}'")
        logger.info(f"記憶體優化: 映射={self.use_memory_mapping}, 保存預處理={self.save_preprocessed}, 增量載入={self.incremental_loading}")
        logger.info(f"效能: 使用 Polars={self.use_polars}, PyArrow={self.use_pyarrow}")
        logger.info(f"資料採樣: 啟用={self.use_sampling}, 策略={self.sampling_strategy}, 比例={self.sampling_ratio}")

    def _get_preprocessed_files(self) -> Dict[str, str]:
        """獲取預處理文件路徑 (使用固定名稱 + Parquet/Numpy/Pickle)"""
        # 使用固定名稱以方便查找和覆蓋舊文件
        file_prefix = "preprocessed_data"

        # 定義各種文件的路徑
        # 特徵和目標使用 Numpy 保存，因為它們是預處理的最終數值輸出
        features_file = os.path.join(self.preprocessed_path, f"{file_prefix}_features.npy")
        target_file = os.path.join(self.preprocessed_path, f"{file_prefix}_target.npy")
        # 其他元數據使用 Pickle 保存
        feature_names_file = os.path.join(self.preprocessed_path, f"{file_prefix}_feature_names.pkl")
        scaler_file = os.path.join(self.preprocessed_path, f"{file_prefix}_scaler.pkl")
        label_encoder_file = os.path.join(self.preprocessed_path, f"{file_prefix}_label_encoder.pkl")
        # 訓練/測試索引使用 Numpy 保存
        train_indices_file = os.path.join(self.preprocessed_path, f"{file_prefix}_train_indices.npy")
        test_indices_file = os.path.join(self.preprocessed_path, f"{file_prefix}_test_indices.npy")

        return {
            'features': features_file,
            'target': target_file,
            'feature_names': feature_names_file,
            'scaler': scaler_file,
            'label_encoder': label_encoder_file,
            'train_indices': train_indices_file,
            'test_indices': test_indices_file
        }

    def _check_preprocessed_exists(self) -> bool:
        """檢查所有必要的預處理文件是否存在"""
        files = self._get_preprocessed_files()
        # 定義必要的文件組件
        required_files = ['features', 'target', 'scaler', 'label_encoder', 'feature_names']
        all_exist = all(os.path.exists(files[f]) for f in required_files)

        if not all_exist:
             missing = [name for name in required_files if not os.path.exists(files[name])]
             logger.info(f"必要的預處理文件缺失: {missing}")
        return all_exist

    @memory_usage_decorator
    def load_data(self):
        """
        載入資料集。
        如果存在且完整，則載入預處理數據；否則載入原始數據。
        """
        logger.info(f"開始載入資料，資料路徑: '{self.data_path}'")
        print_memory_usage("載入資料開始")

        # 1. 嘗試載入預處理數據
        if self.save_preprocessed and self._check_preprocessed_exists():
            logger.info("發現完整的預處理文件，嘗試直接載入...")
            success = self._load_preprocessed_data()
            if success:
                logger.info("預處理資料載入成功。")
                # self.df 保持為 None，因為數據在 self.features/target 中
                print_memory_usage("預處理資料載入後")
                return self.df # 返回 None
            else:
                logger.warning("預處理資料載入失敗，將重新處理原始資料。")
                self._reset_data_attributes() # 清理可能部分加載的數據

        # 2. 如果預處理數據載入失敗或不存在，載入原始數據
        logger.info("載入原始數據...")
        if os.path.isdir(self.data_path):
            # 處理目錄中的所有 CSV 文件
            all_files = sorted([os.path.join(self.data_path, f)
                                for f in os.listdir(self.data_path)
                                if f.endswith('.csv') and not f.startswith('.')]) # 忽略隱藏文件
            if not all_files:
                raise ValueError(f"在目錄 '{self.data_path}' 中未找到有效的 CSV 檔案")
            logger.info(f"找到 {len(all_files)} 個 CSV 檔案進行處理: {all_files[:3]}...") # 只顯示前幾個

            if self.use_polars and POLARS_AVAILABLE:
                self._load_with_polars(all_files)
            elif self.incremental_loading:
                self._load_with_incremental_processing(all_files)
            else:
                self._load_data_at_once(all_files)

        elif os.path.isfile(self.data_path) and self.data_path.endswith('.csv'):
            # 處理單個 CSV 文件
            if self.use_polars and POLARS_AVAILABLE:
                self._load_single_file_with_polars(self.data_path)
            elif self.incremental_loading:
                self._load_single_file_incrementally(self.data_path)
            else:
                self._load_data_at_once([self.data_path])
        elif os.path.isfile(self.data_path) and self.data_path.endswith('.parquet'):
            # 支持讀取單個 Parquet 文件
             logger.info("一次性載入單個 Parquet 檔案...")
             engine = 'pyarrow' if PYARROW_AVAILABLE else 'fastparquet' # fastparquet 是備選
             try:
                  self.df = pd.read_parquet(self.data_path, engine=engine)
                  logger.info(f"載入 Parquet 檔案完成，形狀: {self.df.shape}")
             except Exception as e:
                  logger.error(f"使用 {engine} 引擎讀取 Parquet 失敗: {e}")
                  raise # 拋出錯誤

        else:
            raise FileNotFoundError(f"指定的資料路徑無效或不是支持的文件類型(CSV/Parquet): '{self.data_path}'")

        # 3. 記憶體優化
        if self.df is not None and not self.df.empty:
            logger.info("對載入的原始 DataFrame 進行記憶體優化...")
            self.df = self._enhanced_optimize_dataframe_memory(self.df)
            print_memory_usage("原始數據優化後")
        elif self.df is None or self.df.empty:
            # 如果執行到這裡 self.df 還是空，說明原始數據載入失敗
            raise RuntimeError("原始資料載入失敗，未能獲取 DataFrame。")

        # 載入完成後清理記憶體
        if self.aggressive_gc: clean_memory()

        return self.df # 返回載入並優化後的原始 DataFrame (如果需要)

    # --- 數據載入輔助方法 (_load_with_polars, _load_with_incremental_processing 等) ---
    # --- 這些方法的實現可以保持與之前版本類似，確保它們最終設置 self.df ---
    # --- 為保持程式碼完整性，這裡包含這些方法的骨架或完整實現 ---

    def _load_with_polars(self, file_paths: List[str]):
        """使用 Polars 載入多個 CSV 或 Parquet 文件"""
        if not POLARS_AVAILABLE:
            logger.warning("Polars 未安裝，回退到 Pandas 增量處理。")
            return self._load_with_incremental_processing(file_paths)

        logger.info(f"使用 Polars 載入 {len(file_paths)} 個文件...")
        lazy_frames = []
        for file_path in tqdm(file_paths, desc="使用 Polars 讀取文件"):
            try:
                if file_path.endswith('.csv'):
                    # 嘗試自動推斷分隔符和類型
                    lf = pl.scan_csv(file_path, try_parse_dates=True, ignore_errors=True) # ignore_errors 處理潛在的解析問題
                elif file_path.endswith('.parquet'):
                    lf = pl.scan_parquet(file_path)
                else:
                    logger.warning(f"不支持的文件類型: {file_path}，已跳過。")
                    continue
                lazy_frames.append(lf)
            except Exception as e:
                 logger.error(f"使用 Polars 讀取文件 {file_path} 時出錯: {e}", exc_info=True)

        if not lazy_frames:
             raise ValueError("未能使用 Polars 成功讀取任何文件。")

        try:
            logger.info("合併 Polars LazyFrames...")
            # 合併所有 LazyFrames
            combined_lf = pl.concat(lazy_frames)
            # 執行計算並轉換為 Pandas DataFrame
            logger.info("執行 Polars 計算並轉換為 Pandas DataFrame...")
            self.df = combined_lf.collect().to_pandas()
            logger.info(f"Polars 載入完成，DataFrame 形狀: {self.df.shape}")
            print_memory_usage("Polars 載入後")
        except Exception as e:
             logger.error(f"合併 Polars LazyFrames 或轉換為 Pandas 時出錯: {e}", exc_info=True)
             # 回退到 Pandas 處理
             logger.warning("Polars 處理失敗，回退到 Pandas 增量處理。")
             self.df = None # 清空可能不完整的 df
             self._load_with_incremental_processing(file_paths)


    def _load_single_file_with_polars(self, file_path: str):
        """使用 Polars 載入單個 CSV 或 Parquet 文件"""
        if not POLARS_AVAILABLE:
            logger.warning("Polars 未安裝，回退到 Pandas 處理。")
            return self._load_single_file_incrementally(file_path)

        logger.info(f"使用 Polars 載入單個文件: {file_path}")
        try:
            if file_path.endswith('.csv'):
                lf = pl.scan_csv(file_path, try_parse_dates=True, ignore_errors=True)
            elif file_path.endswith('.parquet'):
                lf = pl.scan_parquet(file_path)
            else:
                raise ValueError(f"不支持的文件類型: {file_path}")

            self.df = lf.collect().to_pandas()
            logger.info(f"Polars 載入完成，DataFrame 形狀: {self.df.shape}")
            print_memory_usage("Polars 載入後")
        except Exception as e:
            logger.error(f"使用 Polars 讀取文件 {file_path} 時出錯: {e}", exc_info=True)
            logger.warning("Polars 處理失敗，回退到 Pandas 增量處理。")
            self.df = None
            self._load_single_file_incrementally(file_path)


    def _load_with_incremental_processing(self, file_paths: List[str]):
        """使用 Pandas 增量讀取和處理多個文件"""
        logger.info(f"使用 Pandas 增量載入 {len(file_paths)} 個文件...")
        all_chunks = []
        total_rows = 0
        engine = 'pyarrow' if PYARROW_AVAILABLE else 'python' # 選擇讀取引擎

        for file_path in tqdm(file_paths, desc="增量讀取文件"):
            try:
                # 計算基於 MB 的 chunksize
                avg_row_size_mb = 0.0001 # 假設一個較小的初始行大小 (MB)
                try:
                    # 嘗試讀取第一行估算大小
                     df_sample = pd.read_csv(file_path, nrows=1, engine=engine)
                     avg_row_size_mb = df_sample.memory_usage(deep=True).sum() / (1024 * 1024)
                except: pass # 如果讀取失敗，使用預設值

                chunksize_rows = int(self.chunk_size_mb / max(avg_row_size_mb, 0.00001)) # 計算行數
                chunksize_rows = max(10000, min(chunksize_rows, self.chunk_row_limit)) # 限制 chunksize 範圍

                logger.debug(f"文件 '{os.path.basename(file_path)}': 估算行大小 {avg_row_size_mb:.4f}MB, 使用 chunksize {chunksize_rows} 行")

                reader = pd.read_csv(file_path, chunksize=chunksize_rows, low_memory=False, engine=engine)
                for chunk in reader:
                    chunk_optimized = self._enhanced_optimize_dataframe_memory(chunk)
                    all_chunks.append(chunk_optimized)
                    total_rows += len(chunk)
                    if self.aggressive_gc: clean_memory() # 在處理每個 chunk 後清理
            except Exception as e:
                 logger.error(f"讀取或處理文件 {file_path} 時出錯: {e}", exc_info=True)
                 continue # 跳過錯誤的文件

        if not all_chunks:
            raise ValueError("未能成功讀取任何數據塊。")

        logger.info(f"正在合併 {len(all_chunks)} 個數據塊，總計 {total_rows} 行...")
        self.df = pd.concat(all_chunks, ignore_index=True)
        logger.info(f"Pandas 增量載入完成，DataFrame 形狀: {self.df.shape}")
        print_memory_usage("Pandas 增量載入後")
        # 釋放 chunks 列表記憶體
        del all_chunks
        gc.collect()


    def _load_single_file_incrementally(self, file_path: str):
        """使用 Pandas 增量讀取和處理單個文件"""
        logger.info(f"使用 Pandas 增量載入單個文件: {file_path}")
        # 與 _load_with_incremental_processing 類似，但只處理一個文件
        self._load_with_incremental_processing([file_path])


    def _load_data_at_once(self, file_paths: List[str]):
        """一次性載入所有指定文件 (可能消耗大量記憶體)"""
        logger.warning("執行一次性數據載入，對於大型數據集可能導致記憶體不足。")
        all_dfs = []
        engine = 'pyarrow' if PYARROW_AVAILABLE else 'python'
        for file_path in tqdm(file_paths, desc="一次性載入文件"):
            try:
                df_part = pd.read_csv(file_path, low_memory=False, engine=engine)
                all_dfs.append(df_part)
            except Exception as e:
                 logger.error(f"一次性讀取文件 {file_path} 時出錯: {e}", exc_info=True)
                 continue

        if not all_dfs:
            raise ValueError("未能成功讀取任何文件。")

        logger.info(f"正在合併 {len(all_dfs)} 個 DataFrames...")
        self.df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"一次性載入完成，DataFrame 形狀: {self.df.shape}")
        print_memory_usage("一次性載入後")
        del all_dfs
        gc.collect()

    # --- IP 地址處理 ---
    def _process_ip_addresses(self, df: pd.DataFrame) -> pd.DataFrame:
        """處理IP地址列，保留子網關係和結構信息 (已修正錯誤處理和類型)"""
        ip_columns = [col for col in df.columns if 'IP' in col or 'ip' in col.lower()]
        if not ip_columns:
            logger.info("未檢測到IP地址列，跳過 IP 地址處理。")
            return df

        logger.info(f"檢測到 {len(ip_columns)} 個可能的IP地址列: {ip_columns}")
        df_processed = df.copy() # 操作副本

        for col in ip_columns:
            if col not in df_processed.columns:
                 logger.warning(f"列 '{col}' 在 DataFrame 中不存在，無法處理。")
                 continue

            # 檢查樣本以確認是否為 IP 列
            is_ip_col = False
            try:
                sample_values = df_processed[col].dropna().astype(str)
                if not sample_values.empty:
                    check_limit = min(100, len(sample_values))
                    valid_ip_count = sum(1 for val in sample_values.head(check_limit) if self._is_valid_ip(val))
                    if check_limit > 0 and valid_ip_count / check_limit > 0.5:
                        is_ip_col = True
            except Exception as e:
                logger.warning(f"檢查列 '{col}' 是否為IP列時出錯: {str(e)}")

            if not is_ip_col:
                logger.info(f"列 '{col}' 非主要IP地址列，跳過處理。")
                continue

            logger.info(f"確認 '{col}' 為IP地址列，開始提取特徵...")

            # --- 創建新特徵列 ---
            octet_cols = [f"{col}_octet{i}" for i in range(1, 5)]
            subnet_cols = [f"{col}_subnet{mask}" for mask in [8, 16, 24]]
            bool_cols = [f"{col}_is_private", f"{col}_is_global"]
            new_cols = octet_cols + subnet_cols + bool_cols
            # 初始化為適當的空值
            for c in octet_cols + subnet_cols: df_processed[c] = -1
            for c in bool_cols: df_processed[c] = 0

            # --- 處理 IP ---
            processed_count = 0
            error_count = 0
            # 使用 apply 可能比迭代更快，但需要處理錯誤
            def process_ip_row(ip_str_orig):
                nonlocal processed_count, error_count # 允許修改外部計數器
                if pd.isna(ip_str_orig):
                    error_count += 1
                    return [-1]*4 + [-1]*3 + [0]*2 # 返回預設值列表

                ip_str = str(ip_str_orig).strip()
                try:
                    ip = ipaddress.ip_address(ip_str)
                    is_private = int(ip.is_private)
                    is_global = int(ip.is_global)
                    octets = [-1] * 4
                    subnets = [-1] * 3

                    if isinstance(ip, ipaddress.IPv4Address):
                        oct_parts = ip_str.split('.')
                        if len(oct_parts) == 4:
                           try:
                               octets = [int(p) for p in oct_parts]
                           except ValueError: # 如果某部分不是數字
                                pass # 保持 -1
                        for i, mask in enumerate([8, 16, 24]):
                            try:
                                network = ipaddress.IPv4Network(f"{ip_str}/{mask}", strict=False)
                                subnets[i] = int(network.network_address)
                            except ValueError: pass # 保持 -1

                    processed_count += 1
                    return octets + subnets + [is_private, is_global]
                except ValueError:
                    error_count += 1
                    return [-1]*4 + [-1]*3 + [0]*2
                except Exception as e:
                    error_count += 1
                    logger.debug(f"處理 IP '{ip_str}' 時發生非預期錯誤: {e}")
                    return [-1]*4 + [-1]*3 + [0]*2

            # 應用處理函數
            logger.info(f"使用 apply 函數處理列 '{col}'...")
            results = df_processed[col].progress_apply(process_ip_row) # 使用 tqdm 的 progress_apply
            logger.info(f"列 '{col}' apply 完成。")

            # 將結果合併回 DataFrame
            results_df = pd.DataFrame(results.tolist(), index=df_processed.index, columns=new_cols)
            for new_col in new_cols:
                 df_processed[new_col] = results_df[new_col]

            logger.info(f"列 '{col}' 處理完成: {processed_count} 個有效IP, {error_count} 個錯誤/無法解析")

            # --- 轉換數據類型 ---
            for octet_col in octet_cols:
                df_processed[octet_col] = pd.to_numeric(df_processed[octet_col], errors='coerce').fillna(-1).astype(np.int16)
            for subnet_col in subnet_cols:
                df_processed[subnet_col] = pd.to_numeric(df_processed[subnet_col], errors='coerce').fillna(-1).astype(np.int64)
            for bool_col in bool_cols:
                 df_processed[bool_col] = df_processed[bool_col].astype(np.int8)

            # --- 移除原始 IP 列 ---
            if col in df_processed.columns:
                 df_processed = df_processed.drop(columns=[col])
                 logger.info(f"已移除原始 IP 地址列 '{col}'")

            logger.info(f"IP 地址列 '{col}' 特徵提取與類型轉換完成")
            if self.aggressive_gc: clean_memory()

        return df_processed

    def _is_valid_ip(self, ip_str):
        """檢查字串是否為有效的 IP 地址"""
        try:
            ipaddress.ip_address(str(ip_str).strip())
            return True
        except ValueError:
            return False

    # --- 特徵選擇 ---
    def _perform_statistical_feature_selection(self, df: pd.DataFrame, target: Union[pd.Series, np.ndarray],
                                               exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """使用統計方法進行特徵選擇 (已修正 NaN/inf 處理)"""
        if exclude_cols is None: exclude_cols = []

        logger.info("執行基於統計的特徵選擇...")
        if df.empty:
             logger.warning("輸入 DataFrame 為空，跳過特徵選擇。")
             return df

        # 確保目標是 NumPy 數組且長度匹配
        if isinstance(target, pd.Series): target_array = target.values
        elif isinstance(target, np.ndarray): target_array = target
        else: target_array = np.array(target)

        if len(target_array) != len(df):
             logger.error(f"特徵選擇錯誤：特徵行數 ({len(df)}) 與目標行數 ({len(target_array)}) 不匹配！")
             return df

        df_copy = df.copy()
        original_cols = df_copy.columns.tolist()
        numeric_cols = df_copy.select_dtypes(include=np.number).columns.tolist()
        cols_to_process = list(set(numeric_cols) - set(exclude_cols))

        if not cols_to_process:
            logger.info("沒有數值特徵可供選擇（可能都被排除了）。")
            return df_copy

        # --- 統一處理 NaN/Inf ---
        logger.info("在特徵選擇前，使用中位數填充 NaN/Inf...")
        imputer = SimpleImputer(strategy='median')
        try:
            # 只對需要處理的列進行填充
            df_copy[cols_to_process] = imputer.fit_transform(df_copy[cols_to_process])
            logger.info("NaN/Inf 填充完成。")
        except ValueError as e:
             logger.error(f"填充 NaN/Inf 時發生錯誤 (可能所有值都是 NaN): {e}")
             # 如果填充失敗，可能無法進行後續選擇，返回副本
             return df_copy

        # --- 相關性分析 ---
        try:
            if len(cols_to_process) > 1:
                corr_matrix = df_copy[cols_to_process].corr().abs()
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
                high_corr_cols_to_drop = set()
                # 找出高度相關的特徵對 (>0.95)
                high_corr_indices = np.where(upper_tri > 0.95)
                if len(high_corr_indices[0]) > 0:
                     logger.warning(f"發現 {len(high_corr_indices[0])} 對高度相關 (>0.95) 特徵:")
                     for i, j in zip(*high_corr_indices):
                         col1 = upper_tri.columns[j]
                         col2 = upper_tri.index[i]
                         logger.debug(f"  - {col1} 和 {col2} (相關性: {upper_tri.iloc[i, j]:.4f})")
                         # 保留方差較大的特徵
                         var1 = df_copy[col1].var()
                         var2 = df_copy[col2].var()
                         if var1 >= var2: high_corr_cols_to_drop.add(col2)
                         else: high_corr_cols_to_drop.add(col1)

                     if high_corr_cols_to_drop:
                         logger.info(f"基於相關性分析，計劃移除 {len(high_corr_cols_to_drop)} 個高相關特徵。")
                         # 從待刪除列表中移除被排除的列
                         high_corr_cols_to_drop = list(high_corr_cols_to_drop - set(exclude_cols))
                         if high_corr_cols_to_drop:
                              logger.info(f"實際移除高相關特徵: {high_corr_cols_to_drop[:5]}...")
                              df_copy.drop(columns=high_corr_cols_to_drop, inplace=True)
                              # 更新待處理列列表
                              cols_to_process = [col for col in cols_to_process if col not in high_corr_cols_to_drop]
                         else:
                              logger.info("所有高相關特徵都在排除列表中，未移除任何特徵。")

        except Exception as e:
            logger.error(f"相關性分析過程中出錯: {e}", exc_info=True)


        # --- 低方差特徵移除 ---
        try:
            if cols_to_process: # 確保還有列需要處理
                variance_selector = VarianceThreshold(threshold=0.01)
                # fit 在已填充的數據上
                variance_selector.fit(df_copy[cols_to_process])
                selected_mask = variance_selector.get_support()
                low_var_cols = [col for is_selected, col in zip(selected_mask, cols_to_process) if not is_selected]

                if low_var_cols:
                    logger.info(f"移除 {len(low_var_cols)} 個低方差 (<0.01) 特徵: {low_var_cols[:5]}...")
                    df_copy.drop(columns=low_var_cols, inplace=True)
                    cols_to_process = [col for col in cols_to_process if col not in low_var_cols]
        except Exception as e:
            logger.error(f"執行低方差特徵過濾時出錯: {e}", exc_info=True)


        # --- 基於統計顯著性的特徵選擇 (ANOVA F-value & Mutual Information) ---
        try:
            if cols_to_process: # 確保還有列需要處理
                X_selection = df_copy[cols_to_process].values # 使用已填充的數據
                y_selection = target_array

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore") # 抑制可能的 UserWarning
                    try:
                        f_scores, p_values = f_classif(X_selection, y_selection)
                    except ValueError as ve: # 例如，如果 y 只有一個類別
                        logger.warning(f"計算 ANOVA F 分數失敗: {ve}。跳過基於 F 分數的選擇。")
                        f_scores, p_values = np.ones(len(cols_to_process)), np.ones(len(cols_to_process)) # 設置為無意義的值

                    try:
                         mi_scores = mutual_info_classif(X_selection, y_selection, random_state=self.random_state)
                    except Exception as mi_e:
                         logger.error(f"計算互信息失敗: {mi_e}。跳過基於 MI 的選擇。")
                         mi_scores = np.zeros(len(cols_to_process)) # 設置為 0

                feature_importance = pd.DataFrame({
                    'Feature': cols_to_process,
                    'F_Score': f_scores, 'P_Value': p_values, 'MI_Score': mi_scores
                }).fillna(0) # 填充 NaN

                logger.info("基於互信息(MI)的前20個重要特徵:")
                for i, row in feature_importance.sort_values('MI_Score', ascending=False).head(20).iterrows():
                    logger.info(f"  {i+1}. {row['Feature']}: MI={row['MI_Score']:.6f} (p={row['P_Value']:.3g})")

                # 根據 P 值和 MI 分數移除不顯著特徵
                # 條件：P 值 > 0.05 (不顯著) 且 MI 分數低於 20% 分位數
                mi_quantile_20 = feature_importance['MI_Score'].quantile(0.2)
                insignificant_features = feature_importance[
                    (feature_importance['P_Value'] > 0.05) & (feature_importance['MI_Score'] < mi_quantile_20)
                ]['Feature'].tolist()
                # 確保不移除被排除的列
                insignificant_features_final = list(set(insignificant_features) - set(exclude_cols))

                if insignificant_features_final:
                    logger.info(f"移除 {len(insignificant_features_final)} 個統計上不顯著的特徵: {insignificant_features_final[:5]}...")
                    df_copy.drop(columns=insignificant_features_final, inplace=True)
        except Exception as e:
            logger.error(f"執行基於統計顯著性的特徵選擇時出錯: {e}", exc_info=True)

        # --- 總結移除的特徵 ---
        final_cols = df_copy.columns.tolist()
        dropped_cols = list(set(original_cols) - set(final_cols))
        logger.info(f"特徵選擇完成，共移除 {len(dropped_cols)} 個特徵。剩餘特徵數: {df_copy.shape[1]}")
        if dropped_cols:
             logger.debug(f"移除的特徵列表: {dropped_cols}")

        return df_copy

    # --- 目標列識別 ---
    def _identify_target_column(self) -> str:
        """智能識別目標列 (增加配置優先級)"""
        # 1. 優先使用配置中指定的目標列
        if self.target_column_name:
             if self.target_column_name in self.df.columns:
                  logger.info(f"使用配置中指定的目標列: '{self.target_column_name}'")
                  return self.target_column_name
             else:
                  logger.warning(f"配置中指定的目標列 '{self.target_column_name}' 不存在於 DataFrame 中。將嘗試自動識別。")

        # 2. 檢查常見的標籤欄位名稱
        common_target_columns = [
            'Label', 'label', 'attack_type', 'Class', 'target', 'Target',
            'is_attack', 'is_malicious', 'Result', 'y', 'output', 'classification', 'category', 'type'
        ]
        for col in common_target_columns:
            if col in self.df.columns:
                logger.info(f"根據常見名稱匹配識別到的目標列: '{col}'")
                return col

        # 3. 嘗試通過列特徵識別目標列 (啟發式)
        logger.info("嘗試通過列特徵自動識別目標列...")
        candidate_cols = []
        potential_label_types = ['object', 'category', 'int8', 'int16', 'int32', 'int64']
        num_rows = len(self.df)
        if num_rows == 0: # 如果 DataFrame 為空，無法識別
             logger.error("DataFrame 為空，無法自動識別目標列。")
             raise ValueError("無法識別目標列，因為 DataFrame 為空。")

        for col in self.df.columns:
            # 排除明顯非標籤的列
            col_lower = col.lower()
            if any(skip in col_lower for skip in ['id', 'timestamp', 'date', 'time', 'ip', 'port', 'feature', 'duration', 'sequence', 'number']):
                continue

            col_data = self.df[col]
            col_type = str(col_data.dtype)

            if col_type in potential_label_types:
                try:
                    num_unique = col_data.nunique(dropna=True)
                    # 目標列通常唯一值數量有限，但又不是常數
                    if 1 < num_unique < min(1000, num_rows * 0.1): # 唯一值數量在 1 和 (1000 或 10%行數) 之間
                        score = 0
                        if col_type in ['object', 'category']: score += 5 # 類別型優先
                        elif 'int' in col_type:
                            val_range = col_data.max() - col_data.min() if pd.api.types.is_numeric_dtype(col_data) and not col_data.empty else float('inf')
                            if val_range < 50: score += 3
                            elif val_range < 200: score += 1
                        # 越靠後的列可能性越大？ (這個啟發式不一定可靠)
                        # position_score = list(self.df.columns).index(col) / len(self.df.columns)
                        # score += position_score * 0.5
                        candidate_cols.append((col, score))
                except Exception as e:
                     logger.debug(f"分析列 '{col}' 時出錯: {e}") # 記錄但不中斷

        if candidate_cols:
            candidate_cols.sort(key=lambda x: x[1], reverse=True)
            best_col, score = candidate_cols[0]
            logger.info(f"根據列特徵分析，最可能的目標列是: '{best_col}' (得分: {score:.2f})")
            return best_col

        # 4. 如果以上方法都失敗，回退到最後一列
        last_col = self.df.columns[-1]
        logger.warning(f"無法自動確定目標列，將使用最後一列 '{last_col}' 作為目標。強烈建議在配置中明確指定 'target_column'。")
        return last_col

    # --- 預處理主流程 ---
    @memory_usage_decorator
    def preprocess(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        執行完整的數據預處理流程。

        返回:
            (features_np, target_np): 預處理後的特徵和目標 NumPy 數組
        """
        logger.info("=" * 50)
        logger.info("開始數據預處理流程...")
        print_memory_usage("預處理開始")

        # 1. 確保數據已載入
        if self.df is None or self.df.empty:
            logger.info("DataFrame 為空，執行 load_data()...")
            self.load_data()
            if self.df is None or self.df.empty:
                raise ValueError("資料載入失敗，無法進行預處理。")
        logger.info(f"預處理開始時 DataFrame 形狀: {self.df.shape}")

        # 2. 資料採樣（如果啟用）
        if self.use_sampling:
            logger.info("步驟 2/11: 執行資料採樣...")
            self._sample_data()
            logger.info(f"採樣後 DataFrame 形狀: {self.df.shape}")
            print_memory_usage("資料採樣後")
        else:
            logger.info("步驟 2/11: 跳過資料採樣。")

        # 3. 清理列名
        logger.info("步驟 3/11: 清理列名...")
        original_columns = self.df.columns.tolist()
        self.df.columns = ["".join(c if c.isalnum() else '_' for c in str(x).strip()) for x in self.df.columns]
        renamed_columns = {old: new for old, new in zip(original_columns, self.df.columns) if old != new}
        if renamed_columns: logger.debug(f"重命名的列: {renamed_columns}")

        # 4. 處理缺失值和無限值
        logger.info("步驟 4/11: 處理缺失值和無限值...")
        nan_counts = self.df.isnull().sum()
        inf_counts = self.df.isin([np.inf, -np.inf]).sum()
        logger.info(f"處理前 NaN 總數: {nan_counts.sum()}, Inf 總數: {inf_counts.sum()}")
        # 使用中位數填充數值列的 NaN
        num_cols_to_impute = self.df.select_dtypes(include=np.number).columns
        imputer = SimpleImputer(strategy='median')
        if len(num_cols_to_impute) > 0:
             self.df[num_cols_to_impute] = imputer.fit_transform(self.df[num_cols_to_impute])
        # 將剩餘的 NaN (可能是對象類型) 填充為 'missing'
        self.df.fillna('missing', inplace=True)
        # 將 Inf 替換為 0 (或其他合適的值)
        self.df.replace([np.inf, -np.inf], 0, inplace=True)
        logger.info("缺失值和無限值處理完成。")
        print_memory_usage("處理缺失值後")

        # 5. 移除常數列
        logger.info("步驟 5/11: 移除常數列...")
        nunique = self.df.nunique(dropna=False) # 計算唯一值數量，包括 NaN ('missing')
        cols_to_drop = nunique[nunique <= 1].index.tolist()
        if cols_to_drop:
             self.df.drop(columns=cols_to_drop, inplace=True)
             logger.info(f"移除了 {len(cols_to_drop)} 個常數特徵: {cols_to_drop}")
        else:
             logger.info("未發現常數特徵。")

        # 6. 標籤識別與分離
        logger.info("步驟 6/11: 識別並分離目標標籤...")
        target_col = self._identify_target_column()
        if target_col not in self.df.columns:
            raise ValueError(f"未能找到目標列 '{target_col}'，預處理中止。")
        self.target = self.df[target_col].copy()
        logger.info(f"目標列 '{target_col}' 已分離。")
        # 暫不從 df 中移除目標列，後續步驟可能需要

        # 7. IP 地址特徵工程
        logger.info("步驟 7/11: 執行 IP 地址特徵工程...")
        self.df = self._process_ip_addresses(self.df)
        logger.info(f"IP 處理後 DataFrame 形狀: {self.df.shape}")
        print_memory_usage("IP 地址處理後")

        # 8. 提取特徵 DataFrame (現在可以移除目標列)
        logger.info("步驟 8/11: 提取特徵 DataFrame...")
        if target_col in self.df.columns:
             self.features = self.df.drop(columns=[target_col]).copy()
        else:
             self.features = self.df.copy() # 如果目標列已被移除
             logger.warning(f"目標列 '{target_col}' 在提取特徵前不存在，請檢查 IP 處理邏輯。")

        # 9. 標籤編碼
        logger.info("步驟 9/11: 對目標標籤進行編碼...")
        try:
            # 確保標籤是整數且從 0 開始
            if not pd.api.types.is_numeric_dtype(self.target) or self.target.min() < 0:
                 self.target = pd.Series(self.label_encoder.fit_transform(self.target.astype(str)), index=self.target.index)
                 logger.info("標籤已使用 LabelEncoder 編碼。")
                 logger.info(f"類別映射: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
            else:
                 # 檢查是否從 0 開始連續
                 unique_labels = sorted(self.target.unique())
                 if unique_labels[0] != 0 or not all(unique_labels[i] == i for i in range(len(unique_labels))):
                      logger.warning("數值標籤不是從0開始的連續整數，重新編碼。")
                      self.target = pd.Series(self.label_encoder.fit_transform(self.target), index=self.target.index)
                      logger.info(f"標籤重新編碼完成。類別映射: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
                 else:
                      logger.info("目標標籤已是從0開始的連續數值類型，無需編碼。")
            # 確保是整數類型
            self.target = self.target.astype(int)
        except Exception as e:
             logger.error(f"標籤編碼失敗: {e}", exc_info=True)
             raise

        # 10. 統計特徵選擇
        logger.info("步驟 10/11: 執行統計特徵選擇...")
        # 定義需要保留的關鍵列 (IP衍生的、時間相關的)
        preserved_columns = [col for col in self.features.columns if '_octet' in col or '_subnet' in col or '_is_private' in col or '_is_global' in col or 'time' in col.lower() or 'duration' in col.lower()]
        self.features = self._perform_statistical_feature_selection(
            self.features, self.target, exclude_cols=preserved_columns
        )
        self.feature_names = list(self.features.columns) # 更新特徵名稱列表
        logger.info(f"特徵選擇後形狀: {self.features.shape}")
        print_memory_usage("特徵選擇後")

        # 11. 處理剩餘非數值特徵並標準化
        logger.info("步驟 11/11: 處理剩餘類別特徵並進行標準化...")
        numeric_cols_final = []
        category_cols_final = []
        cols_to_drop_final = []

        for col in self.features.columns:
             if pd.api.types.is_numeric_dtype(self.features[col]):
                  numeric_cols_final.append(col)
             # 嘗試將 object/category 轉換為數值 (標籤編碼)
             elif pd.api.types.is_object_dtype(self.features[col]) or pd.api.types.is_categorical_dtype(self.features[col]):
                  try:
                       # 使用 factorize 進行標籤編碼，它能處理 'missing' 值
                       codes, _ = pd.factorize(self.features[col])
                       self.features[col] = codes
                       numeric_cols_final.append(col) # 編碼後視為數值
                       logger.debug(f"分類特徵 '{col}' 已進行標籤編碼。")
                  except Exception as e:
                       logger.warning(f"無法對分類特徵 '{col}' 進行標籤編碼: {e}。將移除此列。")
                       cols_to_drop_final.append(col)
             else:
                  logger.warning(f"發現未知類型的特徵 '{col}' ({self.features[col].dtype})，將移除此列。")
                  cols_to_drop_final.append(col)

        # 移除轉換失敗或未知類型的列
        if cols_to_drop_final:
             self.features.drop(columns=cols_to_drop_final, inplace=True)
             logger.info(f"移除了 {len(cols_to_drop_final)} 個無法處理的特徵: {cols_to_drop_final}")
             self.feature_names = list(self.features.columns) # 再次更新特徵名稱

        # --- 標準化所有數值特徵 ---
        if numeric_cols_final:
             # 從 numeric_cols_final 中移除已刪除的列
             numeric_cols_to_scale = [col for col in numeric_cols_final if col in self.features.columns]
             if numeric_cols_to_scale:
                  logger.info(f"對 {len(numeric_cols_to_scale)} 個數值特徵進行標準化...")
                  try:
                       # 再次確保沒有 NaN/Inf (雖然前面處理過，但以防萬一)
                       features_numeric = self.features[numeric_cols_to_scale].values.astype(np.float64) # 使用 float64 提高精度
                       if np.any(np.isnan(features_numeric)) or np.any(np.isinf(features_numeric)):
                            logger.warning("標準化前再次發現 NaN/Inf，使用 0 填充。")
                            features_numeric = np.nan_to_num(features_numeric, nan=0.0, posinf=0.0, neginf=0.0)

                       self.features[numeric_cols_to_scale] = self.scaler.fit_transform(features_numeric)
                       logger.info("數值特徵標準化完成。")
                  except Exception as e:
                       logger.error(f"標準化特徵時發生錯誤: {e}", exc_info=True)
                       raise # 拋出錯誤以停止執行
             else:
                  logger.warning("沒有剩餘的數值特徵需要標準化。")
        else:
             logger.warning("最終數據中沒有數值特徵進行標準化。")

        logger.info(f"預處理最終完成。特徵形狀: {self.features.shape}, 目標形狀: {self.target.shape}")
        print_memory_usage("標準化後")

        # --- 驗證行數一致性 ---
        if len(self.features) != len(self.target):
             raise ValueError(f"預處理後特徵和目標的行數不匹配: {len(self.features)} vs {len(self.target)}")

        # --- 轉換為最終的 NumPy Array ---
        logger.info("將特徵和目標轉換為 NumPy Arrays...")
        final_features_np = self.features.values.astype(np.float32) # 模型通常使用 float32
        final_target_np = self.target.values.astype(int) # 標籤是整數

        # 更新類別屬性
        self.features = final_features_np
        self.target = final_target_np

        # --- 保存預處理資料 ---
        if self.save_preprocessed:
            logger.info("保存預處理後的數據...")
            self._save_preprocessed_data()

        # --- 釋放記憶體 ---
        logger.info("釋放原始 DataFrame 記憶體...")
        self.df = None
        if self.aggressive_gc: clean_memory(aggressive=True)
        logger.info("預處理流程結束。")
        print_memory_usage("預處理結束")
        logger.info("=" * 50)

        return self.features, self.target # 返回 NumPy Arrays


    def _save_preprocessed_data(self):
        """保存預處理後的資料 (Numpy + Pickle)"""
        if self.features is None or self.target is None:
            logger.warning("沒有可保存的預處理資料（features 或 target 為空）。")
            return

        files = self._get_preprocessed_files()
        logger.info(f"開始保存預處理數據到目錄: {self.preprocessed_path}")

        try:
            # 保存 Numpy Arrays
            np.save(files['features'], self.features)
            logger.info(f"  - 特徵已保存至: {files['features']} (形狀: {self.features.shape})")
            np.save(files['target'], self.target)
            logger.info(f"  - 目標已保存至: {files['target']} (形狀: {self.target.shape})")

            # 保存 Pickle 對象
            with open(files['feature_names'], 'wb') as f: pickle.dump(self.feature_names, f)
            logger.info(f"  - 特徵名稱已保存至: {files['feature_names']}")
            with open(files['scaler'], 'wb') as f: pickle.dump(self.scaler, f)
            logger.info(f"  - Scaler 已保存至: {files['scaler']}")
            with open(files['label_encoder'], 'wb') as f: pickle.dump(self.label_encoder, f)
            logger.info(f"  - LabelEncoder 已保存至: {files['label_encoder']}")

            # 保存訓練/測試索引 (如果存在)
            if self.train_indices is not None and self.test_indices is not None:
                np.save(files['train_indices'], self.train_indices)
                np.save(files['test_indices'], self.test_indices)
                logger.info(f"  - 訓練/測試索引已保存至: {files['train_indices']} / {files['test_indices']}")

            logger.info("預處理數據保存成功。")

        except Exception as e:
            logger.error(f"保存預處理數據時發生嚴重錯誤: {str(e)}", exc_info=True)
            # 可以選擇刪除可能部分寫入的文件
            for f_path in files.values():
                 if os.path.exists(f_path):
                      try: os.remove(f_path)
                      except: pass


    def _load_preprocessed_data(self) -> bool:
        """加載預處理後的資料 (Numpy + Pickle)，並進行嚴格檢查"""
        logger.info(f"嘗試從以下路徑加載預處理資料: {self.preprocessed_path}")
        files = self._get_preprocessed_files()

        # --- 嚴格檢查所有必要文件 ---
        required_files = ['features', 'target', 'scaler', 'label_encoder', 'feature_names']
        missing = [name for name in required_files if not os.path.exists(files[name])]
        if missing:
            logger.warning(f"必要的預處理文件缺失: {', '.join(missing)}。無法從緩存加載，將重新預處理。")
            return False

        logger.info("所有必要的預處理文件存在，開始加載...")
        try:
            with track_memory_usage("加載預處理資料"):
                # --- 加載 Numpy Arrays ---
                # 使用 mmap_mode='r' 嘗試記憶體映射，如果文件很大且啟用該選項
                mmap_mode = 'r' if self.use_memory_mapping else None
                self.features = np.load(files['features'], mmap_mode=mmap_mode)
                logger.info(f"  - 特徵已加載，形狀: {self.features.shape}, 類型: {self.features.dtype}, 映射模式: {mmap_mode}")
                self.target = np.load(files['target']) # 標籤通常不大，不使用映射
                logger.info(f"  - 目標已加載，形狀: {self.target.shape}, 類型: {self.target.dtype}")

                # --- 加載 Pickle 對象 ---
                with open(files['feature_names'], 'rb') as f: self.feature_names = pickle.load(f)
                logger.info(f"  - 特徵名稱已加載 ({len(self.feature_names)} 個)")
                with open(files['scaler'], 'rb') as f: self.scaler = pickle.load(f)
                logger.info(f"  - Scaler 已加載 (均值範例: {self.scaler.mean_[:3]}...)")
                with open(files['label_encoder'], 'rb') as f: self.label_encoder = pickle.load(f)
                logger.info(f"  - LabelEncoder 已加載 (類別: {self.label_encoder.classes_[:5]}...)")

                # 驗證加載的對象
                if not isinstance(self.scaler, StandardScaler) or not hasattr(self.scaler, 'mean_'):
                     raise TypeError("加載的 Scaler 無效。")
                if not isinstance(self.label_encoder, LabelEncoder) or not hasattr(self.label_encoder, 'classes_'):
                     raise TypeError("加載的 LabelEncoder 無效。")
                if not isinstance(self.feature_names, list):
                    raise TypeError("加載的特徵名稱不是列表。")
                # 驗證特徵數量是否匹配
                if len(self.feature_names) != self.features.shape[1]:
                     raise ValueError(f"加載的特徵名稱數量 ({len(self.feature_names)}) 與特徵數據維度 ({self.features.shape[1]}) 不匹配！")

                # --- 加載訓練/測試索引 (可選) ---
                train_idx_path = files['train_indices']
                test_idx_path = files['test_indices']
                if os.path.exists(train_idx_path) and os.path.exists(test_idx_path):
                    logger.info("正在加載已保存的訓練/測試集索引...")
                    self.train_indices = np.load(train_idx_path)
                    self.test_indices = np.load(test_idx_path)
                    logger.info(f"  - 訓練索引: {self.train_indices.shape}, 測試索引: {self.test_indices.shape}")

                    # --- 使用索引拆分數據 ---
                    # 驗證索引是否有效
                    max_index = max(self.train_indices.max(), self.test_indices.max())
                    if max_index >= self.features.shape[0]:
                         raise IndexError(f"加載的索引 ({max_index}) 超出數據範圍 ({self.features.shape[0]})！")

                    self.X_train = self.features[self.train_indices]
                    self.X_test = self.features[self.test_indices]
                    self.y_train = self.target[self.train_indices]
                    self.y_test = self.target[self.test_indices]
                    logger.info(f"已使用加載的索引拆分數據。")
                    logger.info(f"  訓練集: X={self.X_train.shape}, y={self.y_train.shape}")
                    logger.info(f"  測試集: X={self.X_test.shape}, y={self.y_test.shape}")
                else:
                    logger.info("未找到訓練/測試集索引文件，將在需要時重新拆分。")

            logger.info("預處理資料加載成功。")
            self.df = None # 預處理載入成功，原始 df 不再需要
            if self.aggressive_gc: clean_memory()
            print_memory_usage("預處理資料載入後")
            return True # 返回 True 表示加載成功

        except FileNotFoundError as fnf_error:
             logger.error(f"加載預處理文件失敗：找不到文件 {fnf_error.filename}")
             self._reset_data_attributes() # 出錯時重置
             return False
        except (pickle.UnpicklingError, EOFError, ValueError, TypeError, IndexError) as load_error:
             logger.error(f"加載預處理文件時發生錯誤（文件可能已損壞或格式不兼容）: {load_error}", exc_info=True)
             self._reset_data_attributes() # 出錯時重置
             return False
        except Exception as e:
            logger.error(f"加載預處理資料時發生未知錯誤: {str(e)}", exc_info=True)
            self._reset_data_attributes() # 出錯時重置
            return False

    def _reset_data_attributes(self):
         """重置資料相關的屬性，通常在加載失敗後調用"""
         logger.warning("重置資料屬性為初始狀態。")
         self.df = None
         self.features = None
         self.target = None
         self.feature_names = None
         self.X_train = self.X_test = self.y_train = self.y_test = None
         self.train_indices = self.test_indices = None
         self.scaler = StandardScaler()
         self.label_encoder = LabelEncoder()
         gc.collect()

    # --- 記憶體優化 ---
    def _enhanced_optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """增強版DataFrame記憶體優化 (增加 float16 警告)"""
        if df is None or df.empty:
            logger.warning("傳入空的 DataFrame，無法進行記憶體優化。")
            return df
        start_mem = df.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"優化前 DataFrame 記憶體使用: {start_mem:.2f} MB")

        df_optimized = df.copy() # 操作副本

        for col in df_optimized.columns:
            col_type = df_optimized[col].dtype

            if pd.api.types.is_integer_dtype(col_type):
                # 處理整數類型
                try:
                    c_min = df_optimized[col].min()
                    c_max = df_optimized[col].max()
                    if pd.isna(c_min) or pd.isna(c_max): continue # 跳過全空的列

                    if c_min >= 0: # 嘗試無符號整數
                         if c_max < np.iinfo(np.uint8).max: df_optimized[col] = df_optimized[col].astype(np.uint8)
                         elif c_max < np.iinfo(np.uint16).max: df_optimized[col] = df_optimized[col].astype(np.uint16)
                         elif c_max < np.iinfo(np.uint32).max: df_optimized[col] = df_optimized[col].astype(np.uint32)
                         else: df_optimized[col] = df_optimized[col].astype(np.uint64)
                    else: # 有符號整數
                         if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max: df_optimized[col] = df_optimized[col].astype(np.int8)
                         elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max: df_optimized[col] = df_optimized[col].astype(np.int16)
                         elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max: df_optimized[col] = df_optimized[col].astype(np.int32)
                         else: df_optimized[col] = df_optimized[col].astype(np.int64)
                except Exception as e:
                     logger.warning(f"優化整數列 '{col}' 時出錯: {e}") # 可能包含非數值
            elif pd.api.types.is_float_dtype(col_type):
                 # 處理浮點數類型
                 try:
                      # 優先嘗試 float32
                      df_optimized[col] = pd.to_numeric(df_optimized[col], errors='coerce').astype(np.float32)
                      # 如果啟用 aggressive 且條件允許，嘗試 float16
                      if self.aggressive_dtypes:
                          # 檢查 NaN 和範圍
                          col_data_float32 = df_optimized[col]
                          if np.all(np.isfinite(col_data_float32.dropna())): # 檢查非 NaN 值
                                c_min = col_data_float32.min()
                                c_max = col_data_float32.max()
                                # float16 範圍檢查
                                if not pd.isna(c_min) and not pd.isna(c_max) and \
                                   c_min > np.finfo(np.float16).min and \
                                   c_max < np.finfo(np.float16).max:
                                      logger.warning(f"列 '{col}' 將積極轉換為 float16，可能損失精度。")
                                      df_optimized[col] = col_data_float32.astype(np.float16)
                 except Exception as e:
                      logger.warning(f"優化浮點數列 '{col}' 時出錯: {e}")
            elif pd.api.types.is_object_dtype(col_type):
                 # 處理對象類型，嘗試轉換為 category
                 try:
                      num_unique_values = df_optimized[col].nunique()
                      num_total_values = len(df_optimized[col])
                      # 經驗法則：唯一值比例小於 50% 時轉換為 category 可能節省記憶體
                      if num_total_values > 0 and num_unique_values / num_total_values < 0.5:
                           df_optimized[col] = df_optimized[col].astype('category')
                 except TypeError: # nunique 可能對混合類型報錯
                      logger.debug(f"列 '{col}' 包含混合類型，無法計算 nunique，跳過 category 優化。")
                 except Exception as e:
                      logger.warning(f"優化對象列 '{col}' 為 category 時出錯: {e}")

        end_mem = df_optimized.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"優化後 DataFrame 記憶體使用: {end_mem:.2f} MB")
        reduction = (start_mem - end_mem) / start_mem * 100 if start_mem > 0 else 0
        logger.info(f"記憶體使用減少: {reduction:.2f}%")

        if self.aggressive_gc:
            gc.collect()

        return df_optimized

    # --- 數據拆分 ---
    @memory_usage_decorator
    def split_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        拆分訓練和測試資料集 (修正版：處理已加載的預處理數據)

        返回:
            (X_train, X_test, y_train, y_test) 作為 NumPy 數組
        """
        # 1. 檢查是否已從預處理文件加載了拆分數據
        if self.train_indices is not None and self.test_indices is not None and \
           self.features is not None and self.target is not None:
            logger.info("檢測到已加載的訓練/測試索引和數據，直接使用。")
            # 確保返回的是 numpy array
            if self.X_train is None: # 可能只加載了索引，未賦值給 X/y
                 self.X_train = self.features[self.train_indices]
                 self.X_test = self.features[self.test_indices]
                 self.y_train = self.target[self.train_indices]
                 self.y_test = self.target[self.test_indices]
                 logger.info("已使用加載的索引生成 X_train/X_test/y_train/y_test。")

            # 驗證數據類型和形狀
            if not all(isinstance(arr, np.ndarray) for arr in [self.X_train, self.X_test, self.y_train, self.y_test]):
                 logger.warning("加載的拆分數據類型不完全是 NumPy Array，將嘗試轉換。")
                 self.X_train = np.array(self.X_train) if not isinstance(self.X_train, np.ndarray) else self.X_train
                 self.X_test = np.array(self.X_test) if not isinstance(self.X_test, np.ndarray) else self.X_test
                 self.y_train = np.array(self.y_train) if not isinstance(self.y_train, np.ndarray) else self.y_train
                 self.y_test = np.array(self.y_test) if not isinstance(self.y_test, np.ndarray) else self.y_test

            # 再次檢查形狀
            if self.X_train.shape[0] != self.y_train.shape[0] or self.X_test.shape[0] != self.y_test.shape[0]:
                 logger.error("加載的訓練或測試集特徵與標籤數量不匹配！")
                 # 可能需要重新拆分，這裡先拋出錯誤
                 raise ValueError("加載的拆分數據行數不匹配。")

            return self.X_train, self.X_test, self.y_train, self.y_test

        # 2. 如果尚未拆分，或者預處理數據未包含拆分信息
        logger.info("未找到預加載的拆分數據，執行數據拆分...")
        if self.features is None or self.target is None:
             logger.info("Features 或 Target 為空，嘗試執行 preprocess()...")
             self.preprocess() # preprocess 會返回 numpy arrays 並設置 self.features/target
             if self.features is None or self.target is None:
                  raise RuntimeError("預處理後未能獲取特徵或目標數據，無法拆分。")

        # 確保 features 和 target 是 NumPy 數組
        if not isinstance(self.features, np.ndarray): self.features = np.array(self.features)
        if not isinstance(self.target, np.ndarray): self.target = np.array(self.target)

        logger.info(f"拆分資料集: 特徵 {self.features.shape}, 目標 {self.target.shape}, 測試集比例 {self.test_size}")

        # 生成索引並拆分
        indices = np.arange(len(self.target))
        try:
            # 嘗試分層拆分
            self.train_indices, self.test_indices = train_test_split(
                indices,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=self.target # 使用 NumPy target array 進行分層
            )
            logger.info("成功執行分層拆分。")
        except ValueError as e:
             logger.warning(f"分層拆分失敗 ({e})，可能是由於某些類別樣本數不足。將嘗試非分層拆分。")
             self.train_indices, self.test_indices = train_test_split(
                indices,
                test_size=self.test_size,
                random_state=self.random_state
            )
             logger.info("已執行非分層拆分。")

        # 3. 使用索引獲取訓練集和測試集
        self.X_train = self.features[self.train_indices]
        self.X_test = self.features[self.test_indices]
        self.y_train = self.target[self.train_indices]
        self.y_test = self.target[self.test_indices]

        # 4. 保存索引（如果啟用了預處理保存）
        if self.save_preprocessed:
            files = self._get_preprocessed_files()
            try:
                np.save(files['train_indices'], self.train_indices)
                np.save(files['test_indices'], self.test_indices)
                logger.info("已保存訓練集和測試集索引到預處理目錄。")
            except Exception as e:
                 logger.error(f"保存訓練/測試索引時出錯: {e}")

        logger.info(f"數據拆分完成:")
        logger.info(f"  訓練集: X={self.X_train.shape}, y={self.y_train.shape}")
        logger.info(f"  測試集: X={self.X_test.shape}, y={self.y_test.shape}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    # --- 其他輔助方法 ---
    def get_attack_types(self) -> Dict[int, Any]:
        """取得攻擊類型（標籤編碼後的數字）到原始類別名稱的對應表"""
        if hasattr(self.label_encoder, 'classes_') and self.label_encoder.classes_ is not None:
            try:
                # 確保 classes_ 是 iterable
                classes_ = list(self.label_encoder.classes_)
                return {i: label for i, label in enumerate(classes_)}
            except Exception as e:
                 logger.error(f"從 LabelEncoder 獲取類別時出錯: {e}")
                 return {}
        else:
             logger.warning("LabelEncoder 未訓練或 classes_ 屬性不可用，無法獲取攻擊類型映射。")
             # 嘗試從 target 猜測（如果 target 已編碼）
             if self.target is not None:
                  try:
                       unique_numeric_labels = sorted(np.unique(self.target))
                       logger.warning("嘗試從 target 數據推斷類別標籤。")
                       return {i: f"推斷類別_{i}" for i in unique_numeric_labels}
                  except Exception:
                       return {}
             return {}

    def _sample_data(self):
        """對 self.df 進行採樣 (修正版：處理空 DataFrame 和分層採樣邊界情況)"""
        if self.df is None or self.df.empty:
            logger.warning("DataFrame 為空，無法進行採樣。")
            return

        label_column = self._identify_target_column()
        if label_column not in self.df.columns:
             logger.error(f"採樣錯誤：找不到目標列 '{label_column}'。跳過採樣。")
             return

        original_size = len(self.df)
        target_size = int(original_size * self.sampling_ratio)

        # 如果採樣比例無效或不需要採樣
        if not (0 < self.sampling_ratio < 1) or target_size >= original_size:
             logger.info(f"採樣比例 ({self.sampling_ratio}) 無效或無需採樣，跳過。")
             return

        logger.info(f"執行數據採樣: 原始大小={original_size}, 目標大小={target_size}, 策略='{self.sampling_strategy}'")

        sampled_indices = None
        if self.sampling_strategy == 'stratified':
            logger.info("嘗試分層採樣...")
            try:
                y_stratify = self.df[label_column]
                value_counts = y_stratify.value_counts()
                # 檢查是否有類別樣本數過少 (少於 2 無法分層)
                if (value_counts < 2).any():
                    logger.warning(f"檢測到樣本數少於 2 的類別: {value_counts[value_counts < 2].index.tolist()}。無法進行分層採樣，將使用隨機採樣。")
                    self.sampling_strategy = 'random' # 回退到隨機
                else:
                     # train_size 設置為目標比例
                     sampled_indices, _ = train_test_split(
                         self.df.index, # 使用索引進行拆分
                         train_size=self.sampling_ratio,
                         random_state=self.random_state,
                         stratify=y_stratify
                     )
                     logger.info(f"分層採樣完成，選中 {len(sampled_indices)} 個樣本。")

            except Exception as e:
                 logger.error(f"分層採樣過程中出錯: {e}。將使用隨機採樣。", exc_info=True)
                 self.sampling_strategy = 'random' # 出錯時回退

        # 如果需要隨機採樣 (包括從分層回退的情況)
        if self.sampling_strategy == 'random' or sampled_indices is None:
             logger.info("執行隨機採樣...")
             if target_size <= 0: # 避免 target_size 為 0 或負數
                  logger.warning(f"計算出的目標採樣大小 ({target_size}) 無效，跳過採樣。")
                  return
             sampled_indices = np.random.choice(self.df.index, size=target_size, replace=False)
             logger.info(f"隨機採樣完成，選中 {len(sampled_indices)} 個樣本。")

        # 使用選中的索引更新 DataFrame
        if sampled_indices is not None:
             self.df = self.df.loc[sampled_indices].copy() # 使用 .loc 和 .copy() 避免警告
             logger.info(f"採樣後 DataFrame 大小: {len(self.df)} 行")
             new_class_counts = self.df[label_column].value_counts()
             logger.info(f"採樣後類別分布:\n{new_class_counts}")
        else:
             logger.error("未能生成採樣索引，採樣失敗。")

        if self.aggressive_gc: gc.collect()


    @memory_usage_decorator
    def get_sample_batch(self, batch_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        獲取一個批次的樣本，通常用於測試或演示 (修正版，使用已拆分的測試集)

        參數:
            batch_size: 批次大小，如果為 None，則使用配置中的 batch_size

        返回:
            (batch_features, batch_labels, original_indices): 特徵、標籤和對應的原始數據索引 (如果可用)
        """
        current_batch_size = batch_size if batch_size is not None else self.batch_size

        # 1. 確保測試集數據已準備好
        if self.X_test is None or self.y_test is None:
            logger.info("測試集數據不可用，嘗試執行 split_data()...")
            try:
                self.split_data() # split_data 會處理 preprocess 的調用（如果需要）
            except Exception as e:
                 logger.error(f"準備測試數據時出錯: {e}", exc_info=True)
                 # 返回空數組表示失敗
                 return np.array([]), np.array([]), np.array([])

        if self.X_test is None or self.y_test is None or len(self.y_test) == 0:
            logger.warning("無法獲取樣本批次，測試數據不可用或為空。")
            return np.array([]), np.array([]), np.array([])

        # 2. 從測試集隨機採樣
        num_test_samples = len(self.y_test)
        actual_batch_size = min(current_batch_size, num_test_samples)
        if actual_batch_size <= 0:
             logger.warning(f"請求的批次大小 ({actual_batch_size}) 無效。")
             return np.array([]), np.array([]), np.array([])

        # 隨機選擇測試集內的索引
        random_indices_in_test_set = np.random.choice(num_test_samples, size=actual_batch_size, replace=False)

        # 3. 獲取對應的數據
        batch_features = self.X_test[random_indices_in_test_set]
        batch_labels = self.y_test[random_indices_in_test_set]

        # 4. 獲取這些樣本在原始數據中的索引（如果可用）
        if self.test_indices is not None:
            # 確保 test_indices 的長度與 X_test/y_test 匹配
            if len(self.test_indices) == num_test_samples:
                 original_indices = self.test_indices[random_indices_in_test_set]
            else:
                 logger.warning("測試索引長度與測試數據不匹配，無法提供原始索引。")
                 # 返回相對於測試集的索引作為替代
                 original_indices = random_indices_in_test_set
        else:
            # 如果沒有保存原始索引，返回相對於測試集的索引
            original_indices = random_indices_in_test_set
            logger.debug("未找到原始測試索引，返回相對於測試集的索引。")

        return batch_features, batch_labels, original_indices


    def build_graph(self, features: Optional[Union[np.ndarray, torch.Tensor]] = None,
                  target: Optional[Union[np.ndarray, torch.Tensor]] = None,
                  graph_creation_config: Optional[Dict[str, Any]] = None) -> Optional[dgl.DGLGraph]:
         """
         使用提供的特徵和目標數據建立一個靜態 DGL 圖。

         參數:
             features: 節點特徵 (NumPy array 或 Tensor, [num_nodes, feat_dim])。如果為 None，則使用 self.features。
             target: 節點標籤 (NumPy array 或 Tensor, [num_nodes])。如果為 None，則使用 self.target。
             graph_creation_config: 建立圖的特定配置 (例如 'method', 'k')，如果為 None，則使用預設值。

         返回:
             dgl.DGLGraph: 創建的圖 (包含 'feat' 和 'label' 數據) 或 None
         """
         # 獲取數據
         current_features = features if features is not None else self.features
         current_target = target if target is not None else self.target

         if current_features is None or current_target is None:
             logger.error("無法建立靜態圖，缺少特徵或目標數據。請先運行 preprocess()。")
             return None

         # 解析圖建立配置
         cfg = graph_creation_config or {}
         method = cfg.get('method', 'knn') # 預設使用 KNN
         k = int(cfg.get('k', 5))          # KNN 的鄰居數
         target_device = cfg.get('device', self.device) # 最終圖所在的設備

         n_nodes = current_features.shape[0]
         if n_nodes != len(current_target):
             logger.error(f"建立靜態圖錯誤：特徵行數 ({n_nodes}) 與目標數量 ({len(current_target)}) 不匹配！")
             return None

         logger.info(f"基於提供的數據建立靜態圖，方法: {method}，節點數: {n_nodes}")

         # 確保數據是 Tensor
         if isinstance(current_features, np.ndarray): features_tensor = torch.tensor(current_features, dtype=torch.float32)
         else: features_tensor = current_features.float() # 假設是 Tensor，確保 float32

         if isinstance(current_target, np.ndarray): target_tensor = torch.tensor(current_target, dtype=torch.long)
         else: target_tensor = current_target.long() # 假設是 Tensor，確保 long

         g = None
         if method == 'knn':
             from sklearn.neighbors import kneighbors_graph # 延遲導入
             try:
                 # 將特徵移到 CPU 進行 KNN 計算
                 features_cpu = features_tensor.cpu().numpy()
                 # 清理 NaN/Inf
                 if np.any(np.isnan(features_cpu)) or np.any(np.isinf(features_cpu)):
                     logger.warning("靜態圖建立 (KNN)：特徵包含 NaN/Inf，使用 0 填充。")
                     features_cpu = np.nan_to_num(features_cpu, nan=0.0, posinf=0.0, neginf=0.0)

                 logger.info(f"計算 KNN 圖 (k={k})...")
                 adj_matrix = kneighbors_graph(features_cpu, n_neighbors=k, mode='connectivity', include_self=False)
                 src, dst = adj_matrix.nonzero()
                 logger.info(f"KNN 圖邊數: {len(src)}")

                 # 創建 DGL 圖 (先在 CPU)
                 g = dgl.graph((src, dst), num_nodes=n_nodes, device='cpu')

             except ImportError:
                 logger.error("建立 KNN 圖需要安裝 scikit-learn。請運行 'pip install scikit-learn'。")
                 return None
             except Exception as e:
                 logger.error(f"建立 KNN 圖時出錯: {e}", exc_info=True)
                 return None
         # 可以添加其他 edge_creation_method 的實現，例如基於閾值的相似度圖等
         else:
             logger.error(f"不支持的靜態圖邊創建方法: {method}")
             return None

         # 添加節點特徵和標籤到圖 (使用原始 Tensor，可能在 GPU 上)
         g.ndata['feat'] = features_tensor
         g.ndata['label'] = target_tensor

         logger.info(f"靜態圖建立完成: {g.num_nodes()} 節點, {g.num_edges()} 邊")
         # 將圖移到目標設備
         try:
             g = g.to(target_device)
             logger.info(f"靜態圖已移至設備: {g.device}")
         except Exception as e:
              logger.error(f"無法將靜態圖移至設備 {target_device}: {e}")
              # 可以選擇返回 CPU 上的圖或返回 None
              # return g.to('cpu')
              return None

         return g
