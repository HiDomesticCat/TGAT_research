#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
資料載入與預處理模組

此模組負責載入 CICDDoS 資料集並進行預處理，包括：
1. 資料清洗
2. 特徵選擇與工程
3. 資料標準化
4. 時間戳記處理
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CICDDoSDataLoader:
    """CICDDoS 資料集載入與預處理類別"""
    
    def __init__(self, data_path, test_size=0.2, random_state=42):
        """
        初始化資料載入器
        
        參數:
            data_path (str): 資料集路徑
            test_size (float): 測試集比例
            random_state (int): 隨機種子
        """
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.df = None
        self.features = None
        self.target = None
        self.feature_names = None
        
    def load_data(self):
        """載入資料集"""
        logger.info(f"載入資料集從: {self.data_path}")
        
        # 檢查是csv檔案還是目錄
        if os.path.isdir(self.data_path):
            # 載入目錄中的所有CSV檔案
            all_files = [os.path.join(self.data_path, f) 
                        for f in os.listdir(self.data_path) 
                        if f.endswith('.csv')]
            
            df_list = []
            for file in all_files:
                try:
                    df = pd.read_csv(file, low_memory=False)
                    df_list.append(df)
                    logger.info(f"成功載入資料: {file}, 形狀: {df.shape}")
                except Exception as e:
                    logger.error(f"載入 {file} 時發生錯誤: {str(e)}")
            
            if df_list:
                self.df = pd.concat(df_list, ignore_index=True)
                logger.info(f"合併資料集形狀: {self.df.shape}")
            else:
                raise ValueError("未找到有效的CSV檔案")
        else:
            # 載入單一CSV檔案
            self.df = pd.read_csv(self.data_path, low_memory=False)
            logger.info(f"載入資料集形狀: {self.df.shape}")
        
        return self.df
    
    def preprocess(self):
        """預處理資料集"""
        if self.df is None:
            self.load_data()
        
        logger.info("開始資料預處理...")
        
        # 清理列名，移除前後空格
        self.df.columns = [col.strip() for col in self.df.columns]

        # 檢查標籤欄位名稱
        if 'Label' in self.df.columns:
            label_column = 'Label'
        elif 'label' in self.df.columns:
            label_column = 'label'
        else:
            # 如果還是找不到標籤欄位，查找包含 "label" 的欄位
            possible_label_cols = [col for col in self.df.columns if 'label' in col.lower()]
            if possible_label_cols:
                label_column = possible_label_cols[0]
                logger.info(f"使用可能的標籤欄位: {label_column}")
            # 如果找不到相關欄位，嘗試使用最後一欄（在您的資料中是 'Label'）
            elif len(self.df.columns) > 0:
                label_column = self.df.columns[-1]
                logger.info(f"使用最後一欄作為標籤: {label_column}")
            else:
                # 如果上述方法都失敗，拋出異常
                raise ValueError("找不到標籤欄位，請檢查資料格式")
        
        # 處理缺失值
        logger.info("處理缺失值...")
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        self.df = self.df.fillna(0)
        
        # 檢查並移除常數特徵
        nunique = self.df.nunique()
        cols_to_drop = nunique[nunique == 1].index
        self.df = self.df.drop(columns=cols_to_drop)
        logger.info(f"移除 {len(cols_to_drop)} 個常數特徵")
        
        # 處理分類特徵
        logger.info("編碼分類特徵...")
        # 假設 'Protocol' 是分類特徵
        if 'Protocol' in self.df.columns:
            self.df = pd.get_dummies(self.df, columns=['Protocol'], drop_first=True)
        
        # 時間戳記處理
        timestamp_cols = [col for col in self.df.columns if 'time' in col.lower()]
        if timestamp_cols:
            logger.info(f"處理時間特徵: {timestamp_cols}")
            for col in timestamp_cols:
                # 如果是時間戳記欄位，轉換為UNIX時間戳 (秒)
                try:
                    self.df[col] = pd.to_datetime(self.df[col]).astype(int) / 10**9
                except:
                    # 如果已經是數值，則不需轉換
                    pass
        
        # 標籤編碼
        logger.info("編碼標籤...")
        # 將攻擊類型 (BENIGN, DoS, DDoS 等) 轉為數字編碼
        self.df[label_column] = self.label_encoder.fit_transform(self.df[label_column])
        
        # 分離特徵和標籤
        logger.info("分離特徵和標籤...")
        # 清理所有欄位名稱，移除前後空格
        self.df.columns = [col.strip() for col in self.df.columns]

        # 排除 IP 地址、時間戳記等不適合用作特徵的欄位
        # 使用部分匹配來識別欄位
        excluded_patterns = ['flow id', 'ip', 'source ip', 'destination ip', 'src ip', 'dst ip', 'timestamp', 'unnamed']
        excluded_cols = []

        for col in self.df.columns:
            # 檢查是否含有任何排除模式
            if any(pattern in col.lower() for pattern in excluded_patterns):
                excluded_cols.append(col)
            # 檢查是否為物件類型（通常表示非數值）
            elif self.df[col].dtype == 'object':
                excluded_cols.append(col)

        logger.info(f"排除以下欄位: {excluded_cols}")

        # 獲取特徵欄位
        feature_cols = [col for col in self.df.columns 
                        if col != label_column and col not in excluded_cols]
        
        self.feature_names = feature_cols
        self.features = self.df[feature_cols]
        self.target = self.df[label_column]
        
        # 特徵標準化
        logger.info("特徵標準化...")
        self.features = self.scaler.fit_transform(self.features)
        
        logger.info("資料預處理完成")
        return self.features, self.target
    
    def split_data(self):
        """拆分訓練和測試資料集"""
        if self.features is None or self.target is None:
            self.preprocess()
        
        logger.info(f"拆分資料集: 測試集比例 {self.test_size}")
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=self.target  # 確保標籤分布一致
        )
        
        return X_train, X_test, y_train, y_test
    
    def get_attack_types(self):
        """取得攻擊類型對應表"""
        return {i: label for i, label in enumerate(self.label_encoder.classes_)}
    
    def get_temporal_edges(self):
        """
        建立時間性邊緣關係
        
        為了構建圖結構，我們需要定義節點間的連接關係。
        在網路流量中，可以基於以下關係建立連接:
        - 相同來源 IP 的封包
        - 相同目標 IP 的封包
        - 相同 (來源 IP, 目標 IP) 對的封包
        - 時間接近的封包
        
        返回:
            list of tuples: (source_idx, destination_idx, timestamp, edge_features)
        """
        # 重新載入原始資料以獲取 IP 位址和時間戳記資訊
        if self.df is None:
            self.load_data()
        
        edges = []
        
        # 獲取原始 IP 和時間戳記欄位
        ip_src_col = 'Src IP' if 'Src IP' in self.df.columns else 'Source IP'
        ip_dst_col = 'Dst IP' if 'Dst IP' in self.df.columns else 'Destination IP'
        time_col = 'Timestamp' if 'Timestamp' in self.df.columns else 'Flow Start Time'
        
        # 將時間戳記轉換為數值
        if isinstance(self.df[time_col].iloc[0], str):
            self.df[time_col] = pd.to_datetime(self.df[time_col]).astype(int) / 10**9
        
        # 為每個封包分配索引 (節點ID)
        node_indices = {i: i for i in range(len(self.df))}
        
        # 根據 IP 對關係建立邊
        ip_pairs = self.df.groupby([ip_src_col, ip_dst_col]).indices
        
        # 使用閾值定義什麼是"時間接近"
        time_threshold = 1.0  # 1秒內的封包被視為時間接近
        
        for (src_ip, dst_ip), indices in ip_pairs.items():
            indices_list = sorted(indices, key=lambda x: self.df.iloc[x][time_col])
            
            # 對於相同 IP 對中的每對連續封包，建立時間性邊
            for i in range(len(indices_list) - 1):
                idx1 = indices_list[i]
                idx2 = indices_list[i + 1]
                
                time1 = self.df.iloc[idx1][time_col]
                time2 = self.df.iloc[idx2][time_col]
                
                # 時間差
                time_diff = time2 - time1
                
                # 如果時間接近，建立邊
                if time_diff <= time_threshold:
                    # 邊特徵: 時間差、封包大小比例等
                    size1 = self.df.iloc[idx1]['Fwd Pkt Len Mean'] if 'Fwd Pkt Len Mean' in self.df.columns else 0
                    size2 = self.df.iloc[idx2]['Fwd Pkt Len Mean'] if 'Fwd Pkt Len Mean' in self.df.columns else 0
                    
                    edge_feat = [time_diff, size2/size1 if size1 > 0 else 0]
                    
                    # 添加有向邊 (前 -> 後)
                    edges.append((node_indices[idx1], node_indices[idx2], time2, edge_feat))
        
        logger.info(f"建立了 {len(edges)} 條時間性邊")
        return edges
    
    def get_sample_batch(self, batch_size=1000):
        """
        獲取樣本批次，用於模擬動態圖更新
        
        參數:
            batch_size (int): 批次大小
            
        返回:
            tuple: (批次資料, 批次標籤)
        """
        if self.features is None or self.target is None:
            self.preprocess()
        
        total_samples = len(self.target)
        if batch_size > total_samples:
            batch_size = total_samples
        
        # 隨機抽樣
        indices = np.random.choice(total_samples, batch_size, replace=False)
        
        batch_features = self.features[indices]
        batch_labels = self.target.iloc[indices].values if isinstance(self.target, pd.Series) else self.target[indices]
        
        return batch_features, batch_labels, indices

# 測試資料載入器
if __name__ == "__main__":
    # 假設 CICDDoS2019 資料集位於當前目錄的 'data' 子目錄
    data_path = "./data"
    
    loader = CICDDoSDataLoader(data_path)
    
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
    edges = loader.get_temporal_edges()
    print(f"前 5 條時間性邊: {edges[:5]}")