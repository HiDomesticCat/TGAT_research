#!/usr/bin/env python
# coding: utf-8 -*-

"""
時間圖神經網絡交叉驗證模組

提供針對圖數據的交叉驗證機制，設計用於時間圖神經網絡的訓練評估，
包含:
1. 時間感知圖數據分割
2. 時間序列交叉驗證
3. 子圖交叉驗證
4. 防止數據泄露的節點分割
"""

import torch
import numpy as np
import dgl
import logging
import time
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
import pandas as pd
from datetime import datetime
from copy import deepcopy
import matplotlib.pyplot as plt
import os
from ..utils.memory_utils import print_memory_usage, track_memory_usage, clean_memory

# 配置日誌
logger = logging.getLogger(__name__)

class GraphDataSplitter:
    """時間圖數據分割器
    
    針對圖數據的交叉驗證分割實現，特別考慮時間依賴性和圖結構。
    支持節點級和圖級的分割策略。
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化圖數據分割器
        
        參數:
            config: 配置字典，包含:
                - n_splits: 分割數量
                - split_type: 分割類型 ('node', 'graph', 'time')
                - shuffle: 是否打亂數據
                - test_size: 測試集比例
                - val_size: 驗證集比例
                - random_state: 隨機種子
                - time_aware: 是否考慮時間依賴性
                - memory_efficient: 是否使用記憶體高效模式
        """
        # 默認配置
        self.config = {
            'n_splits': 5,
            'split_type': 'node',
            'shuffle': True,
            'test_size': 0.2,
            'val_size': 0.1,
            'random_state': 42,
            'time_aware': True,
            'memory_efficient': True
        }
        
        # 更新配置
        if config:
            self.config.update(config)
        
        # 初始化隨機種子
        np.random.seed(self.config['random_state'])
        torch.manual_seed(self.config['random_state'])
        
        # 分割器類型
        self.split_type = self.config['split_type']
        self.n_splits = self.config['n_splits']
        self.memory_efficient = self.config['memory_efficient']
        
        logger.info(f"初始化圖數據分割器: 類型={self.split_type}, 分割數={self.n_splits}")
    
    def split_node_data(self, graph: dgl.DGLGraph, labels: torch.Tensor) -> List[Dict[str, Any]]:
        """
        節點級數據分割
        
        參數:
            graph: DGL圖
            labels: 節點標籤
            
        返回:
            List[Dict]: 每個分割的訓練/驗證/測試索引
        """
        n_nodes = graph.num_nodes()
        indices = np.arange(n_nodes)
        
        # 獲取基本配置
        n_splits = self.config['n_splits']
        shuffle = self.config['shuffle']
        random_state = self.config['random_state']
        
        # 是否需要分層抽樣
        stratify = labels.numpy() if shuffle else None
        
        # 創建分割器
        if stratify is not None:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            split_indices = list(kf.split(indices, stratify))
        else:
            kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            split_indices = list(kf.split(indices))
        
        # 準備分割結果
        splits = []
        for fold, (train_idx, test_idx) in enumerate(split_indices):
            # 進一步分割訓練集為訓練和驗證
            if self.config['val_size'] > 0:
                val_size = int(len(train_idx) * self.config['val_size'] / (1 - self.config['test_size']))
                if shuffle:
                    np.random.shuffle(train_idx)
                val_idx = train_idx[:val_size]
                train_idx = train_idx[val_size:]
            else:
                val_idx = test_idx  # 如果沒有驗證集，使用測試集作為驗證集
            
            splits.append({
                'fold': fold,
                'train_idx': train_idx,
                'val_idx': val_idx,
                'test_idx': test_idx
            })
            
            logger.debug(f"Fold {fold}: 訓練集={len(train_idx)}, 驗證集={len(val_idx)}, 測試集={len(test_idx)}")
        
        return splits
    
    def split_time_data(self, graph: dgl.DGLGraph, labels: torch.Tensor, timestamps: torch.Tensor) -> List[Dict[str, Any]]:
        """
        時間序列數據分割
        
        參數:
            graph: DGL圖
            labels: 節點標籤
            timestamps: 時間戳
            
        返回:
            List[Dict]: 每個分割的訓練/驗證/測試索引
        """
        n_nodes = graph.num_nodes()
        indices = np.arange(n_nodes)
        
        # 獲取基本配置
        n_splits = self.config['n_splits']
        
        # 確保timestamps是numpy數組
        if isinstance(timestamps, torch.Tensor):
            timestamps = timestamps.numpy()
        
        # 按時間排序索引
        sorted_indices = np.argsort(timestamps)
        
        # 創建時間序列分割器
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # 準備分割結果
        splits = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(sorted_indices)):
            # 獲取實際索引
            train_idx = sorted_indices[train_idx]
            test_idx = sorted_indices[test_idx]
            
            # 進一步分割訓練集為訓練和驗證
            if self.config['val_size'] > 0:
                # 對於時間序列，驗證集應該是訓練集的末尾部分
                val_size = int(len(train_idx) * self.config['val_size'])
                val_idx = train_idx[-val_size:]
                train_idx = train_idx[:-val_size]
            else:
                val_idx = test_idx  # 如果沒有驗證集，使用測試集作為驗證集
            
            splits.append({
                'fold': fold,
                'train_idx': train_idx,
                'val_idx': val_idx,
                'test_idx': test_idx
            })
            
            logger.debug(f"Fold {fold}: 訓練集={len(train_idx)}, 驗證集={len(val_idx)}, 測試集={len(test_idx)}")
        
        return splits
    
    def split_graph_data(self, graphs: List[dgl.DGLGraph], labels: torch.Tensor) -> List[Dict[str, Any]]:
        """
        圖級數據分割
        
        參數:
            graphs: 圖列表
            labels: 圖標籤
            
        返回:
            List[Dict]: 每個分割的訓練/驗證/測試索引
        """
        n_graphs = len(graphs)
        indices = np.arange(n_graphs)
        
        # 獲取基本配置
        n_splits = self.config['n_splits']
        shuffle = self.config['shuffle']
        random_state = self.config['random_state']
        
        # 是否需要分層抽樣
        stratify = labels.numpy() if shuffle else None
        
        # 創建分割器
        if stratify is not None:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            split_indices = list(kf.split(indices, stratify))
        else:
            kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            split_indices = list(kf.split(indices))
        
        # 準備分割結果
        splits = []
        for fold, (train_idx, test_idx) in enumerate(split_indices):
            # 進一步分割訓練集為訓練和驗證
            if self.config['val_size'] > 0:
                val_size = int(len(train_idx) * self.config['val_size'] / (1 - self.config['test_size']))
                if shuffle:
                    np.random.shuffle(train_idx)
                val_idx = train_idx[:val_size]
                train_idx = train_idx[val_size:]
            else:
                val_idx = test_idx  # 如果沒有驗證集，使用測試集作為驗證集
            
            splits.append({
                'fold': fold,
                'train_idx': train_idx,
                'val_idx': val_idx,
                'test_idx': test_idx
            })
            
            logger.debug(f"Fold {fold}: 訓練集={len(train_idx)}, 驗證集={len(val_idx)}, 測試集={len(test_idx)}")
        
        return splits
    
    def create_fold_subgraphs(self, graph: dgl.DGLGraph, node_indices: np.ndarray) -> dgl.DGLGraph:
        """
        根據節點索引創建子圖
        
        參數:
            graph: 原始DGL圖
            node_indices: 節點索引
            
        返回:
            dgl.DGLGraph: 子圖
        """
        try:
            # 轉換為張量
            if isinstance(node_indices, np.ndarray):
                node_indices = torch.from_numpy(node_indices)
            
            # 創建子圖
            subgraph = dgl.node_subgraph(graph, node_indices)
            
            # 確保子圖中至少有一條邊
            if subgraph.num_edges() == 0:
                logger.warning(f"子圖沒有邊，添加自環")
                subgraph = dgl.add_self_loop(subgraph)
            
            # 如果記憶體高效模式，則釋放不必要的內存
            if self.memory_efficient:
                clean_memory()
            
            return subgraph
        
        except Exception as e:
            logger.error(f"創建子圖時發生錯誤: {str(e)}")
            # 返回原始圖作為後備選項
            return graph
    
    def split(self, graph: Union[dgl.DGLGraph, List[dgl.DGLGraph]], 
              labels: torch.Tensor, 
              timestamps: Optional[torch.Tensor] = None,
              create_subgraphs: bool = True) -> List[Dict[str, Any]]:
        """
        主分割方法
        
        參數:
            graph: DGL圖或圖列表
            labels: 標籤
            timestamps: 時間戳，用於時間序列分割
            create_subgraphs: 是否創建子圖
            
        返回:
            List[Dict]: 分割結果列表
        """
        # 根據分割類型調用相應的方法
        if self.split_type == 'node':
            splits = self.split_node_data(graph, labels)
        elif self.split_type == 'time' and timestamps is not None:
            splits = self.split_time_data(graph, labels, timestamps)
        elif self.split_type == 'graph' and isinstance(graph, list):
            splits = self.split_graph_data(graph, labels)
        else:
            logger.warning(f"不支持的分割類型: {self.split_type}，使用節點分割")
            splits = self.split_node_data(graph, labels)
        
        # 根據需要創建子圖
        if create_subgraphs and not isinstance(graph, list):
            for i, split in enumerate(splits):
                # 創建訓練子圖
                train_graph = self.create_fold_subgraphs(graph, split['train_idx'])
                
                # 創建驗證子圖
                val_graph = self.create_fold_subgraphs(graph, split['val_idx'])
                
                # 創建測試子圖
                test_graph = self.create_fold_subgraphs(graph, split['test_idx'])
                
                # 添加子圖到分割結果
                splits[i]['train_graph'] = train_graph
                splits[i]['val_graph'] = val_graph
                splits[i]['test_graph'] = test_graph
                
                # 添加子圖的標籤
                splits[i]['train_labels'] = labels[split['train_idx']]
                splits[i]['val_labels'] = labels[split['val_idx']]
                splits[i]['test_labels'] = labels[split['test_idx']]
        
        return splits

class GraphCrossValidator:
    """圖數據交叉驗證器
    
    用於時間圖神經網路的交叉驗證，結合數據分割器和模型訓練評估。
    """
    
    def __init__(self, model_class, model_config: Dict[str, Any], 
                 cv_config: Dict[str, Any] = None,
                 device: str = 'cpu'):
        """
        初始化交叉驗證器
        
        參數:
            model_class: 模型類
            model_config: 模型配置
            cv_config: 交叉驗證配置
            device: 計算裝置
        """
        self.model_class = model_class
        self.model_config = model_config
        self.device = device
        
        # 默認交叉驗證配置
        self.cv_config = {
            'n_splits': 5,
            'split_type': 'node',
            'shuffle': True,
            'random_state': 42,
            'time_aware': True,
            'save_models': True,
            'save_dir': './cv_models',
            'metrics': ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'],
            'verbose': 1,
            'early_stopping': True,
            'patience': 10,
            'plot_results': True
        }
        
        # 更新配置
        if cv_config:
            self.cv_config.update(cv_config)
        
        # 創建數據分割器
        self.data_splitter = GraphDataSplitter({
            'n_splits': self.cv_config['n_splits'],
            'split_type': self.cv_config['split_type'],
            'shuffle': self.cv_config['shuffle'],
            'random_state': self.cv_config['random_state'],
            'time_aware': self.cv_config['time_aware']
        })
        
        # 創建保存目錄
        if self.cv_config['save_models']:
            os.makedirs(self.cv_config['save_dir'], exist_ok=True)
        
        logger.info(f"初始化圖交叉驗證器: 分割數={self.cv_config['n_splits']}, 裝置={device}")
    
    @track_memory_usage
    def train_test_fold(self, fold_data: Dict[str, Any], fold: int) -> Dict[str, Any]:
        """
        訓練並評估單個分割
        
        參數:
            fold_data: 分割數據
            fold: 分割索引
            
        返回:
            Dict: 評估結果
        """
        logger.info(f"訓練與評估 Fold {fold}")
        
        # 提取數據
        train_graph = fold_data.get('train_graph')
        val_graph = fold_data.get('val_graph')
        test_graph = fold_data.get('test_graph')
        
        train_labels = fold_data.get('train_labels')
        val_labels = fold_data.get('val_labels')
        test_labels = fold_data.get('test_labels')
        
        # 確保數據移到正確的設備
        train_labels = train_labels.to(self.device)
        val_labels = val_labels.to(self.device) if val_labels is not None else None
        test_labels = test_labels.to(self.device) if test_labels is not None else None
        
        # 創建模型實例
        model = self.model_class(self.model_config)
        model.to(self.device)
        
        # 訓練模型
        start_time = time.time()
        
        # 獲取早停參數
        early_stopping = self.cv_config.get('early_stopping', True)
        patience = self.cv_config.get('patience', 10)
        
        # 訓練模型
        history = model.train(
            train_graph=train_graph,
            train_labels=train_labels,
            val_graph=val_graph,
            val_labels=val_labels,
            epochs=self.model_config.get('epochs', 100),
            patience=patience if early_stopping else None
        )
        
        train_time = time.time() - start_time
        
        # 評估模型
        _, test_acc, test_metrics = model.evaluate(test_graph, test_labels)
        
        # 保存模型
        if self.cv_config['save_models']:
            model_path = os.path.join(self.cv_config['save_dir'], f"model_fold_{fold}.pt")
            model.save(model_path)
            logger.info(f"模型已保存至: {model_path}")
        
        # 準備結果
        results = {
            'fold': fold,
            'test_accuracy': test_acc,
            'train_time': train_time,
            'history': history,
            'metrics': test_metrics
        }
        
        # 釋放模型和數據的內存
        del model
        clean_memory()
        
        return results
    
    @track_memory_usage
    def cross_validate(self, graph: Union[dgl.DGLGraph, List[dgl.DGLGraph]], 
                       labels: torch.Tensor,
                       timestamps: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        執行交叉驗證
        
        參數:
            graph: DGL圖或圖列表
            labels: 標籤
            timestamps: 時間戳
            
        返回:
            Dict: 交叉驗證結果
        """
        logger.info("開始交叉驗證")
        
        # 分割數據
        splits = self.data_splitter.split(graph, labels, timestamps)
        
        # 初始化結果
        cv_results = {
            'fold_results': [],
            'mean_test_accuracy': 0.0,
            'std_test_accuracy': 0.0,
            'mean_train_time': 0.0,
            'std_train_time': 0.0,
            'metrics': {}
        }
        
        # 跟踪的指標
        metrics = self.cv_config['metrics']
        
        # 對每個分割進行訓練和評估
        for fold, split in enumerate(splits):
            # 訓練和評估
            fold_results = self.train_test_fold(split, fold)
            
            # 添加結果
            cv_results['fold_results'].append(fold_results)
        
        # 計算均值和標準差
        test_accuracies = [r['test_accuracy'] for r in cv_results['fold_results']]
        train_times = [r['train_time'] for r in cv_results['fold_results']]
        
        cv_results['mean_test_accuracy'] = np.mean(test_accuracies)
        cv_results['std_test_accuracy'] = np.std(test_accuracies)
        cv_results['mean_train_time'] = np.mean(train_times)
        cv_results['std_train_time'] = np.std(train_times)
        
        # 計算其他指標的均值和標準差
        for metric in metrics:
            if metric != 'accuracy':
                values = [r['metrics'].get(metric, 0.0) for r in cv_results['fold_results']]
                cv_results['metrics'][f'mean_{metric}'] = np.mean(values)
                cv_results['metrics'][f'std_{metric}'] = np.std(values)
        
        # 輸出結果
        logger.info("交叉驗證完成")
        logger.info(f"平均測試精度: {cv_results['mean_test_accuracy']:.4f} ± {cv_results['std_test_accuracy']:.4f}")
        logger.info(f"平均訓練時間: {cv_results['mean_train_time']:.2f}秒 ± {cv_results['std_train_time']:.2f}秒")
        
        # 繪製結果
        if self.cv_config['plot_results']:
            self.plot_cv_results(cv_results)
        
        return cv_results
    
    def train_final_model(self, graph: dgl.DGLGraph, labels: torch.Tensor) -> Any:
        """
        使用全部數據訓練最終模型
        
        參數:
            graph: DGL圖
            labels: 標籤
            
        返回:
            模型實例
        """
        logger.info("訓練最終模型（使用全部數據）")
        
        # 確保標籤移到正確的設備
        labels = labels.to(self.device)
        
        # 創建模型實例
        model = self.model_class(self.model_config)
        model.to(self.device)
        
        # 訓練模型
        history = model.train(
            train_graph=graph,
            train_labels=labels,
            val_graph=None,  # 沒有驗證集
            val_labels=None,
            epochs=self.model_config.get('final_epochs', self.model_config.get('epochs', 100)),
            patience=None  # 不使用早停
        )
        
        # 保存模型
        if self.cv_config['save_models']:
            model_path = os.path.join(self.cv_config['save_dir'], f"final_model.pt")
            model.save(model_path)
            logger.info(f"最終模型已保存至: {model_path}")
        
        return model
    
    def plot_cv_results(self, cv_results: Dict[str, Any]) -> None:
        """
        繪製交叉驗證結果
        
        參數:
            cv_results: 交叉驗證結果
        """
        # 創建圖表
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        
        # 提取數據
        folds = list(range(len(cv_results['fold_results'])))
        accuracies = [r['test_accuracy'] for r in cv_results['fold_results']]
        
        # 繪製準確率
        axs[0].bar(folds, accuracies, color='skyblue')
        axs[0].axhline(y=cv_results['mean_test_accuracy'], color='r', linestyle='-', label='平均值')
        axs[0].fill_between(
            folds, 
            cv_results['mean_test_accuracy'] - cv_results['std_test_accuracy'], 
            cv_results['mean_test_accuracy'] + cv_results['std_test_accuracy'], 
            color='r', alpha=0.2, label='標準差'
        )
        axs[0].set_xlabel('折')
        axs[0].set_ylabel('測試精度')
        axs[0].set_title('各折的測試精度')
        axs[0].set_xticks(folds)
        axs[0].set_xticklabels([f'折 {f}' for f in folds])
        axs[0].legend()
        
        # 選擇第一折的學習曲線作為示例
        if cv_results['fold_results'][0]['history']:
            history = cv_results['fold_results'][0]['history']
            epochs = list(range(1, len(history['val_accuracy']) + 1))
            
            # 繪製學習曲線
            axs[1].plot(epochs, history['val_accuracy'], 'b-', label='驗證精度')
            axs[1].plot(epochs, history['accuracy'], 'g-', label='訓練精度')
            axs[1].set_xlabel('輪次')
            axs[1].set_ylabel('精度')
            axs[1].set_title('學習曲線 (折 0)')
            axs[1].legend()
        
        plt.tight_layout()
        
        # 保存圖表
        if self.cv_config['save_models']:
            plot_path = os.path.join(self.cv_config['save_dir'], f"cv_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(plot_path)
            logger.info(f"交叉驗證結果圖表已保存至: {plot_path}")
        
        plt.close(fig)
