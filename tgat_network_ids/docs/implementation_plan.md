# TGAT 網路入侵檢測系統完整實現方案

本文檔提供 TGAT 網路入侵檢測系統的完整實現方案，包括系統架構、文件結構、關鍵組件和實現步驟。

## 1. 系統架構圖

```
                            +-------------------------+
                            |      配置管理系統       |
                            | (memory_optimized_config.yaml) |
                            +------------+------------+
                                         |
                                         v
+----------------+        +---------------+---------------+         +---------------+
|   數據處理模塊  | -----> |          核心引擎            | <------- |  圖優化模塊   |
|   DataLoader   |        |       run_enhanced.py       |         | GraphOptimizer |
+----------------+        +---------------+---------------+         +---------------+
                                         |
                                         v
                   +-------------------+-+------------------+
                   |                   |                    |
          +--------v-------+  +--------v-------+   +-------v--------+
          |  模型訓練模塊  |  |  評估與預測模塊 |   | 視覺化與報告模塊 |
          | TrainingEngine |  | EvaluationEngine|   | VisualizationEngine|
          +----------------+  +----------------+   +------------------+
```

## 2. 文件結構與組件關係

```
tgat_network_ids/
├── config/
│   └── memory_optimized_config.yaml   # 配置文件
├── docs/
│   ├── usage_guide.md                 # 使用指南
│   ├── graph_representation.md        # 圖表示文檔
│   └── implementation_plan.md         # 實現計劃
├── src/
│   ├── data/
│   │   ├── optimized_data_loader.py   # 資料加載器
│   │   ├── optimized_graph_builder.py # 圖構建器
│   │   ├── advanced_sampling.py       # 進階圖採樣
│   │   ├── adaptive_window.py         # 自適應時間窗口
│   │   └── temporal_edge_list.py      # 時間邊列表 (新)
│   ├── models/
│   │   ├── optimized_tgat_model.py    # TGAT模型定義
│   │   ├── time_encoding.py           # 時間編碼
│   │   └── training_engine.py         # 訓練引擎 (新)
│   ├── utils/
│   │   ├── memory_utils.py            # 記憶體優化工具
│   │   ├── enhanced_metrics.py        # 增強指標計算
│   │   └── visualization.py           # 視覺化工具
│   └── evaluation/
│       └── evaluation_engine.py        # 評估引擎 (新)
└── scripts/
    ├── run_enhanced.py                # 主運行腳本
    └── run_optimized.py               # 優化版運行腳本
```

## 3. 關鍵組件實現詳情

### 3.1 AdvancedGraphSampler (src/data/advanced_sampling.py)

進階圖採樣策略的實現，支援多種採樣方法：

```python
import torch
import dgl
import numpy as np
import random
from collections import defaultdict

class AdvancedGraphSampler:
    """進階圖採樣類，實現多種採樣策略"""
    
    def __init__(self, method='graphsaint', sample_size=5000, seed=42):
        """
        初始化採樣器
        
        參數:
            method (str): 採樣方法，支援 'graphsaint', 'cluster-gcn', 'frontier', 'historical'
            sample_size (int): 子圖大小
            seed (int): 隨機種子
        """
        self.method = method
        self.sample_size = sample_size
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
    def sample_subgraph(self, full_graph):
        """
        從完整圖中採樣子圖
        
        參數:
            full_graph (dgl.DGLGraph): 完整圖
        
        返回:
            dgl.DGLGraph: 採樣的子圖
        """
        if self.method == 'graphsaint':
            return self._graphsaint_sampling(full_graph)
        elif self.method == 'cluster-gcn':
            return self._cluster_gcn_sampling(full_graph)
        elif self.method == 'frontier':
            return self._frontier_sampling(full_graph)
        elif self.method == 'historical':
            return self._historical_sampling(full_graph)
        else:
            raise ValueError(f"Unknown sampling method: {self.method}")
    
    def _graphsaint_sampling(self, full_graph):
        """GraphSAINT隨機遊走採樣"""
        num_nodes = full_graph.num_nodes()
        if num_nodes <= self.sample_size:
            return full_graph  # 如果圖小於採樣大小，則返回完整圖
        
        # 隨機選擇起始節點
        start_nodes = self.rng.choice(num_nodes, size=min(100, num_nodes//10), replace=False)
        
        # 從起始節點進行隨機遊走
        visited = set(start_nodes)
        frontier = list(start_nodes)
        
        while len(visited) < self.sample_size and frontier:
            current = frontier.pop(0)
            neighbors = full_graph.successors(current).numpy().tolist()
            
            # 隨機選擇未訪問的鄰居
            random.shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append(neighbor)
                    
                    if len(visited) >= self.sample_size:
                        break
        
        # 創建子圖
        nodes = list(visited)
        subgraph = full_graph.subgraph(nodes)
        return subgraph
        
    def _cluster_gcn_sampling(self, full_graph):
        """簡化版的Cluster-GCN採樣，使用節點度數聚類"""
        num_nodes = full_graph.num_nodes()
        if num_nodes <= self.sample_size:
            return full_graph
        
        # 使用度數作為聚類特徵
        degrees = full_graph.in_degrees() + full_graph.out_degrees()
        _, indices = torch.sort(degrees, descending=True)
        
        # 選擇高度數節點和周圍節點
        selected_nodes = indices[:self.sample_size // 10].numpy().tolist()
        visited = set(selected_nodes)
        frontier = list(selected_nodes)
        
        # BFS擴展
        while len(visited) < self.sample_size and frontier:
            current = frontier.pop(0)
            neighbors = full_graph.successors(current).numpy().tolist()
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append(neighbor)
                    
                    if len(visited) >= self.sample_size:
                        break
        
        # 創建子圖
        nodes = list(visited)
        subgraph = full_graph.subgraph(nodes)
        return subgraph
    
    def _frontier_sampling(self, full_graph):
        """多源隨機遊走子圖採樣"""
        num_nodes = full_graph.num_nodes()
        if num_nodes <= self.sample_size:
            return full_graph
            
        # 選擇多個起始節點
        num_seeds = min(20, self.sample_size // 50)
        seeds = self.rng.choice(num_nodes, size=num_seeds, replace=False)
        
        visited = set(seeds)
        frontiers = [[seed] for seed in seeds]
        
        # 從每個起始節點進行平行遊走
        while len(visited) < self.sample_size and any(frontiers):
            for i, frontier in enumerate(frontiers):
                if not frontier:
                    continue
                    
                # 從當前前沿中彈出一個節點
                node = frontier.pop(0)
                neighbors = full_graph.successors(node).numpy().tolist()
                
                # 添加未訪問的鄰居
                new_nodes = [n for n in neighbors if n not in visited]
                if new_nodes:
                    # 隨機選擇一個新鄰居
                    new_node = self.rng.choice(new_nodes)
                    visited.add(new_node)
                    frontier.append(new_node)
                    
                    if len(visited) >= self.sample_size:
                        break
        
        # 創建子圖
        nodes = list(visited)
        subgraph = full_graph.subgraph(nodes)
        return subgraph
    
    def _historical_sampling(self, full_graph):
        """基於時間信息的歷史採樣"""
        # 獲取邊的時間特徵
        edge_time = full_graph.edata.get('timestamp', None)
        
        if edge_time is None:
            # 如果沒有時間信息，退化為隨機採樣
            return self._graphsaint_sampling(full_graph)
        
        # 按時間排序邊
        edge_ids = torch.argsort(edge_time, descending=True)  # 降序，最近的邊優先
        recent_edges = edge_ids[:self.sample_size * 5]  # 選擇最近的一些邊
        
        # 獲取這些邊涉及的節點
        edge_list = full_graph.edges()
        src_nodes = edge_list[0][recent_edges]
        dst_nodes = edge_list[1][recent_edges]
        
        # 合併源節點和目標節點
        all_nodes = torch.cat([src_nodes, dst_nodes])
        unique_nodes = torch.unique(all_nodes)
        
        # 如果節點太多，再次採樣
        if len(unique_nodes) > self.sample_size:
            unique_nodes = unique_nodes[:self.sample_size]
        
        # 創建子圖
        subgraph = full_graph.subgraph(unique_nodes.tolist())
        return subgraph
```

### 3.2 TemporalEdgeList (src/data/temporal_edge_list.py)

高效的時間邊列表實現，用於時間窗口查詢：

```python
import bisect
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class TemporalEdgeList:
    """高效的時間邊列表實現，支援快速時間窗口查詢"""
    
    def __init__(self):
        # 按時間戳索引的邊列表
        self.time_indexed_edges = defaultdict(list)
        # 時間戳排序列表
        self.timestamps = []
        # 邊計數器
        self.edge_count = 0
        
    def add_edge(self, src, dst, timestamp, features=None):
        """添加一個時間邊"""
        if timestamp not in self.time_indexed_edges:
            bisect.insort(self.timestamps, timestamp)
        
        self.time_indexed_edges[timestamp].append((src, dst, features))
        self.edge_count += 1
        
        return self.edge_count
    
    def get_edges_in_window(self, start_time, end_time):
        """獲取時間窗口內的所有邊"""
        # 二分搜索找到時間範圍
        start_idx = bisect.bisect_left(self.timestamps, start_time)
        end_idx = bisect.bisect_right(self.timestamps, end_time)
        
        # 提取該時間範圍內的所有邊
        edges = []
        for i in range(start_idx, end_idx):
            timestamp = self.timestamps[i]
            edges.extend((src, dst, timestamp, features) 
                        for src, dst, features in self.time_indexed_edges[timestamp])
        
        logger.debug(f"Found {len(edges)} edges in time window [{start_time}, {end_time}]")
        return edges
    
    def get_recent_edges(self, time_window):
        """獲取最近的邊"""
        if not self.timestamps:
            return []
        
        latest_time = self.timestamps[-1]
        start_time = latest_time - time_window
        
        return self.get_edges_in_window(start_time, latest_time)
    
    def clear_old_edges(self, time_threshold):
        """清理舊邊以釋放記憶體"""
        if not self.timestamps:
            return 0
        
        # 找到閾值的位置
        idx = bisect.bisect_left(self.timestamps, time_threshold)
        
        # 如果沒有舊邊，直接返回
        if idx == 0:
            return 0
        
        # 記錄要清理的邊數量
        num_edges_to_clear = sum(len(self.time_indexed_edges[t]) for t in self.timestamps[:idx])
        
        # 清理舊邊
        for t in self.timestamps[:idx]:
            del self.time_indexed_edges[t]
        
        # 更新時間戳列表
        self.timestamps = self.timestamps[idx:]
        
        logger.info(f"Cleared {num_edges_to_clear} edges before timestamp {time_threshold}")
        return num_edges_to_clear
```

### 3.3 OptimizedGraphBuilder 擴展 (src/data/optimized_graph_builder.py)

增強圖構建器以支援稀疏表示和自適應時間窗口：

```python
def create_optimal_representation(self, nodes, edges, density_threshold=0.01):
    """
    根據圖密度選擇最佳表示方式
    
    參數:
        nodes: 節點列表
        edges: 邊列表
        density_threshold: 密度閾值，低於此值使用稀疏表示
    
    返回:
        graph: 構建的圖 (DGLGraph)
    """
    # 計算圖密度
    num_nodes = len(nodes)
    num_edges = len(edges)
    
    max_possible_edges = num_nodes * (num_nodes - 1)  # 有向圖
    density = num_edges / max_possible_edges if max_possible_edges > 0 else 0
    
    # 根據配置決定是否啟用自適應選擇
    use_adaptive = self.config.get('graph', {}).get('use_adaptive_representation', False)
    
    if use_adaptive:
        # 自適應選擇表示方式
        use_sparse = density < density_threshold
    else:
        # 根據配置固定選擇
        use_sparse = self.config.get('graph', {}).get('use_sparse_representation', True)
    
    # 根據選擇創建圖表示
    if use_sparse:
        logger.info(f"Using sparse representation for graph with {num_nodes} nodes and {num_edges} edges (density: {density:.6f})")
        return self._create_sparse_graph(nodes, edges)
    else:
        logger.info(f"Using dense representation for graph with {num_nodes} nodes and {num_edges} edges (density: {density:.6f})")
        return self._create_dense_graph(nodes, edges)
    
def _create_sparse_graph(self, nodes, edges):
    """
    使用稀疏表示創建圖
    
    參數:
        nodes: 節點列表
        edges: 邊列表
    
    返回:
        graph: 構建的稀疏圖 (DGLGraph)
    """
    if not edges:
        # 如果沒有邊，創建一個空圖
        graph = dgl.graph(([],  []), num_nodes=len(nodes))
        return graph
    
    # 提取源節點和目標節點
    src_nodes, dst_nodes = zip(*edges)
    src_tensor = torch.tensor(src_nodes)
    dst_tensor = torch.tensor(dst_nodes)
    
    # 創建稀疏圖
    graph = dgl.graph((src_tensor, dst_tensor), num_nodes=len(nodes))
    
    return graph
    
def _create_dense_graph(self, nodes, edges):
    """
    使用密集表示創建圖
    
    參數:
        nodes: 節點列表
        edges: 邊列表
    
    返回:
        graph: 構建的密集圖 (DGLGraph)
    """
    num_nodes = len(nodes)
    
    # 首先創建一個空的鄰接矩陣
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    
    # 填充鄰接矩陣
    for src, dst in edges:
        adj_matrix[src, dst] = 1.0
    
    # 從鄰接矩陣構建DGL圖
    graph = dgl.from_scipy(adj_matrix.to_sparse_csr().to_scipy())
    
    return graph
```

### 3.4 模型訓練引擎 (src/models/training_engine.py)

```python
import torch
import numpy as np
import logging
import time
from datetime import datetime
import os
import json

logger = logging.getLogger(__name__)

class TrainingEngine:
    """TGAT模型訓練引擎"""
    
    def __init__(self, model, optimizer, criterion, device, config):
        """初始化訓練引擎"""
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        
        # 訓練配置
        train_config = config.get('train', {})
        self.epochs = train_config.get('epochs', 50)
        self.patience = train_config.get('patience', 5)
        self.batch_size = train_config.get('batch_size', 512)
        self.use_dynamic_batch_size = train_config.get('use_dynamic_batch_size', False)
        self.memory_threshold = train_config.get('memory_threshold', 0.8)
        
        # 記錄最佳模型和訓練歷史
        self.best_model_state = None
        self.best_val_metrics = {'accuracy': 0.0}
        self.training_history = []
        
        # 混合精度訓練
        self.use_mixed_precision = config.get('model', {}).get('use_mixed_precision', False)
        if self.use_mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        
        # 輸出路徑
        output_config = config.get('output', {})
        self.model_dir = output_config.get('model_dir', './models')
        self.result_dir = output_config.get('result_dir', './results')
        
        # 創建目錄
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
    
    def train_one_epoch(self, data_loader, graph_builder):
        """訓練一個輪次"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 獲取訓練批次
        train_batches = data_loader.get_train_batches()
        
        for i, batch in enumerate(train_batches):
            # 提取特徵和標籤
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 構建批次圖
            batch_graph = graph_builder.build_batch_graph(batch)
            
            # 清除梯度
            self.optimizer.zero_grad()
            
            # 前向傳播 (混合精度或標準)
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_graph, features)
                    loss = self.criterion(outputs, labels)
                    
                # 反向傳播
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 標準訓練
                outputs = self.model(batch_graph, features)
                loss = self.criterion(outputs, labels)
                
                # 反向傳播
                loss.backward()
                self.optimizer.step()
            
            # 計算準確率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 累積損失
            total_loss += loss.item()
            
            # 日誌
            if (i + 1) % 10 == 0:
                logger.info(f'Batch [{i+1}/{len(train_batches)}], Loss: {loss.item():.4f}')
        
        # 計算平均損失和準確率
        avg_loss = total_loss / len(train_batches)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, data_loader, graph_builder):
        """驗證模型性能"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 獲取驗證批次
        val_batches = data_loader.get_val_batches()
        
        with torch.no_grad():
            for batch in val_batches:
                # 提取特徵和標籤
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 構建批次圖
                batch_graph = graph_builder.build_batch_graph(batch)
                
                # 前向傳播
                outputs = self.model(batch_graph, features)
                loss = self.criterion(outputs, labels)
                
                # 計算準確率
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 累積損失
                total_loss += loss.item()
        
        # 計算平均損失和準確率
        avg_loss = total_loss / len(val_batches)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, data_loader, graph_builder):
        """完整訓練流程"""
        logger.info(f"Starting training for {self.epochs} epochs...")
        start_time = time.time()
        
        best_val_accuracy = 0.0
        early_stop_counter = 0
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # 訓練一個輪次
            train_loss, train_acc = self.train_one_epoch(data_loader, graph_builder)
            
            # 驗證
            val_loss, val_acc = self.validate(data_loader, graph_builder)
            
            # 記錄訓練歷史
            epoch_time = time.time() - epoch_start
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'time': epoch_time
            })
            
            # 輸出日誌
            logger.info(f'Epoch [{epoch+1}/{self.epochs}], '
                        f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                        f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                        f'Time: {epoch_time:.2f}s')
            
            # 檢查是否為最佳模型
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                self.best_val_metrics = {
                    'accuracy': val_acc,
                    'loss': val_loss,
                    'epoch': epoch + 1
                }
                self.best_model_state = self.model.state_dict().copy()
                self._save_best_model()
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                
            # 早停
            if early_stop_counter >= self.patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
        
        # 訓練結束
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")
        
        # 保存訓練歷史
        self._save_training_history()
        
        return {
            'best_val_metrics': self.best_val_metrics,
            'training_history': self.training_history,
            'total_time': total_time
        }
    
    def _save_best_model(self):
        """保存最佳模型"""
        model_path = os.path.join(self.model_dir, 'best_model.pt')
        torch.save({
            'model_state_dict': self.best_model_state,
            'val_accuracy': self.best_val_metrics['accuracy'],
            'val_loss': self.best_val_metrics['loss'],
            'epoch': self.best_val_metrics['epoch']
        }, model_path)
        logger.info(f"Best model saved to {model_path}")
    
    def _save_training_history(self):
        """保存訓練歷史"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = os.path.join(self.result_dir, f'history_{timestamp}.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")
```

## 4. 實現步驟

### 4.1 第一階段: 基礎結構實現

1. 創建並實現 `advanced_sampling.py`
2. 創建並實現 `temporal_edge_list.py`
3. 擴展 `optimized_graph_builder.py` 以支持多種圖表示和採樣方法
4. 更新配置文件，添加相應的選項

### 4.2 第二階段: 訓練引擎實現

1. 創建並實現 `training_engine.py`
2. 創建並實現 `evaluation_engine.py`
3. 修改 `run_enhanced.py` 以使用新引擎

### 4.3 第三階段: 整合與測試

1. 整合所有組件
2. 添加記憶體優化和效率跟蹤功能
3. 添加全面的日誌記錄
4. 執行端到端測試
5. 優化性能和記憶體使用

### 4.4 第四階段: 添加高級功能

1. 實現混合精度訓練
2. 實現梯度累積
3. 添加早停和檢查點機制
4. 添加自動調整批次大小功能

## 5. 測試與驗證計劃

1. **功能測試**
   - 測試圖採樣功能
   - 測試時間邊列表功能
   - 測試模型訓練引擎

2. **性能測試**
   - 測量不同圖表示方法的記憶體使用
   - 比較不同採樣方法的計算效率
   - 評估訓練速度和收斂性能

3. **記憶體使用測試**
   - 監控大型圖處理期間的記憶體使用
   - 測試自動記憶體優化功能

## 6. 部署指南

1. **準備工作**
   - 安裝所需依賴: PyTorch, DGL, NumPy, NetworkX
   - 準備資料集
   - 設置配置文件

2. **執行命令**
   ```bash
   python scripts/run_enhanced.py --config config/memory_optimized_config.yaml --mode train --use_sparse_representation --use_mixed_precision
   ```

這個實現計劃概述了TGAT網路入侵檢測系統的主要組件和實現步驟。按照此計劃，我們可以有系統地開發和整合各個模組，創建一個高效、可擴展的入侵檢測系統。
