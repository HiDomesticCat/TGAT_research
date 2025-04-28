# TGAT 網路入侵檢測系統未來開發計劃

本文檔概述了TGAT網路入侵檢測系統的未來開發路線圖，重點是圖表示及其優化方面。

## 1. 實際訓練邏輯實現

當前版本的框架已經完成，但訓練邏輯只有骨架。下一步開發將實現：

### 1.1 數據處理和批次訓練

```python
# 計劃實現的批次訓練邏輯
def train_one_epoch(model, graph_builder, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    batch_count = 0
    
    # 從圖構建器獲取動態圖批次
    for batch_idx, (batch_graph, batch_features, batch_labels) in enumerate(graph_builder.generate_batches()):
        # 將數據移至設備
        batch_graph = batch_graph.to(device)
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        
        # 清除梯度
        optimizer.zero_grad()
        
        # 前向傳播
        outputs = model(batch_graph, batch_features)
        
        # 計算損失
        loss = criterion(outputs, batch_labels)
        
        # 反向傳播
        loss.backward()
        
        # 更新參數
        optimizer.step()
        
        # 累計損失
        total_loss += loss.item()
        batch_count += 1
    
    # 返回平均損失
    return total_loss / batch_count if batch_count > 0 else 0.0
```

### 1.2 進階圖採樣實現

使用各種採樣策略減少計算成本：

```python
class AdvancedGraphSampler:
    def __init__(self, method='graphsaint', sample_size=5000):
        self.method = method
        self.sample_size = sample_size
        
    def sample_subgraph(self, full_graph):
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
```

## 2. 稀疏表示優化

### 2.1 混合稀疏表示

為不同密度的圖結構實現自適應表示：

```python
def create_optimal_graph_representation(nodes, edges, density_threshold=0.01):
    """
    根據圖密度選擇最佳表示形式
    
    參數:
        nodes: 節點列表
        edges: 邊列表
        density_threshold: 密度閾值，低於此值使用稀疏表示
    
    返回:
        最佳圖表示對象
    """
    num_nodes = len(nodes)
    num_edges = len(edges)
    
    # 計算圖密度
    max_possible_edges = num_nodes * (num_nodes - 1)  # 有向圖
    density = num_edges / max_possible_edges if max_possible_edges > 0 else 0
    
    if density < density_threshold:
        # 稀疏表示 (CSR/COO 格式)
        return create_sparse_graph(nodes, edges)
    else:
        # 密集表示 (鄰接矩陣)
        return create_dense_graph(nodes, edges)
```

### 2.2 分塊處理和記憶體映射

對於超大圖，實現分塊處理和記憶體映射機制：

```python
# 分塊處理大型圖
def process_large_graph_in_chunks(edge_file, chunk_size=1000000):
    # 使用記憶體映射讀取邊文件
    with open(edge_file, 'r') as f:
        # 初始化稀疏矩陣構建器
        row_indices = []
        col_indices = []
        values = []
        
        # 分塊讀取邊
        while True:
            chunk = list(itertools.islice(f, chunk_size))
            if not chunk:
                break
                
            # 處理當前塊
            for line in chunk:
                src, dst, weight = parse_edge(line)
                row_indices.append(src)
                col_indices.append(dst)
                values.append(weight)
            
            # 定期清理以釋放記憶體
            if len(row_indices) > 10 * chunk_size:
                # 構建臨時稀疏矩陣並存儲
                temp_sparse_matrix = create_sparse_matrix(row_indices, col_indices, values)
                save_sparse_matrix_chunk(temp_sparse_matrix)
                
                # 清空緩衝
                row_indices = []
                col_indices = []
                values = []
                
        # 最終合併所有塊
        final_sparse_matrix = merge_sparse_matrix_chunks()
        return final_sparse_matrix
```

## 3. 時間圖優化

### 3.1 時間敏感的邊表示

為時間圖實現更高效的表示：

```python
class TemporalEdgeList:
    """
    高效的時間邊列表實現，支持快速時間窗口查詢
    """
    
    def __init__(self):
        # 按時間戳索引的邊列表
        self.time_indexed_edges = defaultdict(list)
        # 時間戳排序列表
        self.timestamps = []
        
    def add_edge(self, src, dst, timestamp, features=None):
        """添加一個時間邊"""
        if timestamp not in self.time_indexed_edges:
            bisect.insort(self.timestamps, timestamp)
        
        self.time_indexed_edges[timestamp].append((src, dst, features))
    
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
        
        return edges
```

## 4. 分佈式圖處理

為處理真正大規模的圖，計劃實現分佈式處理：

```python
# 分佈式圖處理架構（偽代碼）
class DistributedGraphProcessor:
    def __init__(self, num_partitions):
        self.num_partitions = num_partitions
        self.partitions = [GraphPartition(i) for i in range(num_partitions)]
        
    def partition_graph(self, full_graph):
        """將圖分割為多個分區"""
        # 使用 METIS 或類似算法進行圖分區
        partition_assignments = graph_partition_algorithm(full_graph, self.num_partitions)
        
        # 分配節點和邊到各分區
        for node_id, partition_id in enumerate(partition_assignments):
            self.partitions[partition_id].add_node(node_id)
            
        for src, dst in full_graph.edges():
            src_partition = partition_assignments[src]
            dst_partition = partition_assignments[dst]
            
            # 添加到源分區
            self.partitions[src_partition].add_edge(src, dst)
            
            # 如果跨分區，還需添加邊界信息
            if src_partition != dst_partition:
                self.partitions[src_partition].add_boundary_node(dst, dst_partition)
                self.partitions[dst_partition].add_boundary_node(src, src_partition)
    
    def distributed_processing(self, process_function):
        """在所有分區上並行執行處理函數"""
        # 可以使用多進程、分佈式計算框架等
        with ProcessPoolExecutor(max_workers=self.num_partitions) as executor:
            results = list(executor.map(process_function, self.partitions))
            
        return self.aggregate_results(results)
```

## 5. 實際訓練流程整合

將所有這些組件整合到完整的訓練流程：

```python
def complete_training_procedure(config, args):
    # 初始化數據加載器、圖構建器和模型
    data_loader = EnhancedMemoryOptimizedDataLoader(config)
    graph_builder = OptimizedGraphBuilder(config)
    model = OptimizedTGATModel(config)
    
    # 設置優化器
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=config.learning_rate, 
                                weight_decay=config.weight_decay)
    
    # 訓練循環
    for epoch in range(config.epochs):
        # 訓練一個輪次
        train_loss = train_one_epoch(model, graph_builder, optimizer, 
                                    torch.nn.CrossEntropyLoss(), config.device)
        
        # 驗證
        val_loss, val_metrics = validate_model(model, graph_builder, config.device)
        
        # 日誌
        logger.info(f"Epoch {epoch+1}/{config.epochs} - "
                   f"Train Loss: {train_loss:.4f}, "
                   f"Val Loss: {val_loss:.4f}, "
                   f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        
        # 保存檢查點
        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            save_checkpoint(model, optimizer, epoch, val_metrics, config.checkpoint_dir)
            
    # 最終評估
    test_metrics = evaluate_model(model, graph_builder, config.device)
    logger.info(f"Test Metrics: {test_metrics}")
    
    return model, test_metrics
```

這些開發計劃將使TGAT網路入侵檢測系統能夠有效處理各種規模的網路流量數據，從小型實驗環境到大型企業網絡。
