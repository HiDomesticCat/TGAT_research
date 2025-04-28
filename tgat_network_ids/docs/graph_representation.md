# 圖表示方法 (Graph Representation)

TGAT 網路入侵檢測系統支持多種圖表示方法，能夠根據數據規模和計算資源靈活選擇。本文檔說明系統中實現的圖表示方法、優缺點及使用場景。

## 1. 圖表示概述

圖表示是指如何將網路流量數據結構化為圖形式的方法，主要包含兩種基本表示法：

### 1.1 密集表示 (Dense Representation)

- **實現方式**：完整的鄰接矩陣 (Adjacency Matrix)
- **數據結構**：`n×n` 矩陣，其中 `n` 是節點數量
- **記憶體使用**：`O(n²)` 
- **適用場景**：節點數量較小的圖（<10,000 節點）

### 1.2 稀疏表示 (Sparse Representation)

- **實現方式**：鄰接列表 (Adjacency List) 或稀疏矩陣格式 (CSR/CSC/COO)
- **數據結構**：僅存儲非零元素和索引
- **記憶體使用**：`O(|E|)`，其中 `|E|` 是邊的數量
- **適用場景**：大型稀疏圖（節點間連接相對較少）

## 2. 圖表示實現細節

在系統實現中，我們提供了多種圖表示實現，可通過命令行參數和配置文件進行選擇：

### 2.1 鄰接矩陣表示

```python
# 密集表示（鄰接矩陣）
adj_matrix = torch.zeros((num_nodes, num_nodes))
for src, dst in edges:
    adj_matrix[src, dst] = 1
```

### 2.2 稀疏表示

```python
# 稀疏表示（COO格式）
edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
edge_attr = torch.ones(len(src_list))  # 邊特徵/權重
graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
```

### 2.3 分塊稀疏表示（針對超大型圖）

```python
# 分塊處理大型稀疏圖
chunks = []
for edge_batch in batch_iterator(edges, batch_size=10000):
    src_batch, dst_batch = zip(*edge_batch)
    chunks.append((torch.tensor(src_batch), torch.tensor(dst_batch)))

# 逐塊構建最終圖
final_graph = combine_graph_chunks(chunks, num_nodes)
```

## 3. 記憶體優化技術

為了進一步優化記憶體使用，系統實現了以下技術：

### 3.1 引用計數優化

- 自動追蹤節點及邊的引用情況
- 定期清理不再需要的記憶體

### 3.2 漸進式圖建構

- 隨時間窗口滑動增量建構圖結構
- 避免完整重建圖，顯著提升處理大型時序圖的效率

### 3.3 混合精度表示

- 使用較低精度的數據類型（如 `float16` 代替 `float32`）
- 可減少約 50% 的記憶體使用量

## 4. 在命令行中設定圖表示方法

實用的命令行參數組合：

```bash
# 使用稀疏表示（推薦用於大型圖）
--use_sparse_representation

# 配合進階採樣使用，進一步優化記憶體和計算效率
--use_sparse_representation --use_advanced_sampling --sampling_method graphsaint

# 最大程度記憶體優化（大型圖 + 記憶體受限環境）
--use_sparse_representation --use_mixed_precision --gradient_checkpointing
```

## 5. 性能比較

| 表示方法 | 10K節點 | 100K節點 | 1M節點 |
|---------|---------|----------|--------|
| 密集表示 | 0.4 GB  | 40 GB    | 4 TB   |
| 稀疏表示（平均10個連接/節點）| 0.01 GB | 0.1 GB | 1 GB |
| 分塊稀疏表示 | 0.01 GB | 0.08 GB | 0.8 GB |

## 6. 推薦與最佳實踐

1. **中小型圖** (<10K節點)：可使用密集或稀疏表示，密集表示訪問更快
2. **大型圖** (10K-100K節點)：推薦使用稀疏表示
3. **超大型圖** (>100K節點)：必須使用稀疏表示，並結合進階採樣技術

記憶體受限環境下的建議配置：

```yaml
graph:
  use_sparse_representation: true
  use_subgraph_sampling: true
  max_nodes_per_subgraph: 5000
  edge_batch_size: 5000
  prune_inactive_nodes: true
  inactive_threshold: 1800
```
