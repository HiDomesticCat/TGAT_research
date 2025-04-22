# TGAT 網路入侵檢測系統

本專案實現了一個基於時間圖注意力網絡（Temporal Graph Attention Network, TGAT）的網路入侵檢測系統。該系統使用圖神經網絡對網路流量進行建模，並檢測可能表示攻擊的異常模式。

## 快速開始

### 安裝

```bash
# 克隆儲存庫
git clone https://github.com/yourusername/TGAT_research.git
cd TGAT_research

# 安裝依賴
pip install -r tgat_network_ids/requirements.txt
```

### 運行系統

```bash
# 使用原始運行腳本
python TGAT_research/tgat_network_ids/scripts/run.py \
  --config TGAT_research/tgat_network_ids/config/memory_optimized_config.yaml \
  --mode train \
  --data_path ./data/your_data \
  --visualize \
  --monitor_memory

# 使用新的優化運行腳本
python TGAT_research/tgat_network_ids/scripts/run_optimized.py \
  --config TGAT_research/tgat_network_ids/config/memory_optimized_config.yaml \
  --mode train \
  --data_path ./data/your_data \
  --visualize \
  --monitor_memory \
  --use_sparse_representation \
  --use_mixed_precision

# 或使用 shell 腳本（從項目根目錄運行）
./TGAT_research/tgat_network_ids/scripts/run_memory_optimized.sh
```

## 使用指南

### 訓練模型

```bash
# 使用優化版運行腳本
python TGAT_research/tgat_network_ids/scripts/run_optimized.py \
  --config TGAT_research/tgat_network_ids/config/memory_optimized_config.yaml \
  --mode train \
  --data_path ./data/your_data \
  --visualize \
  --monitor_memory \
  --use_sparse_representation \
  --use_gradient_checkpointing
```

### 測試模型

```bash
python TGAT_research/tgat_network_ids/scripts/run_optimized.py \
  --config TGAT_research/tgat_network_ids/config/memory_optimized_config.yaml \
  --mode test \
  --model_path ./models/best_model.pt \
  --data_path ./data/your_data \
  --visualize \
  --use_sparse_representation
```

### 即時檢測

```bash
python TGAT_research/tgat_network_ids/scripts/run_optimized.py \
  --config TGAT_research/tgat_network_ids/config/memory_optimized_config.yaml \
  --mode detect \
  --model_path ./models/best_model.pt \
  --data_path ./data/your_data \
  --visualize \
  --use_mixed_precision
```

### 使用資料採樣功能

系統現在支持資料採樣功能，可以大幅減少訓練時間和記憶體使用量，同時保持模型性能。

#### 啟用分層採樣

分層採樣確保各類攻擊模式都有足夠的表示，即使在大幅減少資料量的情況下也能保持模型性能。

```yaml
# 在 config/memory_optimized_config.yaml 中設置
data:
  use_sampling: true              # 啟用資料採樣
  sampling_strategy: "stratified" # 使用分層採樣
  sampling_ratio: 0.1             # 使用10%的資料
  min_samples_per_class: 1000     # 每個類別至少保留1000個樣本
```

#### 使用隨機採樣

如果只需要快速測試系統功能，可以使用隨機採樣：

```yaml
data:
  use_sampling: true
  sampling_strategy: "random"     # 使用隨機採樣
  sampling_ratio: 0.05            # 使用5%的資料
```

#### 禁用採樣

如果需要使用全量資料進行訓練：

```yaml
data:
  use_sampling: false             # 禁用採樣，使用全量資料
```

## 配置

系統可通過 YAML 配置文件進行高度自定義。主要配置文件位於 `TGAT_research/tgat_network_ids/config/memory_optimized_config.yaml`。

### 主要配置選項

```yaml
# 數據配置
data:
  path: ./data/your_data  # 數據集路徑
  test_size: 0.2          # 用於測試的數據百分比
  batch_size: 128         # 訓練和評估的批次大小
  use_memory_mapping: true  # 使用記憶體映射
  save_preprocessed: true   # 保存預處理數據
  incremental_loading: true # 增量式數據加載
  use_polars: true          # 使用Polars加速資料處理

# 圖結構配置
graph:
  temporal_window: 300      # 時間窗口 (秒)
  use_subgraph_sampling: true  # 使用子圖採樣
  max_nodes_per_subgraph: 5000  # 子圖最大節點數
  max_edges_per_subgraph: 10000  # 子圖最大邊數
  use_sparse_representation: true  # 使用稀疏表示
  edge_batch_size: 5000      # 邊批次處理大小
  prune_inactive_nodes: true  # 清理不活躍節點
  inactive_threshold: 1800   # 不活躍閾值 (秒)
  use_block_sparse: true     # 使用分塊稀疏表示
  block_size: 128            # 分塊大小
  use_csr_format: true       # 使用CSR格式
  use_dgl_transform: true    # 使用DGL的transform API
  cache_neighbor_sampling: true  # 緩存鄰居採樣結果
  adaptive_pruning: true     # 自適應清理策略
  smart_memory_allocation: true  # 智能記憶體分配

# 模型配置
model:
  hidden_dim: 64          # 隱藏層維度
  out_dim: 64             # 輸出維度
  time_dim: 16            # 時間編碼維度
  num_layers: 2           # TGAT 層數
  num_heads: 4            # 注意力頭數
  dropout: 0.1            # 丟棄率
  use_mixed_precision: true  # 使用混合精度訓練
  use_gradient_accumulation: true  # 使用梯度累積
  gradient_accumulation_steps: 4   # 梯度累積步數
  use_gradient_checkpointing: true  # 使用梯度檢查點
  use_sparse_gradients: false  # 使用稀疏梯度

# 訓練配置
train:
  lr: 0.001               # 學習率
  weight_decay: 1e-5      # 權重衰減係數
  epochs: 100             # 最大訓練輪數
  patience: 10            # 早停耐心值
  use_dynamic_batch_size: true  # 根據記憶體動態調整批次大小
  memory_threshold: 0.8   # 批次大小調整的記憶體使用閾值
  use_progressive_training: true  # 使用漸進式訓練

# 檢測配置
detection:
  threshold: 0.7          # 攻擊檢測閾值
  window_size: 50         # 檢測窗口大小
  use_sliding_window: true  # 使用滑動窗口
  sliding_window_size: 100  # 滑動窗口大小
  sliding_window_step: 50   # 滑動窗口步長
```

## 開發指南

### 專案結構

```
TGAT_research/
└── tgat_network_ids/
    ├── config/                  # 配置文件
    ├── legacy_code/             # 原始實現（供參考）
    ├── scripts/                 # 運行腳本
    │   ├── run.py               # 原始運行腳本
    │   ├── run_optimized.py     # 新的優化運行腳本
    │   └── run_memory_optimized.sh  # Shell腳本
    ├── src/                     # 源代碼
    │   ├── data/                # 數據加載和處理
    │   │   ├── memory_optimized_data_loader.py    # 原始記憶體優化數據加載器
    │   │   ├── memory_optimized_graph_builder.py  # 原始記憶體優化圖構建器
    │   │   ├── optimized_data_loader.py           # 增強版優化數據加載器
    │   │   └── optimized_graph_builder.py         # 增強版優化圖構建器
    │   ├── models/              # 模型實現
    │   │   ├── tgat_model.py                # 原始TGAT模型
    │   │   ├── memory_optimized_train.py    # 記憶體優化訓練器
    │   │   └── optimized_tgat_model.py      # 優化版TGAT模型
    │   ├── utils/               # 工具函數
    │   │   ├── memory_utils.py  # 記憶體優化工具
    │   │   └── utils.py         # 一般工具函數
    │   └── visualization/       # 視覺化工具
    ├── SQL/                     # SQL 相關文件
    ├── README.md                # 本文件
    └── requirements.txt         # 依賴項
```

### 主要組件

1. **數據加載 (`src/data/`)**: 
   - `memory_optimized_data_loader.py`: 加載和預處理網路流量數據
   - `memory_optimized_graph_builder.py`: 從網路數據構建時間圖
   - `optimized_data_loader.py`: 加強版數據加載器，支持更多優化選項
   - `optimized_graph_builder.py`: 加強版圖構建器，專注於圖表示優化

2. **模型實現 (`src/models/`)**: 
   - `tgat_model.py`: 實現 TGAT 模型架構
   - `memory_optimized_train.py`: 提供訓練和評估功能
   - `optimized_tgat_model.py`: 優化版TGAT模型，專注於時間表示和圖注意力優化

3. **工具 (`src/utils/`)**: 
   - `memory_utils.py`: 記憶體優化工具
   - `utils.py`: 一般工具函數

4. **視覺化 (`src/visualization/`)**: 
   - `visualization.py`: 圖和結果視覺化工具

5. **運行腳本 (`scripts/`)**: 
   - `run.py`: 原始運行腳本
   - `run_optimized.py`: 整合所有優化組件的腳本
   - `run_memory_optimized.sh`: Shell腳本

### 擴展系統

#### 添加新功能

要向系統添加新功能，請按照以下步驟操作：

1. **確定適當的模塊**：確定哪個模塊應該包含您的新功能
2. **實現您的功能**：將您的代碼添加到適當的文件中
3. **更新配置**：如果您的功能需要配置，請將其添加到配置文件中
4. **測試您的功能**：使用現有系統測試您的功能

#### 示例：添加新的圖神經網絡模型

1. 在 `src/models/` 中創建一個新文件（例如 `src/models/your_model.py`）
2. 實現您的模型類，確保它具有與 TGAT 模型相同的接口
3. 更新 `src/models/__init__.py` 以導出您的模型
4. 在 `config/memory_optimized_config.yaml` 中添加配置選項
5. 修改 `scripts/run_optimized.py` 以在配置中指定時使用您的模型

## 記憶體優化功能

本實現包括多種記憶體優化技術：

1. **資料採樣**：減少訓練資料量同時保持模型性能
   - 使用 `stratified` 分層採樣確保各類攻擊模式都有足夠表示
   - 使用 `random` 隨機採樣快速測試系統功能
   - 可配置採樣比例和每個類別的最小樣本數

2. **高效資料格式**：使用更高效的文件格式加速資料加載
   - 自動將 CSV 轉換為 Parquet 格式並緩存
   - 使用 PyArrow 加速 CSV 和 Parquet 處理
   - 支持增量式資料加載和處理
   - 支持使用Polars加速大型資料處理

3. **增量式數據加載**：分塊加載數據以減少記憶體使用
   - 使用 `load_dataframe_chunked` 分批讀取大型 CSV 文件
   - 使用 `save_dataframe_chunked` 分批保存大型 DataFrame

4. **記憶體映射大型數據集**：對大型數據集使用記憶體映射
   - 使用 `memory_mapped_array` 創建記憶體映射數組
   - 使用 `load_memory_mapped_array` 加載已存在的記憶體映射

5. **優化圖表示**：使用高效的稀疏圖表示方法
   - 支持多種稀疏表示格式：CSR、COO、分塊稀疏
   - 使用邊索引格式減少記憶體消耗（PyG風格）
   - 使用稀疏張量表示輔助圖計算
   - 動態轉換密集和稀疏表示以平衡記憶體和計算效率

6. **智能子圖採樣**：使用優化採樣策略來減少記憶體使用
   - 基於節點重要性和連接度進行加權採樣
   - 自適應調整採樣率以控制圖大小
   - 保留關鍵節點和邊的同時減少總體圖大小
   - 使用滑動窗口接近採樣優化邊生成過程

7. **高效邊處理**：使用批次處理和稀疏操作
   - 邊批次添加和處理以減少記憶體峰值
   - 預先過濾無效邊以避免冗餘處理
   - 使用智能批次大小根據記憶體使用情況動態調整
   - 批次添加期間定期清理記憶體

8. **時間窗口優化**：高效處理時間性圖
   - 使用高效的時間窗口過濾方法
   - 緩存時間子圖以提高重複查詢效率
   - 智能時間量化技術減少不必要的精度
   - 時間表示的記憶體高效編碼和緩存

9. **自適應節點清理**：根據活躍度管理節點
   - 基於重要性和最後活躍時間的自適應清理策略
   - 自動調整清理間隔以平衡性能和記憶體使用
   - 優化清理過程的記憶體效率
   - 保留關鍵節點以維持圖結構的完整性

10. **混合精度訓練**：使用較低精度進行訓練以減少記憶體使用
    - 使用 PyTorch 的 AMP (Automatic Mixed Precision)
    - 自動處理精度轉換和梯度縮放
    - 在時間編碼中使用半精度存儲優化記憶體使用

11. **梯度優化技術**：
    - **梯度檢查點**：在前向傳播中選擇性地保存中間結果
    - **梯度累積**：在多個批次上累積梯度
    - **稀疏梯度**：使用閾值過濾小梯度值，轉換為稀疏表示

12. **高效注意力機制**：
    - 使用稀疏注意力矩陣減少記憶體使用
    - 注意力分數稀疏化，保留主要連接
    - 減少每個注意力頭的維度，使用投影層保持表達能力
    - 高效消息傳遞函數減少中間張量數量

13. **主動記憶體管理**：在訓練期間主動管理記憶體使用
    - 使用 `clean_memory` 定期清理未使用的記憶體
    - 使用 `print_memory_usage` 監控記憶體使用情況
    - 使用 `MemoryMonitor` 類持續監控和記錄記憶體使用
    - 支持限制GPU記憶體使用量

14. **記憶體洩漏檢測**：識別和修復記憶體洩漏
    - 使用 `detect_memory_leaks` 檢測函數中的記憶體洩漏
    - 提供詳細的洩漏源信息

15. **記憶體優化建議**：提供自動化記憶體優化建議
    - 使用 `print_optimization_suggestions` 獲取針對當前系統狀態的優化建議
    - 根據 CPU 和 GPU 記憶體使用情況提供不同建議

## 圖表示優化詳細說明

### 稀疏表示方法

本系統實現了多種圖的稀疏表示方法，用於處理大型圖數據：

1. **CSR (Compressed Sparse Row)格式**
   - 適用於大型稀疏圖，降低記憶體消耗達70-90%
   - 優化的行索引和列索引存儲
   - 支援高效的圖遍歷和計算操作

2. **COO (Coordinate)格式**
   - 簡單的二元組表示，使用源節點和目標節點索引
   - 適合動態添加邊的情況
   - 支持快速轉換為其他格式

3. **分塊稀疏格式**
   - 將大型稀疏矩陣分割成固定大小的塊
   - 只存儲非零塊，大幅降低存儲需求
   - 支持密集計算優化的塊級操作

4. **邊索引格式 (PyG風格)**
   - 使用兩個數組分別存儲所有邊的源節點和目標節點
   - 便於與PyTorch Geometric兼容
   - 支持高效的圖卷積操作

### 時間表示優化

系統中的時間表示也進行了特別優化：

1. **高效時間編碼**
   - 使用參數減少的編碼器降低模型大小
   - 減少時間編碼維度同時保持表達能力
   - 使用非線性轉換增強表達能力

2. **時間編碼緩存**
   - 使用LRU緩存存儲常用時間編碼結果
   - 時間量化技術減少緩存項數量
   - 高緩存命中率大幅提升處理速度

3. **半精度時間特徵**
   - 對時間特徵使用float16格式，減少內存使用量
   - 在推理階段自動轉換為全精度

### 轉換與兼容性

系統提供了多種格式之間的高效轉換：

1. **圖轉換API**
   - `to_sparse_tensor()`: 將圖轉換為稀疏張量表示
   - `to_scipy_sparse()`: 轉換為SciPy稀疏矩陣
   - 支持與DGL和PyG的格式互換

2. **模型轉換**
   - `to_sparse_tensor()`: 將模型權重轉換為稀疏表示
   - `to_dense_tensor()`: 將稀疏模型轉回密集表示
   - 自動計算稀疏化閾值，平衡精度和記憶體

## 故障排除

### 常見問題

1. **記憶體不足錯誤**：
   - 在配置中減小批次大小
   - 啟用動態批次大小調整
   - 啟用混合精度訓練
   - 啟用稀疏圖表示
   - 使用更多步驟的梯度累積
   - 啟用子圖採樣並設置較小的節點和邊限制

2. **CUDA 設備問題**：
   - 確保您的 GPU 配置正確
   - 檢查 CUDA 是否正確安裝
   - 嘗試使用 `--gpu -1` 在 CPU 上運行
   - 使用 `--limit_gpu_memory` 限制GPU內存使用

3. **數據加載錯誤**：
   - 確保您的數據格式正確
   - 檢查配置中的數據路徑
   - 嘗試使用較小的數據集進行測試
   - 啟用 `use_polars` 加速數據處理

4. **圖構建問題**：
   - 確保時間窗口設置適當
   - 調整子圖採樣參數
   - 啟用自適應清理策略
   - 檢查是否有節點ID重複問題

5. **運行時間過長**：
   - 啟用更激進的子圖採樣
   - 減少時間窗口大小
   - 啟用邊批次處理
   - 減少TGAT層數和頭數
   - 使用更小的hidden_dim和time_dim

6. **路徑問題**：
   - 確保從正確的目錄運行腳本
   - 檢查腳本中的相對路徑是否正確
   - 如果使用 shell 腳本，確保它有執行權限 (`chmod +x run_memory_optimized.sh`)

## 許可證

[MIT 許可證](LICENSE)
