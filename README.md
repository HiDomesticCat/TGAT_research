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
# 使用運行腳本
python tgat_network_ids/scripts/run.py --config tgat_network_ids/config/memory_optimized_config.yaml --mode train --data_path ./data/your_data --visualize --monitor_memory

# 或使用 shell 腳本
./tgat_network_ids/scripts/run_memory_optimized.sh
```

## 使用指南

### 訓練模型

```bash
python tgat_network_ids/scripts/run.py \
  --config tgat_network_ids/config/memory_optimized_config.yaml \
  --mode train \
  --data_path ./data/your_data \
  --visualize \
  --monitor_memory
```

### 測試模型

```bash
python tgat_network_ids/scripts/run.py \
  --config tgat_network_ids/config/memory_optimized_config.yaml \
  --mode test \
  --model_path ./models/best_model.pt \
  --data_path ./data/your_data \
  --visualize
```

### 即時檢測

```bash
python tgat_network_ids/scripts/run.py \
  --config tgat_network_ids/config/memory_optimized_config.yaml \
  --mode detect \
  --model_path ./models/best_model.pt \
  --data_path ./data/your_data \
  --visualize
```

## 配置

系統可通過 YAML 配置文件進行高度自定義。主要配置文件位於 `tgat_network_ids/config/memory_optimized_config.yaml`。

### 主要配置選項

```yaml
# 數據配置
data:
  path: ./data/your_data  # 數據集路徑
  test_size: 0.2          # 用於測試的數據百分比
  batch_size: 128         # 訓練和評估的批次大小

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

# 訓練配置
train:
  lr: 0.001               # 學習率
  weight_decay: 1e-5      # 權重衰減係數
  epochs: 100             # 最大訓練輪數
  patience: 10            # 早停耐心值
  use_dynamic_batch_size: true  # 根據記憶體動態調整批次大小
  memory_threshold: 0.8   # 批次大小調整的記憶體使用閾值
  use_progressive_training: true  # 使用漸進式訓練
```

## 開發指南

### 專案結構

```
tgat_network_ids/
├── config/                  # 配置文件
├── legacy_code/             # 原始實現（供參考）
├── scripts/                 # 運行腳本
├── src/                     # 源代碼
│   ├── data/                # 數據加載和處理
│   ├── models/              # 模型實現
│   ├── utils/               # 工具函數
│   └── visualization/       # 視覺化工具
├── SQL/                     # SQL 相關文件
├── README.md                # 本文件
└── requirements.txt         # 依賴項
```

### 主要組件

1. **數據加載 (`src/data/`)**: 
   - `memory_optimized_data_loader.py`: 加載和預處理網路流量數據
   - `memory_optimized_graph_builder.py`: 從網路數據構建時間圖

2. **模型實現 (`src/models/`)**: 
   - `tgat_model.py`: 實現 TGAT 模型架構
   - `memory_optimized_train.py`: 提供訓練和評估功能

3. **工具 (`src/utils/`)**: 
   - `memory_utils.py`: 記憶體優化工具

4. **視覺化 (`src/visualization/`)**: 
   - `visualization.py`: 圖和結果視覺化工具

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
5. 修改 `src/memory_optimized_main.py` 以在配置中指定時使用您的模型

## 記憶體優化功能

本實現包括多種記憶體優化技術：

1. **增量式數據加載**：分塊加載數據以減少記憶體使用
2. **記憶體映射大型數據集**：對大型數據集使用記憶體映射
3. **子圖採樣**：對訓練採樣子圖以減少記憶體使用
4. **混合精度訓練**：使用較低精度進行訓練以減少記憶體使用
5. **梯度累積**：在多個批次上累積梯度
6. **動態批次大小**：根據可用記憶體調整批次大小
7. **主動記憶體管理**：在訓練期間主動管理記憶體使用

## 故障排除

### 常見問題

1. **記憶體不足錯誤**：
   - 在配置中減小批次大小
   - 啟用動態批次大小調整
   - 啟用混合精度訓練
   - 使用更多步驟的梯度累積

2. **CUDA 設備問題**：
   - 確保您的 GPU 配置正確
   - 檢查 CUDA 是否正確安裝
   - 嘗試使用 `--gpu -1` 在 CPU 上運行

3. **數據加載錯誤**：
   - 確保您的數據格式正確
   - 檢查配置中的數據路徑
   - 嘗試使用較小的數據集進行測試

## 許可證

[MIT 許可證](LICENSE)
