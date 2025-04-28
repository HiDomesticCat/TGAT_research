# TGAT 網路入侵檢測系統使用指南

此文檔提供 TGAT 網路入侵檢測系統的使用說明，包括命令行參數、配置選項以及最佳實踐。

## 1. 基本執行

執行增強版系統的基本命令：

```bash
python tgat_network_ids/scripts/run_enhanced.py --config tgat_network_ids/config/memory_optimized_config.yaml
```

## 2. 命令行參數

### 基本參數

| 參數 | 說明 | 示例 |
|------|------|------|
| `--config` | 配置文件路徑 | `--config tgat_network_ids/config/memory_optimized_config.yaml` |
| `--data_path` | 資料集路徑 | `--data_path ./data/my_dataset/` |
| `--mode` | 執行模式（train/eval/predict） | `--mode train` |
| `--use_gpu` | 啟用 GPU 加速 | `--use_gpu` |
| `--epochs` | 訓練輪數 | `--epochs 100` |

### 圖表示與記憶體優化參數

| 參數 | 說明 | 示例 |
|------|------|------|
| `--use_sparse_representation` | 使用稀疏圖表示 | `--use_sparse_representation` |
| `--use_mixed_precision` | 使用混合精度訓練 | `--use_mixed_precision` |
| `--gradient_checkpointing` | 使用梯度檢查點 | `--gradient_checkpointing` |

### 進階採樣參數

需要同時啟用 `--use_advanced_sampling` 和指定採樣方法：

| 參數 | 說明 | 示例 |
|------|------|------|
| `--use_advanced_sampling` | 啟用進階圖採樣 | `--use_advanced_sampling` |
| `--sampling_method` | 採樣方法 | `--sampling_method graphsaint` |
| `--sample_size` | 樣本大小 | `--sample_size 5000` |

可用的採樣方法：`graphsaint`, `cluster-gcn`, `frontier`, `historical`

### 時間窗口參數

| 參數 | 說明 | 示例 |
|------|------|------|
| `--use_adaptive_window` | 啟用自適應時間窗口 | `--use_adaptive_window` |
| `--adaptive_window_config` | 自適應窗口配置文件 | `--adaptive_window_config path/to/config.yaml` |

### 其他功能參數

| 參數 | 說明 | 示例 |
|------|------|------|
| `--use_memory` | 啟用記憶機制 | `--use_memory` |
| `--memory_size` | 記憶緩衝區大小 | `--memory_size 2000` |
| `--use_position_embedding` | 使用位置嵌入 | `--use_position_embedding` |
| `--visualize` | 生成視覺化圖表 | `--visualize` |
| `--save_model` | 保存模型檢查點 | `--save_model` |
| `--output_dir` | 指定輸出目錄 | `--output_dir ./output_results/` |
| `--log_level` | 設置日誌級別 | `--log_level DEBUG` |

## 3. 實用指令範例

### 基本訓練

```bash
python tgat_network_ids/scripts/run_enhanced.py \
  --config tgat_network_ids/config/memory_optimized_config.yaml \
  --mode train \
  --data_path ./my_data/ \
  --use_gpu
```

### 使用稀疏表示和混合精度

```bash
python tgat_network_ids/scripts/run_enhanced.py \
  --config tgat_network_ids/config/memory_optimized_config.yaml \
  --mode train \
  --data_path ./my_data/ \
  --use_gpu \
  --use_sparse_representation \
  --use_mixed_precision
```

### 使用進階採樣

```bash
python tgat_network_ids/scripts/run_enhanced.py \
  --config tgat_network_ids/config/memory_optimized_config.yaml \
  --mode train \
  --data_path ./my_data/ \
  --use_gpu \
  --use_advanced_sampling \
  --sampling_method graphsaint \
  --sample_size 5000
```

### 評估模式

```bash
python tgat_network_ids/scripts/run_enhanced.py \
  --config tgat_network_ids/config/memory_optimized_config.yaml \
  --mode eval \
  --data_path ./test_data/ \
  --use_gpu
```

### 預測模式

```bash
python tgat_network_ids/scripts/run_enhanced.py \
  --config tgat_network_ids/config/memory_optimized_config.yaml \
  --mode predict \
  --data_path ./new_data/ \
  --use_gpu
```

## 4. 常見問題與解決

1. **記憶體不足問題**：
   - 啟用稀疏表示: `--use_sparse_representation`
   - 減少批次大小: 在配置文件中修改 `batch_size`
   - 啟用梯度檢查點: `--gradient_checkpointing`

2. **解析參數錯誤**：
   - 確保使用雙連字符（如 `--use_gpu` 而非 `-use_gpu`）
   - 確保布爾參數無值（如 `--use_gpu` 而非 `--use_gpu=True`）
   - 字符串和整數參數需要值（如 `--mode train`、`--epochs 50`）

3. **配置文件問題**：
   - 確保 YAML 文件格式正確
   - 數值參數應避免科學記數法（如使用 `0.00005` 而非 `5e-5`）
