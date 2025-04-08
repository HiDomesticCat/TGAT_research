#!/bin/bash

# 記憶體優化版 TGAT 網路攻擊檢測系統執行腳本

# 創建必要的目錄
mkdir -p ./preprocessed_data
mkdir -p ./models_memory_optimized
mkdir -p ./results_memory_optimized
mkdir -p ./visualizations_memory_optimized
mkdir -p ./memory_reports
mkdir -p ./checkpoints_memory_optimized

# 設置 Python 垃圾回收參數
export PYTHONMALLOC=malloc
export PYTHONMALLOCSTATS=1
export PYTHONGC=1

# 設置 CUDA 記憶體分配器
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 執行記憶體優化版主程式
python TGAT_research/tgat_network_ids/scripts/run.py \
  --config TGAT_research/tgat_network_ids/config/memory_optimized_config.yaml \
  --mode train \
  --data_path ./data/test_v1 \
  --visualize \
  --monitor_memory
