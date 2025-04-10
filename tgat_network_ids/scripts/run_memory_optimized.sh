#!/bin/bash

# 記憶體優化版 TGAT 網路攻擊檢測系統執行腳本

# 創建必要的目錄
mkdir -p ./recode
mkdir -p ./recode/preprocessed_data
mkdir -p ./recode/models_memory_optimized
mkdir -p ./recode/results_memory_optimized
mkdir -p ./recode/visualizations_memory_optimized
mkdir -p ./recode/memory_reports
mkdir -p ./recode/checkpoints_memory_optimized

# 設置 Python 垃圾回收參數
export PYTHONMALLOC=malloc
export PYTHONMALLOCSTATS=1
export PYTHONGC=1

# 設置 CUDA 記憶體分配器 - 更保守的設置
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,garbage_collection_threshold:0.6

# 設置 CUDA 緩存大小 (限制為 10GB，適合 40GB RAM)
export CUDA_CACHE_MAXSIZE=10737418240

# 設置 OMP 線程數 (針對 8 核 CPU 優化，保留 2 核給系統)
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6
export VECLIB_MAXIMUM_THREADS=6
export NUMBA_NUM_THREADS=6

# 設置 CPU 頻率調整策略 (如果支持)
if command -v cpupower &> /dev/null; then
    echo "設置 CPU 頻率調整策略為 performance..."
    sudo cpupower frequency-set -g performance || true
fi

# 設置 NUMA 控制 (如果可用)
if command -v numactl &> /dev/null; then
    NUMA_CMD="numactl --localalloc"
else
    NUMA_CMD=""
fi

# 設置 Python 記憶體分析器
export PYTHONTRACEMALLOC=10

# 顯示記憶體使用情況
echo "系統記憶體使用情況:"
free -h

# 如果有 GPU，顯示 GPU 記憶體使用情況
if command -v nvidia-smi &> /dev/null; then
    echo "GPU 記憶體使用情況:"
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv
fi

echo "開始執行記憶體優化版 TGAT 網路攻擊檢測系統..."

# 設置進程優先級 (降低優先級以避免系統卡頓)
echo "設置進程優先級..."
renice -n 10 $$ || true

# 設置 I/O 調度器 (如果支持)
if command -v ionice &> /dev/null; then
    echo "設置 I/O 調度器為 best-effort..."
    ionice -c 2 -n 7 -p $$ || true
fi

# 執行記憶體優化版主程式
echo "使用 $OMP_NUM_THREADS 個線程執行..."
$NUMA_CMD python ./tgat_network_ids/src/memory_optimized_main.py \
  --config ./tgat_network_ids/config/memory_optimized_config.yaml \
  --mode train \
  --data_path ./tgat_network_ids/data/test_v1/ \
  --visualize \
  --monitor_memory \
  --gpu 0

# 顯示執行後的記憶體使用情況
echo "執行完成，系統記憶體使用情況:"
free -h

# 如果有 GPU，顯示執行後的 GPU 記憶體使用情況
if command -v nvidia-smi &> /dev/null; then
    echo "執行完成，GPU 記憶體使用情況:"
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv
fi
