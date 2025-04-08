# Memory-Optimized TGAT for Network Intrusion Detection

This repository contains a memory-optimized implementation of Temporal Graph Attention Network (TGAT) for network intrusion detection. The system analyzes network traffic data to detect potential attacks using graph-based deep learning techniques.

## Memory Optimization Features

The implementation includes various memory optimization techniques to reduce memory usage during training and inference:

1. **Data Loading and Preprocessing Optimizations**
   - Incremental data loading: Load large datasets in chunks
   - Memory mapping: Use disk as virtual memory
   - Data compression: Reduce memory footprint
   - Preprocessed data caching: Avoid redundant computations

2. **Graph Structure Optimizations**
   - Subgraph sampling: Limit the number of nodes and edges
   - Sparse representation: Reduce memory usage
   - Batch processing of edges: Add edges in batches
   - Periodic pruning of inactive nodes: Free up memory

3. **Model Training Optimizations**
   - Mixed precision training: Use FP16 instead of FP32
   - Gradient accumulation: Reduce batch size
   - Gradient checkpointing: Reduce memory usage of intermediate activations
   - Dynamic batch sizing: Adjust batch size based on memory usage
   - Progressive training: Start with small dataset and gradually increase

4. **Active Memory Management**
   - Memory usage monitoring: Monitor memory usage in real-time
   - Active garbage collection: Periodically clean up unused memory
   - GPU memory optimization: Clear CUDA cache

## Project Structure

- `memory_utils.py`: Memory optimization utility module
- `memory_optimized_data_loader.py`: Memory-optimized data loading and preprocessing module
- `memory_optimized_graph_builder.py`: Memory-optimized dynamic graph structure building module
- `memory_optimized_train.py`: Memory-optimized model training and evaluation module
- `memory_optimized_main.py`: Memory-optimized main program
- `memory_optimized_config.yaml`: Memory-optimized configuration file

## Requirements

The project requires the following dependencies:

```
torch>=1.7.0
dgl>=0.6.0
numpy>=1.19.0
pandas>=1.1.0
scikit-learn>=0.23.0
matplotlib>=3.3.0
seaborn>=0.11.0
pyyaml>=5.3.0
tqdm>=4.48.0
psutil>=5.7.0
```

## Usage

1. **Configure the System**
   - Edit the `memory_optimized_config.yaml` file to adjust parameters according to your needs
   - Pay special attention to memory optimization related configuration options

2. **Run the System**
   - Training mode:
     ```
     python memory_optimized_main.py --mode train --data_path YOUR_DATA_PATH --visualize --monitor_memory
     ```
   
   - Testing mode:
     ```
     python memory_optimized_main.py --mode test --model_path YOUR_MODEL_PATH --data_path YOUR_DATA_PATH --visualize
     ```
   
   - Detection mode:
     ```
     python memory_optimized_main.py --mode detect --model_path YOUR_MODEL_PATH --data_path YOUR_DATA_PATH --visualize
     ```

3. **Monitor Memory Usage**
   - Use the `--monitor_memory` parameter to enable memory monitoring
   - Memory usage reports will be saved in the `memory_reports` directory

## Configuration Options

The `memory_optimized_config.yaml` file contains various configuration options for the system. Key memory optimization parameters include:

- `data.use_memory_mapping`: Enable memory mapping for large datasets
- `data.incremental_loading`: Enable incremental data loading
- `graph.use_subgraph_sampling`: Enable subgraph sampling
- `model.use_mixed_precision`: Enable mixed precision training
- `model.use_gradient_accumulation`: Enable gradient accumulation
- `model.use_gradient_checkpointing`: Enable gradient checkpointing
- `train.use_dynamic_batch_size`: Enable dynamic batch sizing
- `train.use_progressive_training`: Enable progressive training
- `system.monitor_memory`: Enable memory usage monitoring

## License

This project is licensed under the MIT License - see the LICENSE file for details.
