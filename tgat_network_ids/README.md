# TGAT Network Intrusion Detection System

This project implements a Temporal Graph Attention Network (TGAT) for network intrusion detection. The system uses graph neural networks to model network traffic and detect anomalous patterns that may indicate attacks.

## Project Structure

```
tgat_network_ids/
├── config/                  # Configuration files
│   └── memory_optimized_config.yaml
├── legacy_code/             # Original implementation (for reference)
├── scripts/                 # Scripts for running the system
│   ├── run.py               # Main script to run the system
│   └── run_memory_optimized.sh
├── src/                     # Source code
│   ├── data/                # Data loading and processing
│   │   ├── memory_optimized_data_loader.py
│   │   └── memory_optimized_graph_builder.py
│   ├── models/              # Model implementations
│   │   ├── memory_optimized_train.py
│   │   └── tgat_model.py
│   ├── utils/               # Utility functions
│   │   └── memory_utils.py
│   ├── visualization/       # Visualization tools
│   │   └── visualization.py
│   └── memory_optimized_main.py
├── SQL/                     # SQL-related files
├── README.md                # This file
└── requirements.txt         # Dependencies
```

## Memory Optimization Features

This implementation includes several memory optimization techniques:

1. **Incremental Data Loading**: Load data in chunks to reduce memory usage
2. **Memory-Mapped Large Datasets**: Use memory mapping for large datasets
3. **Subgraph Sampling**: Sample subgraphs for training to reduce memory usage
4. **Mixed Precision Training**: Use lower precision for training to reduce memory usage
5. **Gradient Accumulation**: Accumulate gradients over multiple batches
6. **Dynamic Batch Size**: Adjust batch size based on available memory
7. **Active Memory Management**: Actively manage memory usage during training

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running the System

```bash
# Using the run script
python scripts/run.py --config config/memory_optimized_config.yaml --mode train --data_path ./data/your_data --visualize --monitor_memory

# Or using the shell script
./scripts/run_memory_optimized.sh
```

### Configuration

Edit `config/memory_optimized_config.yaml` to configure the system:

```yaml
# Data configuration
data:
  path: ./data/your_data
  test_size: 0.2
  random_state: 42
  batch_size: 128

# Model configuration
model:
  hidden_dim: 64
  out_dim: 64
  time_dim: 16
  num_layers: 2
  num_heads: 4
  dropout: 0.1
  use_mixed_precision: true
  use_gradient_accumulation: true
  gradient_accumulation_steps: 4
  use_gradient_checkpointing: true

# Training configuration
train:
  lr: 0.001
  weight_decay: 1e-5
  epochs: 100
  patience: 10
  batch_size: 128
  use_dynamic_batch_size: true
  memory_threshold: 0.8
  use_progressive_training: true
  progressive_training_initial_ratio: 0.3
  progressive_training_growth_rate: 0.1

# Output configuration
output:
  model_dir: ./models
  result_dir: ./results
  visualization_dir: ./visualizations
  memory_report_dir: ./memory_reports
  checkpoint_dir: ./checkpoints
```

## License

[MIT License](LICENSE)
