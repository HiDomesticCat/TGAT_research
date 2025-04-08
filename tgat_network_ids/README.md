# TGAT Network Intrusion Detection System

This project implements a Temporal Graph Attention Network (TGAT) for network intrusion detection. The system uses graph neural networks to model network traffic and detect anomalous patterns that may indicate attacks.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/TGAT_research.git
cd TGAT_research

# Install dependencies
pip install -r tgat_network_ids/requirements.txt
```

### Running the System

```bash
# Using the run script
python tgat_network_ids/scripts/run.py --config tgat_network_ids/config/memory_optimized_config.yaml --mode train --data_path ./data/your_data --visualize --monitor_memory

# Or using the shell script
./tgat_network_ids/scripts/run_memory_optimized.sh
```

## Usage Guide

### Training a Model

```bash
python tgat_network_ids/scripts/run.py \
  --config tgat_network_ids/config/memory_optimized_config.yaml \
  --mode train \
  --data_path ./data/your_data \
  --visualize \
  --monitor_memory
```

### Testing a Model

```bash
python tgat_network_ids/scripts/run.py \
  --config tgat_network_ids/config/memory_optimized_config.yaml \
  --mode test \
  --model_path ./models/best_model.pt \
  --data_path ./data/your_data \
  --visualize
```

### Real-time Detection

```bash
python tgat_network_ids/scripts/run.py \
  --config tgat_network_ids/config/memory_optimized_config.yaml \
  --mode detect \
  --model_path ./models/best_model.pt \
  --data_path ./data/your_data \
  --visualize
```

## Configuration

The system is highly configurable through YAML configuration files. The main configuration file is located at `tgat_network_ids/config/memory_optimized_config.yaml`.

### Key Configuration Options

```yaml
# Data configuration
data:
  path: ./data/your_data  # Path to your dataset
  test_size: 0.2          # Percentage of data to use for testing
  batch_size: 128         # Batch size for training and evaluation

# Model configuration
model:
  hidden_dim: 64          # Hidden dimension size
  out_dim: 64             # Output dimension size
  time_dim: 16            # Time encoding dimension
  num_layers: 2           # Number of TGAT layers
  num_heads: 4            # Number of attention heads
  dropout: 0.1            # Dropout rate
  use_mixed_precision: true  # Use mixed precision training
  use_gradient_accumulation: true  # Use gradient accumulation
  gradient_accumulation_steps: 4   # Number of steps for gradient accumulation
  use_gradient_checkpointing: true  # Use gradient checkpointing

# Training configuration
train:
  lr: 0.001               # Learning rate
  weight_decay: 1e-5      # Weight decay for regularization
  epochs: 100             # Maximum number of epochs
  patience: 10            # Early stopping patience
  use_dynamic_batch_size: true  # Dynamically adjust batch size based on memory
  memory_threshold: 0.8   # Memory usage threshold for batch size adjustment
  use_progressive_training: true  # Use progressive training
```

## Development Guide

### Project Structure

```
tgat_network_ids/
├── config/                  # Configuration files
├── legacy_code/             # Original implementation (for reference)
├── scripts/                 # Scripts for running the system
├── src/                     # Source code
│   ├── data/                # Data loading and processing
│   ├── models/              # Model implementations
│   ├── utils/               # Utility functions
│   └── visualization/       # Visualization tools
├── SQL/                     # SQL-related files
├── README.md                # This file
└── requirements.txt         # Dependencies
```

### Key Components

1. **Data Loading (`src/data/`)**: 
   - `memory_optimized_data_loader.py`: Loads and preprocesses network traffic data
   - `memory_optimized_graph_builder.py`: Constructs temporal graphs from network data

2. **Model Implementation (`src/models/`)**: 
   - `tgat_model.py`: Implements the TGAT model architecture
   - `memory_optimized_train.py`: Provides training and evaluation functionality

3. **Utilities (`src/utils/`)**: 
   - `memory_utils.py`: Memory optimization utilities

4. **Visualization (`src/visualization/`)**: 
   - `visualization.py`: Tools for visualizing graphs and results

### Extending the System

#### Adding New Features

To add new features to the system, follow these steps:

1. **Identify the appropriate module**: Determine which module should contain your new feature
2. **Implement your feature**: Add your code to the appropriate file
3. **Update the configuration**: If your feature requires configuration, add it to the config file
4. **Test your feature**: Test your feature with the existing system

#### Example: Adding a New Graph Neural Network Model

1. Create a new file in `src/models/` (e.g., `src/models/your_model.py`)
2. Implement your model class, ensuring it has the same interface as the TGAT model
3. Update `src/models/__init__.py` to export your model
4. Add configuration options to `config/memory_optimized_config.yaml`
5. Modify `src/memory_optimized_main.py` to use your model when specified in the config

## Memory Optimization Features

This implementation includes several memory optimization techniques:

1. **Incremental Data Loading**: Load data in chunks to reduce memory usage
2. **Memory-Mapped Large Datasets**: Use memory mapping for large datasets
3. **Subgraph Sampling**: Sample subgraphs for training to reduce memory usage
4. **Mixed Precision Training**: Use lower precision for training to reduce memory usage
5. **Gradient Accumulation**: Accumulate gradients over multiple batches
6. **Dynamic Batch Size**: Adjust batch size based on available memory
7. **Active Memory Management**: Actively manage memory usage during training

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   - Reduce batch size in the configuration
   - Enable dynamic batch size adjustment
   - Enable mixed precision training
   - Use gradient accumulation with more steps

2. **CUDA Device Issues**:
   - Ensure your GPU is properly configured
   - Check if CUDA is installed correctly
   - Try running on CPU with `--gpu -1`

3. **Data Loading Errors**:
   - Ensure your data is in the correct format
   - Check the data path in the configuration
   - Try using a smaller dataset for testing

## License

[MIT License](LICENSE)
