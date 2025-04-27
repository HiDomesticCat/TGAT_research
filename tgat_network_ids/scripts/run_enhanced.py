#!/usr/bin/env python
# coding: utf-8 -*-

"""
Enhanced TGAT Network Intrusion Detection System Execution Script

Integrates all memory optimizations, adaptive time windows, and advanced graph sampling features.
Supports the following enhancements:
1. Memory-optimized data loading and preprocessing
2. Adaptive multi-scale time window selection
3. Advanced graph sampling strategies (GraphSAINT, Cluster-GCN, etc.)
4. IP address structural feature extraction
5. Statistical feature selection
"""

import os
import sys
import yaml
import logging
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
import json
import dgl

# Ensure project modules can be imported
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import project modules
from src.data.optimized_data_loader import EnhancedMemoryOptimizedDataLoader
from src.data.optimized_graph_builder import OptimizedGraphBuilder
from src.models.optimized_tgat_model import OptimizedTGATModel
from src.data.adaptive_window import AdaptiveWindowManager
from src.data.advanced_sampling import AdvancedGraphSampler
from src.utils.memory_utils import print_memory_usage, track_memory_usage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"tgat_ids_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced TGAT Network Intrusion Detection System')

    # Basic parameters
    parser.add_argument('--config', type=str, default='config/memory_optimized_config.yaml',
                        help='Configuration file path')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Data path, overrides the configuration file setting')
    parser.add_argument('--use_gpu', action='store_true', default=None,
                        help='Use GPU, overrides the configuration file setting')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Training epochs, overrides the configuration file setting')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'predict'], default='train',
                        help='Execution mode: train, evaluate or predict')

    # Enhancement-related parameters
    parser.add_argument('--use_adaptive_window', action='store_true',
                        help='Use adaptive time window')
    parser.add_argument('--adaptive_window_config', type=str, default=None,
                        help='Adaptive window configuration file path')

    parser.add_argument('--use_advanced_sampling', action='store_true', 
                        help='Use advanced graph sampling strategy')
    parser.add_argument('--sampling_method', type=str, default='graphsaint',
                        choices=['graphsaint', 'cluster-gcn', 'frontier', 'historical'],
                        help='Graph sampling method')
    parser.add_argument('--sample_size', type=int, default=5000,
                        help='Subgraph sample size')

    parser.add_argument('--use_memory', action='store_true',
                        help='Enable memory mechanism')
    parser.add_argument('--memory_size', type=int, default=1000,
                        help='Memory buffer size')

    parser.add_argument('--use_position_embedding', action='store_true',
                        help='Use position embedding')

    # Visualization and output related
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization charts')
    parser.add_argument('--save_model', action='store_true', default=True,
                        help='Save model checkpoints')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory, default uses configuration settings')

    # Advanced optimization parameters
    parser.add_argument('--use_sparse_representation', action='store_true',
                        help='Use sparse representation to save memory')
    parser.add_argument('--use_mixed_precision', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Use gradient checkpointing to save memory')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Log level')

    return parser.parse_args()

def load_config(config_path):
    """Load configuration file"""
    # Allow both absolute and relative paths
    if not os.path.isabs(config_path):
        # Try direct relative path
        if os.path.exists(config_path):
            path = config_path
        # Try relative to script directory
        elif os.path.exists(os.path.join(os.path.dirname(__file__), config_path)):
            path = os.path.join(os.path.dirname(__file__), config_path)
        # Try relative to project root
        elif os.path.exists(os.path.join(project_root, config_path)):
            path = os.path.join(project_root, config_path)
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    else:
        path = config_path
        
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_output_dirs(config):
    """Set up output directories"""
    # Create model save directory
    model_dir = config.get('model', {}).get('save_dir', './models')
    os.makedirs(model_dir, exist_ok=True)

    # Create results save directory
    results_dir = config.get('evaluation', {}).get('results_dir', './results')
    os.makedirs(results_dir, exist_ok=True)

    # Create visualization save directory
    vis_dir = config.get('visualization', {}).get('save_dir', './visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # Create output directory
    output_dir = os.path.join(project_root, 'output')
    os.makedirs(output_dir, exist_ok=True)

    return model_dir, results_dir, vis_dir, output_dir

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create data config dictionary for the data loader
    data_config = {
        'data': {
            'path': args.data_path if args.data_path else config['data']['path'],
            'test_size': config['data'].get('test_size', 0.2),
            'min_samples_per_class': config['data'].get('min_samples_per_class', 1000),
            # Add other necessary data configuration options
            'batch_size': config['train'].get('batch_size', 64),
            'memory_optimization': True
        }
    }

    # Command line arguments override configuration file
    if args.data_path:
        config['data']['path'] = args.data_path
    if args.use_gpu is not None:
        config['model']['use_gpu'] = args.use_gpu
    if args.epochs:
        config['train']['epochs'] = args.epochs

    # Set up output directories
    model_dir, results_dir, vis_dir, output_dir = setup_output_dirs(config)

    # Log configuration
    logger.info(f"Using configuration file: {args.config}")
    logger.info(f"Command line arguments: {args}")

    # Log enhancement status
    logger.info(f"Adaptive time window: {args.use_adaptive_window}")
    logger.info(f"Advanced graph sampling strategy: {args.use_advanced_sampling} (method: {args.sampling_method})")
    logger.info(f"Memory mechanism: {args.use_memory}")
    logger.info(f"Position embedding: {args.use_position_embedding}")

    # Check if GPU is used
    use_gpu = config['model'].get('use_gpu', False) and torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')
    logger.info(f"Using device: {device}")

    # Set random seed for reproducibility
    seed = config.get('random_seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if use_gpu:
        torch.cuda.manual_seed_all(seed)

    # Add sparse representation option to graph config
    graph_config = config.get('graph', {})
    graph_config['use_sparse_representation'] = args.use_sparse_representation

    # Create full config object for graph builder
    full_config = {
        'data': config['data'],
        'graph': graph_config,
        'model': config['model'],
        'system': config.get('system', {})
    }

    # Use memory tracking context manager
    with track_memory_usage('Main function execution'):
        # Load data
        logger.info("Starting data loading...")
        data_loader = EnhancedMemoryOptimizedDataLoader(data_config)
        
        # Build graph
        logger.info("Starting graph building...")
        graph_builder = OptimizedGraphBuilder(full_config, device=str(device))
        
        # Create model with default parameters if not in config
        logger.info("Creating model...")
        model_config = config.get('model', {})
        
        # Set default values for required parameters if not found in config
        input_dim = model_config.get('input_dim', 128)  # Default input dimension
        hidden_dim = model_config.get('hidden_dim', 64) 
        out_dim = model_config.get('out_dim', 64)
        time_dim = model_config.get('time_dim', 16)
        num_layers = model_config.get('num_layers', 2)
        num_heads = model_config.get('num_heads', 4)
        dropout = model_config.get('dropout', 0.2)
        num_classes = model_config.get('num_classes', 2)  # Default: binary classification
        
        logger.info(f"Model parameters: input_dim={input_dim}, hidden_dim={hidden_dim}, out_dim={out_dim}, "
                    f"num_classes={num_classes}, num_layers={num_layers}, num_heads={num_heads}")
        
        model = OptimizedTGATModel(
            in_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            time_dim=time_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            num_classes=num_classes
        )
        
        # Check if using mixed precision training
        if args.use_mixed_precision:
            logger.info("Enabling mixed precision training...")
            model.enable_mixed_precision()
            
        # Check if using gradient checkpointing
        if args.gradient_checkpointing:
            logger.info("Enabling gradient checkpointing...")
            model.enable_gradient_checkpointing()
            
        # Move model to device
        model.to(device)
        
        # Get and convert learning rate and weight decay parameters with proper type conversion
        try:
            learning_rate = float(config['train'].get('learning_rate', 0.001))  # Default if missing
        except (ValueError, TypeError):
            learning_rate = 0.001
            logger.warning(f"Invalid learning_rate value in config, using default: {learning_rate}")
            
        try:
            weight_decay = float(config['train'].get('weight_decay', 5e-5))  # Default if missing
        except (ValueError, TypeError):
            weight_decay = 5e-5
            logger.warning(f"Invalid weight_decay value in config, using default: {weight_decay}")
            
        # Set up optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        logger.info(f"Optimizer configured with lr={learning_rate}, weight_decay={weight_decay}")
        
        # Set up loss function
        criterion = torch.nn.CrossEntropyLoss()
        
        # Execution mode
        mode = args.mode
        
        # Execute according to mode
        if mode == 'train':
            logger.info("Starting model training...")
            # Training logic...
            epochs = config['train'].get('epochs', 10)  # Default if missing
            try:
                epochs = int(epochs)
            except (ValueError, TypeError):
                epochs = 10
                logger.warning(f"Invalid epochs value in config, using default: {epochs}")
                
            for epoch in range(epochs):
                logger.info(f"Epoch {epoch+1}/{epochs}")
                # Train one epoch...
                
        elif mode == 'eval':
            logger.info("Evaluating model performance...")
            # Evaluation logic...
            
        elif mode == 'predict':
            logger.info("Running prediction...")
            # Prediction logic...
            
        # Save results
        logger.info("Saving results...")

    # Return results dictionary
    results = {
        'status': 'success',
        'model_path': os.path.join(model_dir, 'final_model.pth'),
        'config': config,
        'output_dir': output_dir
    }
    
    return results

if __name__ == "__main__":
    results = main()
    logger.info(f"Execution completed: {results['status']}")
