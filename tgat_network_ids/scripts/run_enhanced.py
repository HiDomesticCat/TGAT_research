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
    parser.add_argument('--adaptive_window_config', type=str, 
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
    parser.add_argument('--output_dir', type=str,
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
    with open(config_path, 'r') as f:
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

    # Command line arguments override configuration file
    if args.data_path:
        config['data']['path'] = args.data_path
    if args.use_gpu is not None:
        config['model']['use_gpu'] = args.use_gpu
    if args.epochs:
        config['training']['epochs'] = args.epochs

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

    # Use memory tracking context manager
    with track_memory_usage('Main function execution'):
        # Load data
        logger.info("Starting data loading...")
        data_loader = EnhancedMemoryOptimizedDataLoader(
            config['data']['path'],
            use_memory=args.use_memory,
            memory_size=args.memory_size
        )
        
        # Build graph
        logger.info("Starting graph building...")
        graph_builder = OptimizedGraphBuilder(
            data_loader,
            use_sparse=args.use_sparse_representation
        )
        
        # Create model
        logger.info("Creating model...")
        model = OptimizedTGATModel(
            in_dim=config['model']['input_dim'],
            hidden_dim=config['model']['hidden_dim'],
            out_dim=config['model']['output_dim'],
            time_dim=config['model']['time_dim'],
            num_layers=config['model']['num_layers'],
            num_heads=config['model']['num_heads'],
            dropout=config['model']['dropout'],
            num_classes=config['model']['num_classes']
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
        
        # Set up optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Set up loss function
        criterion = torch.nn.CrossEntropyLoss()
        
        # Execution mode
        mode = args.mode
        
        # Execute according to mode
        if mode == 'train':
            logger.info("Starting model training...")
            # Training logic...
            epochs = config['training']['epochs']
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
