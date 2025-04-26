#!/usr/bin/env python
# coding: utf-8 -*-

"""
Memory-optimized TGAT training module

This module provides memory-efficient implementation for training TGAT model,
including mixed precision training, gradient accumulation, and checkpointing.
"""

import os
import torch
import numpy as np
import time
import logging
from datetime import datetime
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from ..utils.memory_utils import clean_memory, print_memory_usage, track_memory_usage

logger = logging.getLogger(__name__)

class MemoryOptimizedTGATTrainer:
    """Memory-optimized trainer for TGAT models"""
    
    def __init__(self, model, config, device=None):
        """
        Initialize the memory-optimized trainer
        
        Args:
            model: The TGAT model to train
            config: Training configuration
            device: Training device (GPU or CPU)
        """
        self.model = model
        self.config = config
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Set up optimizer
        self.optimizer = Adam(
            self.model.parameters(), 
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 0.0)
        )
        
        # Set up loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Set up memory optimization options
        self.use_mixed_precision = config.get('use_mixed_precision', False)
        self.use_gradient_accumulation = config.get('use_gradient_accumulation', False)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self.use_gradient_checkpointing = config.get('use_gradient_checkpointing', False)
        
        # Initialize training variables
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = float('inf')
        self.training_history = []
        
        # Initialize scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None
        
        # Configure model based on memory optimization settings
        if self.use_gradient_checkpointing and hasattr(self.model, 'enable_gradient_checkpointing'):
            self.model.enable_gradient_checkpointing()
        
        logger.info(f"Initialized MemoryOptimizedTGATTrainer on {self.device}")
        logger.info(f"Memory optimizations: Mixed precision={self.use_mixed_precision}, "
                   f"Gradient accumulation={self.use_gradient_accumulation}, "
                   f"Gradient checkpointing={self.use_gradient_checkpointing}")
    
    def train_epoch(self, dataloader):
        """
        Train for one epoch
        
        Args:
            dataloader: DataLoader providing training data
            
        Returns:
            dict: Training metrics for the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0
        start_time = time.time()
        
        # Set gradients to zero before starting
        self.optimizer.zero_grad()
        
        for i, (g, labels) in enumerate(dataloader):
            batch_count += 1
            
            # Move data to device
            g = g.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass with optional mixed precision
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(g)
                    loss = self.criterion(outputs, labels)
                
                # Scale loss and perform backward pass
                self.scaler.scale(loss).backward()
                
                # Accumulate gradients if enabled
                if (i + 1) % self.gradient_accumulation_steps == 0 or (i + 1) == len(dataloader):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # Standard forward and backward pass
                outputs = self.model(g)
                loss = self.criterion(outputs, labels)
                
                # Apply gradient accumulation if enabled
                if self.use_gradient_accumulation:
                    loss = loss / self.gradient_accumulation_steps
                
                loss.backward()
                
                # Step optimizer after accumulation steps
                if not self.use_gradient_accumulation or (i + 1) % self.gradient_accumulation_steps == 0 or (i + 1) == len(dataloader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Update metrics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Increment global step
            self.global_step += 1
            
            # Clean memory periodically
            if (i + 1) % 10 == 0:
                clean_memory()
        
        # Calculate metrics
        epoch_time = time.time() - start_time
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'epoch_time': epoch_time
        }
        
        return metrics
    
    def evaluate(self, dataloader):
        """
        Evaluate the model on validation data
        
        Args:
            dataloader: DataLoader providing validation data
            
        Returns:
            dict: Evaluation metrics
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        predictions = []
        true_labels = []
        start_time = time.time()
        batch_count = 0
        
        with torch.no_grad():
            for g, labels in dataloader:
                batch_count += 1
                
                # Move data to device
                g = g.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass (no mixed precision needed for evaluation)
                outputs = self.model(g)
                loss = self.criterion(outputs, labels)
                
                # Update metrics
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store predictions and true labels for detailed metrics
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        eval_time = time.time() - start_time
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = val_loss / batch_count if batch_count > 0 else 0.0
        
        metrics = {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'eval_time': eval_time,
            'predictions': predictions,
            'true_labels': true_labels
        }
        
        return metrics
    
    def train(self, train_dataloader, val_dataloader=None, num_epochs=10):
        """
        Train the model
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data (optional)
            num_epochs: Number of epochs to train
            
        Returns:
            dict: Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            with track_memory_usage(f"Training epoch {epoch+1}"):
                train_metrics = self.train_epoch(train_dataloader)
            
            # Evaluate if validation data is provided
            val_metrics = {}
            if val_dataloader:
                with track_memory_usage(f"Validation epoch {epoch+1}"):
                    val_metrics = self.evaluate(val_dataloader)
                
                # Check for improved validation metric
                if val_metrics['val_loss'] < self.best_val_metric:
                    self.best_val_metric = val_metrics['val_loss']
                    if self.config.get('save_best_model', True):
                        self.save_checkpoint(is_best=True)
            
            # Combine metrics and add to history
            epoch_metrics = {
                'epoch': epoch + 1,
                **train_metrics,
                **val_metrics
            }
            self.training_history.append(epoch_metrics)
            
            # Log progress
            log_msg = f"Epoch {epoch+1}/{num_epochs} - Loss: {train_metrics['loss']:.4f}"
            log_msg += f", Accuracy: {train_metrics['accuracy']:.4f}"
            if val_dataloader:
                log_msg += f", Val Loss: {val_metrics['val_loss']:.4f}"
                log_msg += f", Val Accuracy: {val_metrics['val_accuracy']:.4f}"
            logger.info(log_msg)
            
            # Save regular checkpoint if configured
            if (epoch + 1) % self.config.get('checkpoint_interval', 5) == 0:
                self.save_checkpoint()
            
            # Clean memory between epochs
            clean_memory()
        
        logger.info(f"Training completed. Best validation loss: {self.best_val_metric:.4f}")
        return self.training_history
    
    def save_checkpoint(self, is_best=False):
        """
        Save model checkpoint
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_val_metric': self.best_val_metric,
            'config': self.config
        }
        
        # Create directory if it doesn't exist
        save_dir = self.config.get('model_save_dir', './models')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save checkpoint
        filename = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        checkpoint_path = os.path.join(save_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model separately if this is the best
        if is_best:
            best_path = os.path.join(save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_metric = checkpoint['best_val_metric']
        
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return True

class SparseGraphDataset(Dataset):
    """Memory-efficient dataset for sparse graph data"""
    
    def __init__(self, graphs, labels):
        """
        Initialize the dataset
        
        Args:
            graphs: List of DGL graphs
            labels: List of labels
        """
        self.graphs = graphs
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]
