#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TGAT Model Implementation Module

Implementation of Temporal Graph Attention Network (TGAT) model,
based on papers "Temporal Graph Attention Network for Recommendation" 
and "Inductive Representation Learning on Temporal Graphs"

This module includes:
1. Temporal Encoding
2. Graph Attention Layers
3. TGAT model architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import math
import logging
import numpy as np
from dgl.nn.pytorch import GATConv

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeEncoding(nn.Module):
    def __init__(self, dimension):
        super(TimeEncoding, self).__init__()
        self.dimension = dimension
        
        # Initialize weights and biases
        self.w = nn.Linear(1, dimension, bias=True)
        
        # Use Xavier initialization for weights
        nn.init.xavier_uniform_(self.w.weight)
        nn.init.zeros_(self.w.bias)
    
    def forward(self, t):
        # Ensure input is float and 2D
        t = t.float()
        if t.dim() == 1:
            t = t.unsqueeze(1)
        
        # Ensure weights are also float
        self.w.weight = nn.Parameter(self.w.weight.float())
        self.w.bias = nn.Parameter(self.w.bias.float())
        
        # Use cosine function for encoding
        output = torch.cos(self.w(t))
        return output

class TemporalGATLayer(nn.Module):
    """Temporal Graph Attention Layer"""
    
    def __init__(self, in_dim, out_dim, time_dim, num_heads=4, feat_drop=0.6, attn_drop=0.6, residual=True):
        """
        Initialize temporal graph attention layer
        
        Parameters:
            in_dim (int): Input feature dimension
            out_dim (int): Output feature dimension
            time_dim (int): Time encoding dimension
            num_heads (int): Number of attention heads
            feat_drop (float): Feature dropout rate
            attn_drop (float): Attention dropout rate
            residual (bool): Whether to use residual connection
        """
        super(TemporalGATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.time_dim = time_dim
        self.num_heads = num_heads
        
        # Time encoder
        self.time_enc = TimeEncoding(time_dim)
        
        # Feature dropout layer
        self.feat_drop = nn.Dropout(feat_drop)
        
        # Define attention layer
        self.gat = GATConv(
            in_feats=in_dim + time_dim,  # Features + Time
            out_feats=out_dim // num_heads,
            num_heads=num_heads,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            residual=residual,
            activation=F.elu,
            allow_zero_in_degree=True
        )
    
    def forward(self, g, h, time_tensor):
        # Get time encoding
        # Check time feature length, expand or truncate if not matching
        if len(time_tensor) != g.num_edges():
            if len(time_tensor) < g.num_edges():
                # If time features are fewer than edges, repeat the last value
                time_tensor = torch.cat([
                    time_tensor, 
                    torch.ones(g.num_edges() - len(time_tensor), device=time_tensor.device) * time_tensor[-1]
                ])
            else:
                # If time features are more than edges, truncate
                time_tensor = time_tensor[:g.num_edges()]
        
        time_emb = self.time_enc(time_tensor)  # [num_edges, time_dim]
        
        # Feature dropout
        h = self.feat_drop(h)
        
        # Ensure time encoding has correct shape
        if time_emb.dim() == 1:
            time_emb = time_emb.unsqueeze(0)
        
        # Combine time features and edge features
        g.edata['time_feat'] = time_emb
        
        def message_func(edges):
            # Use concatenation instead of addition
            time_expanded = edges.data['time_feat'].expand_as(edges.src['h'][:, :self.time_dim])
            return {'m': torch.cat([edges.src['h'], time_expanded], dim=1)}
        
        def reduce_func(nodes):
            # Aggregate messages, take mean
            return {'h_time': nodes.mailbox['m'].mean(1)}
        
        # Apply message passing
        g.update_all(message_func, reduce_func)
        
        # Process time features
        h_time = g.ndata.get('h_time', torch.zeros(h.shape[0], h.shape[1] + self.time_dim, device=h.device))
        
        # Check and adjust h_time shape
        if h_time.shape[1] != self.in_dim + self.time_dim:
            # If shape doesn't match, truncate or pad
            if h_time.shape[1] > self.in_dim + self.time_dim:
                h_time = h_time[:, :self.in_dim + self.time_dim]
            else:
                # Use zero padding
                padding = torch.zeros(h_time.shape[0], self.in_dim + self.time_dim - h_time.shape[1], device=h_time.device)
                h_time = torch.cat([h_time, padding], dim=1)
        
        # Apply GAT layer
        h_new = self.gat(g, h_time)
        
        # Combine multi-head attention results
        h_new = h_new.view(h_new.shape[0], -1)
        
        return h_new

class TGAT(nn.Module):
    """Temporal Graph Attention Network Model"""
    
    def __init__(self, in_dim, hidden_dim, out_dim, time_dim, num_layers=2, num_heads=4, dropout=0.1, num_classes=2):
        """
        Initialize TGAT model
        
        Parameters:
            in_dim (int): Input feature dimension
            hidden_dim (int): Hidden layer dimension
            out_dim (int): Output feature dimension
            time_dim (int): Time encoding dimension
            num_layers (int): Number of TGAT layers
            num_heads (int): Number of attention heads
            dropout (float): Dropout rate
            num_classes (int): Number of classification classes
        """
        super(TGAT, self).__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.time_dim = time_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_classes = num_classes
        
        # Feature projection layer
        self.feat_project = nn.Linear(in_dim, hidden_dim)
        
        # TGAT layers
        self.layers = nn.ModuleList()
        
        # First layer projects input features to hidden dimension
        self.layers.append(
            TemporalGATLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                time_dim=time_dim,
                num_heads=num_heads,
                feat_drop=dropout,
                attn_drop=dropout
            )
        )
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.layers.append(
                TemporalGATLayer(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    time_dim=time_dim,
                    num_heads=num_heads,
                    feat_drop=dropout,
                    attn_drop=dropout
                )
            )
        
        # Last layer
        if num_layers > 1:
            self.layers.append(
                TemporalGATLayer(
                    in_dim=hidden_dim,
                    out_dim=out_dim,
                    time_dim=time_dim,
                    num_heads=num_heads,
                    feat_drop=dropout,
                    attn_drop=dropout
                )
            )
        
        # Classification layer
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, num_classes)
        )
        
        # Initialize parameters
        self.reset_parameters()
        
        logger.info(f"Initialized TGAT model: {num_layers} layers, {num_heads} attention heads, {num_classes} classification classes")
    
    def reset_parameters(self):
        """Reset parameters"""
        gain = nn.init.calculate_gain('relu')
        
        # Initialize feature projection layer
        nn.init.xavier_normal_(self.feat_project.weight, gain=gain)
        nn.init.zeros_(self.feat_project.bias)
        
        # Initialize classifier
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)
    
    def forward(self, g):
        """
        Forward propagation
        
        Parameters:
            g (dgl.DGLGraph): Input graph
            
        Returns:
            torch.Tensor: Node classification output [num_nodes, num_classes]
        """
        # Get node features
        h = g.ndata['h']
        
        # Project features to hidden dimension
        h = self.feat_project(h)
        
        # Get edge time features
        time_tensor = g.edata.get('time', None)
        
        # If no time features, use all 0 timestamps
        if time_tensor is None:
            time_tensor = torch.zeros(g.num_edges(), device=h.device)
        
        # Apply TGAT layers
        for i, layer in enumerate(self.layers):
            h = layer(g, h, time_tensor)
        
        # Apply classifier
        logits = self.classifier(h)
        
        return logits
    
    def encode(self, g):
        """
        Encode nodes only, without classification
        
        Parameters:
            g (dgl.DGLGraph): Input graph
            
        Returns:
            torch.Tensor: Node representations [num_nodes, out_dim]
        """
        # Get node features
        h = g.ndata['h']
        
        # Project features to hidden dimension
        h = self.feat_project(h)
        
        # Get edge time features
        time_tensor = g.edata.get('time', None)
        
        # If no time features, use all 0 timestamps
        if time_tensor is None:
            time_tensor = torch.zeros(g.num_edges(), device=h.device)
        
        # Apply TGAT layers
        for i, layer in enumerate(self.layers):
            h = layer(g, h, time_tensor)
        
        return h

# Test TGAT model
if __name__ == "__main__":
    import numpy as np
    import dgl
    
    # Create a simple graph
    src = torch.tensor([0, 1, 2, 3, 4])
    dst = torch.tensor([1, 2, 3, 4, 0])
    g = dgl.graph((src, dst))
    
    # Add node features
    num_nodes = 5
    in_dim = 10
    h = torch.randn(num_nodes, in_dim)
    g.ndata['h'] = h
    
    # Add edge time features
    time = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    g.edata['time'] = time
    
    # Set model parameters
    hidden_dim = 16
    out_dim = 16
    time_dim = 8
    num_layers = 2
    num_heads = 4
    dropout = 0.1
    num_classes = 3
    
    # Initialize model
    model = TGAT(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        time_dim=time_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        num_classes=num_classes
    )
    
    # Forward propagation
    logits = model(g)
    
    print(f"Model output shape: {logits.shape}")
    print(f"Expected shape: [5, 3]")
    
    # Test node encoding
    node_embeddings = model.encode(g)
    print(f"Node representation shape: {node_embeddings.shape}")
    print(f"Expected shape: [5, 16]")
