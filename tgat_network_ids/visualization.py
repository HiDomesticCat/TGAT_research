#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization Module

This module is responsible for:
1. Graph structure visualization
2. Model prediction result visualization
3. Network attack detection result presentation
"""

import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation
from sklearn.manifold import TSNE
import logging
import dgl

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NetworkVisualizer:
    """Network Graph Visualization Tool"""
    
    def __init__(self, figsize=(12, 10)):
        """
        Initialize visualization tool
        
        Parameters:
            figsize (tuple): Figure size
        """
        self.figsize = figsize
    
    def visualize_graph(self, g, node_labels=None, node_colors=None, title=None, save_path=None):
        """
        Visualize graph structure
        
        Parameters:
            g (dgl.DGLGraph): DGL graph
            node_labels (list, optional): Node label list
            node_colors (list, optional): Node color list
            title (str, optional): Chart title
            save_path (str, optional): Save path
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        plt.figure(figsize=self.figsize)
        
        # Ensure graph is on CPU, then convert to NetworkX
        if g.device.type != 'cpu':
            g = g.cpu()
        nx_g = dgl.to_networkx(g)
        
        # Define layout
        if g.num_nodes() < 100:
            pos = nx.spring_layout(nx_g, seed=42)
        else:
            # Use faster layout algorithm for large graphs
            pos = nx.kamada_kawai_layout(nx_g)
        
        # Prepare node colors
        if node_colors is None and node_labels is not None:
            # Set colors based on labels
            cmap = matplotlib.colormaps['viridis']
            node_colors = [cmap(label) for label in node_labels]
        elif node_colors is None:
            # Default color
            node_colors = 'skyblue'
        
        # Draw nodes
        nx.draw_networkx_nodes(
            nx_g, pos, 
            node_color=node_colors, 
            node_size=300, 
            alpha=0.8
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            nx_g, pos, 
            width=1.0, 
            alpha=0.5, 
            arrows=True,
            arrowsize=10,
            arrowstyle='->'
        )
        
        # Draw node labels
        if g.num_nodes() < 50:  # Only show labels when node count is moderate
            nx.draw_networkx_labels(
                nx_g, pos, 
                font_size=10, 
                font_family='sans-serif'
            )
        
        # Set title
        plt.title(title or "Network Graph Visualization", fontsize=16)
        
        plt.axis('off')
        
        # Save chart
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Graph saved to: {save_path}")
        
        plt.tight_layout()
        
        return plt.gcf()
    
    def visualize_embeddings(self, embeddings, labels, label_names=None, title=None, save_path=None):
        """
        Visualize node embeddings
        
        Parameters:
            embeddings (torch.Tensor or np.ndarray): Node embedding matrix [num_nodes, embed_dim]
            labels (list or torch.Tensor): Node labels
            label_names (list, optional): Label name list
            title (str, optional): Chart title
            save_path (str, optional): Save path
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        plt.figure(figsize=self.figsize)
        
        # Ensure embeddings are NumPy arrays
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        
        # Ensure labels are NumPy arrays
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # Dynamically adjust perplexity
        n_samples = embeddings.shape[0]
        perplexity = min(30, n_samples - 1)  # Ensure perplexity is less than sample count

        # Reduce dimensions to 2D for visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embed_2d = tsne.fit_transform(embeddings)
        
        # Prepare label names
        if label_names is None:
            label_names = {i: f"Class {i}" for i in set(labels)}
        
        # Draw scatter plot
        unique_labels = set(labels)
        for label in unique_labels:
            indices = [i for i, l in enumerate(labels) if l == label]
            plt.scatter(
                embed_2d[indices, 0], 
                embed_2d[indices, 1], 
                label=label_names.get(label, str(label)),
                alpha=0.7
            )
        
        plt.legend()
        
        # Set title
        plt.title(title or "Node Embeddings t-SNE Visualization", fontsize=16)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        
        # Save chart
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Embedding visualization saved to: {save_path}")
        
        plt.tight_layout()
        
        return plt.gcf()
    
    def visualize_attack_detection(self, timestamps, scores, threshold=0.5, 
                                  attack_indices=None, title=None, save_path=None):
        """
        Visualize attack detection results
        
        Parameters:
            timestamps (list): Timestamp list
            scores (list): Anomaly score list
            threshold (float): Attack determination threshold
            attack_indices (list, optional): Known attack indices
            title (str, optional): Chart title
            save_path (str, optional): Save path
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        plt.figure(figsize=self.figsize)
        
        # Draw anomaly scores
        plt.plot(timestamps, scores, 'b-', alpha=0.6, label='Anomaly Score')
        
        # Draw threshold line
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
        
        # Mark detected attacks
        detected_indices = [i for i, score in enumerate(scores) if score > threshold]
        if detected_indices:
            plt.scatter(
                [timestamps[i] for i in detected_indices], 
                [scores[i] for i in detected_indices], 
                color='red', s=50, label='Detected Attacks'
            )
        
        # Mark known attacks
        if attack_indices:
            plt.scatter(
                [timestamps[i] for i in attack_indices], 
                [scores[i] for i in attack_indices], 
                color='orange', marker='x', s=100, label='Known Attacks'
            )
        
        plt.xlabel('Timestamp')
        plt.ylabel('Anomaly Score')
        
        # Set title
        plt.title(title or "Network Attack Detection", fontsize=16)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save chart
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Attack detection visualization saved to: {save_path}")
        
        plt.tight_layout()
        
        return plt.gcf()
    
    def visualize_dynamic_graph_evolution(self, graph_snapshots, node_labels_list=None, 
                                         title_template="Time {}", save_path=None):
        """
        Visualize dynamic graph evolution (animation)
        
        Parameters:
            graph_snapshots (list): DGL graph snapshot list
            node_labels_list (list, optional): Node label list for each snapshot
            title_template (str): Title template
            save_path (str, optional): Save path (GIF format)
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Ensure all graphs are on CPU, then convert to NetworkX
        nx_graphs = []
        for g in graph_snapshots:
            if g.device.type != 'cpu':
                g = g.cpu()
            nx_graphs.append(dgl.to_networkx(g))
        
        # Get all nodes from all graphs and calculate fixed layout
        all_nodes = set()
        for g in nx_graphs:
            all_nodes.update(g.nodes())
        
        all_nodes = sorted(list(all_nodes))
        
        # Calculate fixed layout (using the last snapshot for layout)
        pos = nx.spring_layout(nx_graphs[-1], seed=42)
        
        # Define update function
        def update(frame):
            ax.clear()
            
            g = nx_graphs[frame]
            
            # Get node colors
            node_colors = 'skyblue'
            if node_labels_list is not None and frame < len(node_labels_list):
                labels = node_labels_list[frame]
                cmap = plt.cm.get_cmap('viridis', len(set(labels)))
                node_colors = [cmap(label) for label in labels]
            
            # Draw nodes
            nx.draw_networkx_nodes(
                g, pos, 
                node_color=node_colors, 
                node_size=300, 
                alpha=0.8,
                ax=ax
            )
            
            # Draw edges
            nx.draw_networkx_edges(
                g, pos, 
                width=1.0, 
                alpha=0.5, 
                arrows=True,
                arrowsize=10,
                arrowstyle='->',
                ax=ax
            )
            
            # Draw labels
            if len(g.nodes()) < 50:
                nx.draw_networkx_labels(
                    g, pos, 
                    font_size=10, 
                    font_family='sans-serif',
                    ax=ax
                )
            
            # Set title
            ax.set_title(title_template.format(frame), fontsize=16)
            ax.axis('off')
            
            return ax
        
        # Create animation
        ani = FuncAnimation(
            fig, update, frames=len(nx_graphs), 
            interval=1000, repeat=True, blit=False
        )
        
        # Save animation
        if save_path:
            ani.save(save_path, writer='pillow', fps=1)
            logger.info(f"Dynamic graph evolution animation saved to: {save_path}")
        
        plt.tight_layout()
        plt.close()
        
        return ani
    
    def plot_feature_importance(self, feature_names, importances, title=None, 
                               top_n=10, save_path=None):
        """
        Plot feature importance
        
        Parameters:
            feature_names (list): Feature name list
            importances (list): Feature importance list
            title (str, optional): Chart title
            top_n (int): Show top N important features
            save_path (str, optional): Save path
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        plt.figure(figsize=(10, 8))
        
        # Create feature importance DataFrame
        feature_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort and select top N
        feature_imp = feature_imp.sort_values('Importance', ascending=False).head(top_n)
        
        # Draw bar chart
        sns.barplot(x='Importance', y='Feature', data=feature_imp, palette='viridis')
        
        # Set title
        plt.title(title or f"Top {top_n} Important Features", fontsize=16)
        plt.xlabel("Importance")
        plt.ylabel("Features")
        
        plt.tight_layout()
        
        # Save chart
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"Feature importance chart saved to: {save_path}")
        
        return plt.gcf()
    
    def plot_real_time_detection(self, times, attack_probs, threshold=0.5, 
                                window_size=100, update_interval=1, save_path=None):
        """
        Real-time plot of attack detection results (for interactive environments)
        
        Parameters:
            times (list): Time list
            attack_probs (list): Attack probability list
            threshold (float): Attack threshold
            window_size (int): Display window size
            update_interval (float): Update interval (seconds)
            save_path (str, optional): Save path
            
        Note: This function needs to be run in an environment that supports interactive plotting (e.g., Jupyter Notebook)
        """
        import matplotlib.animation as animation
        from IPython.display import display, clear_output
        
        plt.figure(figsize=self.figsize)
        
        line, = plt.plot([], [], 'b-', alpha=0.7)
        threshold_line, = plt.plot([], [], 'r--', alpha=0.7)
        alert_points, = plt.plot([], [], 'ro', markersize=8)
        
        plt.xlabel('Time')
        plt.ylabel('Attack Probability')
        plt.title('Real-time Network Attack Detection')
        plt.grid(True, alpha=0.3)
        
        # Initialization function
        def init():
            line.set_data([], [])
            threshold_line.set_data([], [])
            alert_points.set_data([], [])
            return line, threshold_line, alert_points
        
        # Update function
        def update(frame):
            # Get data within current view range
            start_idx = max(0, frame - window_size)
            end_idx = frame + 1
            
            current_times = times[start_idx:end_idx]
            current_probs = attack_probs[start_idx:end_idx]
            
            # Update line chart
            line.set_data(current_times, current_probs)
            
            # Update threshold line
            threshold_line.set_data([current_times[0], current_times[-1]], [threshold, threshold])
            
            # Update alert points
            alert_indices = [i for i, prob in enumerate(current_probs) if prob > threshold]
            alert_times = [current_times[i] for i in alert_indices]
            alert_probs = [current_probs[i] for i in alert_indices]
            alert_points.set_data(alert_times, alert_probs)
            
            # Dynamically adjust x-axis range
            plt.xlim(current_times[0], current_times[-1])
            plt.ylim(0, 1.05)
            
            return line, threshold_line, alert_points
        
        # Create animation
        ani = animation.FuncAnimation(
            plt.gcf(), update, frames=len(times),
            init_func=init, blit=True, interval=update_interval*1000
        )
        
        # Save animation
        if save_path:
            ani.save(save_path, writer='pillow', fps=10)
            logger.info(f"Real-time detection animation saved to: {save_path}")
        
        plt.tight_layout()
        
        return ani

# Test visualization tool
if __name__ == "__main__":
    import torch
    import numpy as np
    
    # Create a simple graph
    src = torch.tensor([0, 1, 2, 3, 4])
    dst = torch.tensor([1, 2, 3, 4, 0])
    g = dgl.graph((src, dst))
    
    # Add node features
    num_nodes = 5
    in_dim = 10
    h = torch.randn(num_nodes, in_dim)
    g.ndata['h'] = h
    
    # Create some labels
    labels = [0, 1, 0, 1, 0]
    
    # Initialize visualization tool
    visualizer = NetworkVisualizer()
    
    # Test graph visualization
    visualizer.visualize_graph(g, node_labels=labels, title="Network Graph Visualization Test")
    
    # Test embedding visualization
    embeddings = torch.randn(num_nodes, 16)  # Simulate node embeddings
    visualizer.visualize_embeddings(
        embeddings, 
        labels, 
        label_names={0: "Normal", 1: "Attack"}, 
        title="Node Embedding Visualization Test"
    )
    
    # Test attack detection visualization
    timestamps = list(range(100))
    scores = [0.2 + 0.6 * np.sin(i/10) for i in range(100)]  # Simulate anomaly scores
    attack_indices = [30, 31, 32, 60, 61, 62, 63]  # Simulate known attacks
    
    visualizer.visualize_attack_detection(
        timestamps, 
        scores, 
        threshold=0.7,
        attack_indices=attack_indices,
        title="Attack Detection Visualization Test"
    )
    
    # Test feature importance visualization
    feature_names = [f"Feature {i}" for i in range(15)]
    importances = np.random.rand(15)
    
    visualizer.plot_feature_importance(
        feature_names, 
        importances, 
        title="Feature Importance Test", 
        top_n=10
    )
    
    # Note: Dynamic graph evolution and real-time detection visualization require interactive environment
    print("Visualization tool test completed")
