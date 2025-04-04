#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
視覺化模組

此模組負責:
1. 圖結構視覺化
2. 模型預測結果視覺化
3. 網路攻擊檢測結果呈現
"""

import networkx as nx
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
    """網路圖視覺化工具"""
    
    def __init__(self, figsize=(12, 10)):
        """
        初始化視覺化工具
        
        參數:
            figsize (tuple): 圖表尺寸
        """
        self.figsize = figsize
    
    def visualize_graph(self, g, node_labels=None, node_colors=None, title=None, save_path=None):
        """
        視覺化圖結構
        
        參數:
            g (dgl.DGLGraph): DGL 圖
            node_labels (list, optional): 節點標籤列表
            node_colors (list, optional): 節點顏色列表
            title (str, optional): 圖表標題
            save_path (str, optional): 儲存路徑
            
        返回:
            matplotlib.figure.Figure: 圖表物件
        """
        plt.figure(figsize=self.figsize)
        
        # 轉換為 NetworkX 圖
        nx_g = dgl.to_networkx(g)
        
        # 定義佈局
        if g.num_nodes() < 100:
            pos = nx.spring_layout(nx_g, seed=42)
        else:
            # 大型圖使用更快的佈局算法
            pos = nx.kamada_kawai_layout(nx_g)
        
        # 準備節點顏色
        if node_colors is None and node_labels is not None:
            # 根據標籤設置顏色
            cmap = plt.cm.get_cmap('viridis', len(set(node_labels)))
            node_colors = [cmap(label) for label in node_labels]
        elif node_colors is None:
            # 預設顏色
            node_colors = 'skyblue'
        
        # 繪製節點
        nx.draw_networkx_nodes(
            nx_g, pos, 
            node_color=node_colors, 
            node_size=300, 
            alpha=0.8
        )
        
        # 繪製邊
        nx.draw_networkx_edges(
            nx_g, pos, 
            width=1.0, 
            alpha=0.5, 
            arrows=True,
            arrowsize=10,
            arrowstyle='->'
        )
        
        # 繪製節點標籤
        if g.num_nodes() < 50:  # 只在節點數量適中時顯示標籤
            nx.draw_networkx_labels(
                nx_g, pos, 
                font_size=10, 
                font_family='sans-serif'
            )
        
        # 設置標題
        if title:
            plt.title(title, fontsize=16)
        
        plt.axis('off')
        
        # 儲存圖表
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"圖表已儲存至: {save_path}")
        
        plt.tight_layout()
        
        return plt.gcf()
    
    def visualize_embeddings(self, embeddings, labels, label_names=None, title=None, save_path=None):
        """
        視覺化節點嵌入
        
        參數:
            embeddings (torch.Tensor or np.ndarray): 節點嵌入矩陣 [num_nodes, embed_dim]
            labels (list or torch.Tensor): 節點標籤
            label_names (list, optional): 標籤名稱列表
            title (str, optional): 圖表標題
            save_path (str, optional): 儲存路徑
            
        返回:
            matplotlib.figure.Figure: 圖表物件
        """
        plt.figure(figsize=self.figsize)
        
        # 確保嵌入是 NumPy 陣列
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        
        # 確保標籤是 NumPy 陣列
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # 降維到 2D 以便視覺化
        tsne = TSNE(n_components=2, random_state=42)
        embed_2d = tsne.fit_transform(embeddings)
        
        # 準備標籤名稱
        if label_names is None:
            label_names = {i: f"Class {i}" for i in set(labels)}
        
        # 繪製散點圖
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
        
        # 設置標題
        if title:
            plt.title(title, fontsize=16)
        else:
            plt.title("節點嵌入 t-SNE 視覺化", fontsize=16)
        
        # 儲存圖表
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"嵌入視覺化已儲存至: {save_path}")
        
        plt.tight_layout()
        
        return plt.gcf()
    
    def visualize_attack_detection(self, timestamps, scores, threshold=0.5, 
                                  attack_indices=None, title=None, save_path=None):
        """
        視覺化攻擊檢測結果
        
        參數:
            timestamps (list): 時間戳記列表
            scores (list): 異常分數列表
            threshold (float): 攻擊判定閾值
            attack_indices (list, optional): 已知攻擊的索引
            title (str, optional): 圖表標題
            save_path (str, optional): 儲存路徑
            
        返回:
            matplotlib.figure.Figure: 圖表物件
        """
        plt.figure(figsize=self.figsize)
        
        # 繪製異常分數
        plt.plot(timestamps, scores, 'b-', alpha=0.6, label='異常分數')
        
        # 繪製閾值線
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'閾值 ({threshold})')
        
        # 標記檢測到的攻擊
        detected_indices = [i for i, score in enumerate(scores) if score > threshold]
        if detected_indices:
            plt.scatter(
                [timestamps[i] for i in detected_indices], 
                [scores[i] for i in detected_indices], 
                color='red', s=50, label='檢測到的攻擊'
            )
        
        # 標記已知攻擊
        if attack_indices:
            plt.scatter(
                [timestamps[i] for i in attack_indices], 
                [scores[i] for i in attack_indices], 
                color='orange', marker='x', s=100, label='已知攻擊'
            )
        
        plt.xlabel('時間')
        plt.ylabel('異常分數')
        
        # 設置標題
        if title:
            plt.title(title, fontsize=16)
        else:
            plt.title("網路攻擊檢測結果", fontsize=16)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 儲存圖表
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"攻擊檢測視覺化已儲存至: {save_path}")
        
        plt.tight_layout()
        
        return plt.gcf()
    
    def visualize_dynamic_graph_evolution(self, graph_snapshots, node_labels_list=None, 
                                         title_template="時間 {}", save_path=None):
        """
        視覺化動態圖演化 (動畫)
        
        參數:
            graph_snapshots (list): DGL 圖快照列表
            node_labels_list (list, optional): 每個快照的節點標籤列表
            title_template (str): 標題模板
            save_path (str, optional): 儲存路徑 (GIF 格式)
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # 將所有圖轉換為 NetworkX 圖
        nx_graphs = [dgl.to_networkx(g) for g in graph_snapshots]
        
        # 獲取所有圖的節點並計算固定佈局
        all_nodes = set()
        for g in nx_graphs:
            all_nodes.update(g.nodes())
        
        all_nodes = sorted(list(all_nodes))
        
        # 計算固定佈局 (使用最後一個快照來計算佈局)
        pos = nx.spring_layout(nx_graphs[-1], seed=42)
        
        # 定義更新函數
        def update(frame):
            ax.clear()
            
            g = nx_graphs[frame]
            
            # 獲取節點顏色
            node_colors = 'skyblue'
            if node_labels_list is not None and frame < len(node_labels_list):
                labels = node_labels_list[frame]
                cmap = plt.cm.get_cmap('viridis', len(set(labels)))
                node_colors = [cmap(label) for label in labels]
            
            # 繪製節點
            nx.draw_networkx_nodes(
                g, pos, 
                node_color=node_colors, 
                node_size=300, 
                alpha=0.8,
                ax=ax
            )
            
            # 繪製邊
            nx.draw_networkx_edges(
                g, pos, 
                width=1.0, 
                alpha=0.5, 
                arrows=True,
                arrowsize=10,
                arrowstyle='->',
                ax=ax
            )
            
            # 繪製標籤
            if len(g.nodes()) < 50:
                nx.draw_networkx_labels(
                    g, pos, 
                    font_size=10, 
                    font_family='sans-serif',
                    ax=ax
                )
            
            # 設置標題
            ax.set_title(title_template.format(frame), fontsize=16)
            ax.axis('off')
            
            return ax
        
        # 建立動畫
        ani = FuncAnimation(
            fig, update, frames=len(nx_graphs), 
            interval=1000, repeat=True, blit=False
        )
        
        # 儲存動畫
        if save_path:
            ani.save(save_path, writer='pillow', fps=1)
            logger.info(f"動態圖演化動畫已儲存至: {save_path}")
        
        plt.tight_layout()
        plt.close()
        
        return ani
    
    def plot_feature_importance(self, feature_names, importances, title=None, 
                               top_n=10, save_path=None):
        """
        繪製特徵重要性
        
        參數:
            feature_names (list): 特徵名稱列表
            importances (list): 特徵重要性列表
            title (str, optional): 圖表標題
            top_n (int): 顯示前 N 個重要特徵
            save_path (str, optional): 儲存路徑
            
        返回:
            matplotlib.figure.Figure: 圖表物件
        """
        plt.figure(figsize=(10, 8))
        
        # 創建特徵重要性 DataFrame
        feature_imp = pd.DataFrame({
            '特徵': feature_names,
            '重要性': importances
        })
        
        # 排序並選取前 N 個
        feature_imp = feature_imp.sort_values('重要性', ascending=False).head(top_n)
        
        # 繪製條形圖
        sns.barplot(x='重要性', y='特徵', data=feature_imp, palette='viridis')
        
        # 設置標題
        if title:
            plt.title(title, fontsize=16)
        else:
            plt.title(f"前 {top_n} 個重要特徵", fontsize=16)
        
        plt.tight_layout()
        
        # 儲存圖表
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            logger.info(f"特徵重要性圖表已儲存至: {save_path}")
        
        return plt.gcf()
    
    def plot_real_time_detection(self, times, attack_probs, threshold=0.5, 
                                window_size=100, update_interval=1, save_path=None):
        """
        即時繪製攻擊檢測結果 (適用於互動式環境)
        
        參數:
            times (list): 時間列表
            attack_probs (list): 攻擊概率列表
            threshold (float): 攻擊閾值
            window_size (int): 顯示窗口大小
            update_interval (float): 更新間隔 (秒)
            save_path (str, optional): 儲存路徑
            
        注意: 此函數需要在支持互動式繪圖的環境中運行 (如 Jupyter Notebook)
        """
        import matplotlib.animation as animation
        from IPython.display import display, clear_output
        
        plt.figure(figsize=self.figsize)
        
        line, = plt.plot([], [], 'b-', alpha=0.7)
        threshold_line, = plt.plot([], [], 'r--', alpha=0.7)
        alert_points, = plt.plot([], [], 'ro', markersize=8)
        
        plt.xlabel('時間')
        plt.ylabel('攻擊概率')
        plt.title('即時網路攻擊檢測')
        plt.grid(True, alpha=0.3)
        
        # 初始化函數
        def init():
            line.set_data([], [])
            threshold_line.set_data([], [])
            alert_points.set_data([], [])
            return line, threshold_line, alert_points
        
        # 更新函數
        def update(frame):
            # 獲取當前視圖範圍內的資料
            start_idx = max(0, frame - window_size)
            end_idx = frame + 1
            
            current_times = times[start_idx:end_idx]
            current_probs = attack_probs[start_idx:end_idx]
            
            # 更新線圖
            line.set_data(current_times, current_probs)
            
            # 更新閾值線
            threshold_line.set_data([current_times[0], current_times[-1]], [threshold, threshold])
            
            # 更新警報點
            alert_indices = [i for i, prob in enumerate(current_probs) if prob > threshold]
            alert_times = [current_times[i] for i in alert_indices]
            alert_probs = [current_probs[i] for i in alert_indices]
            alert_points.set_data(alert_times, alert_probs)
            
            # 動態調整 x 軸範圍
            plt.xlim(current_times[0], current_times[-1])
            plt.ylim(0, 1.05)
            
            return line, threshold_line, alert_points
        
        # 創建動畫
        ani = animation.FuncAnimation(
            plt.gcf(), update, frames=len(times),
            init_func=init, blit=True, interval=update_interval*1000
        )
        
        # 儲存動畫
        if save_path:
            ani.save(save_path, writer='pillow', fps=10)
            logger.info(f"即時檢測動畫已儲存至: {save_path}")
        
        plt.tight_layout()
        
        return ani

# 測試視覺化工具
if __name__ == "__main__":
    import torch
    import numpy as np
    
    # 建立一個簡單的圖
    src = torch.tensor([0, 1, 2, 3, 4])
    dst = torch.tensor([1, 2, 3, 4, 0])
    g = dgl.graph((src, dst))
    
    # 添加節點特徵
    num_nodes = 5
    in_dim = 10
    h = torch.randn(num_nodes, in_dim)
    g.ndata['h'] = h
    
    # 創建一些標籤
    labels = [0, 1, 0, 1, 0]
    
    # 初始化視覺化工具
    visualizer = NetworkVisualizer()
    
    # 測試圖視覺化
    visualizer.visualize_graph(g, node_labels=labels, title="網路圖視覺化測試")
    
    # 測試嵌入視覺化
    embeddings = torch.randn(num_nodes, 16)  # 模擬節點嵌入
    visualizer.visualize_embeddings(
        embeddings, 
        labels, 
        label_names={0: "正常", 1: "攻擊"}, 
        title="節點嵌入視覺化測試"
    )
    
    # 測試攻擊檢測視覺化
    timestamps = list(range(100))
    scores = [0.2 + 0.6 * np.sin(i/10) for i in range(100)]  # 模擬異常分數
    attack_indices = [30, 31, 32, 60, 61, 62, 63]  # 模擬已知攻擊
    
    visualizer.visualize_attack_detection(
        timestamps, 
        scores, 
        threshold=0.7,
        attack_indices=attack_indices,
        title="攻擊檢測視覺化測試"
    )
    
    # 測試特徵重要性視覺化
    feature_names = [f"Feature {i}" for i in range(15)]
    importances = np.random.rand(15)
    
    visualizer.plot_feature_importance(
        feature_names, 
        importances, 
        title="特徵重要性測試", 
        top_n=10
    )
    
    # 注意: 動態圖演化和即時檢測視覺化需要互動式環境才能正常顯示
    print("視覺化工具測試完成")