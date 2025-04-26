#!/usr/bin/env python
# coding: utf-8 -*-

"""
增強評估指標模組

提供針對網路入侵檢測系統 (NIDS) 的增強評估指標，包括:
1. FPR@TPR (在特定查全率下的誤報率)
2. 特定閾值下的精確度-召回率權衡
3. AUC-ROC 和 AUC-PR 曲線分析
4. 異常檢測特定指標
"""

import numpy as np
import pandas as pd
import torch
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)
from typing import Dict, List, Union, Tuple, Any, Optional

# 配置日誌
logger = logging.getLogger(__name__)

def compute_fpr_at_tpr(
        y_true: np.ndarray, 
        y_proba: np.ndarray, 
        target_tpr: float = 0.95,
        pos_label: int = 1) -> dict:
    """
    計算在指定TPR(真陽率)下的FPR(誤報率)
    
    參數:
        y_true: 真實標籤 (0或1)
        y_proba: 正類別的預測概率
        target_tpr: 目標TPR閾值 (默認0.95)
        pos_label: 正類別標籤 (默認1)
        
    返回:
        dict: 包含FPR@TPR和對應閾值的字典
    """
    # 確保輸入為numpy數組
    y_true_np = np.array(y_true)
    y_proba_np = np.array(y_proba)
    
    # 處理多類別問題
    if y_proba_np.ndim > 1 and y_proba_np.shape[1] > 1:
        # 如果是二分類問題，取正類別的概率
        if len(np.unique(y_true_np)) <= 2:
            y_proba_np = y_proba_np[:, pos_label]
        else:
            # 對於多類別問題，我們需要將其轉換為一對其餘的評估
            # 創建一個二元標籤數組，原本是pos_label類別的為1，其餘為0
            y_true_binary = (y_true_np == pos_label).astype(int)
            return compute_fpr_at_tpr(y_true_binary, y_proba_np[:, pos_label], target_tpr)
    
    # 計算ROC曲線
    fpr, tpr, thresholds = roc_curve(y_true_np, y_proba_np, pos_label=pos_label)
    
    # 找到第一個大於等於target_tpr的索引
    idx = np.argmax(tpr >= target_tpr)
    
    # 獲取對應的FPR和閾值
    fpr_at_target_tpr = fpr[idx]
    threshold_at_target_tpr = thresholds[idx]
    actual_tpr = tpr[idx]
    
    return {
        'fpr_at_target_tpr': fpr_at_target_tpr,
        'threshold': threshold_at_target_tpr,
        'actual_tpr': actual_tpr,
        'target_tpr': target_tpr
    }

def compute_multiclass_fpr_at_tpr(
        y_true: np.ndarray, 
        y_proba: np.ndarray, 
        target_tpr: float = 0.95) -> dict:
    """
    針對多類別問題計算每個類別在指定TPR下的FPR
    
    參數:
        y_true: 真實標籤 (類別編碼)
        y_proba: 預測概率矩陣 [樣本數, 類別數]
        target_tpr: 目標TPR閾值 (默認0.95)
        
    返回:
        dict: 包含每個類別的FPR@TPR和對應閾值的字典
    """
    # 確保輸入為numpy數組
    y_true_np = np.array(y_true)
    y_proba_np = np.array(y_proba)
    
    # 確保y_proba是二維的
    if y_proba_np.ndim == 1:
        y_proba_np = y_proba_np.reshape(-1, 1)
    
    # 獲取類別數量
    n_classes = y_proba_np.shape[1]
    
    # 對每個類別計算指標
    results = {}
    for i in range(n_classes):
        # 創建一對其餘的評估
        y_true_binary = (y_true_np == i).astype(int)
        
        # 計算這個類別的評估指標
        class_result = compute_fpr_at_tpr(y_true_binary, y_proba_np[:, i], target_tpr, pos_label=1)
        results[f'class_{i}'] = class_result
    
    # 計算宏平均和加權平均
    fprs = [results[f'class_{i}']['fpr_at_target_tpr'] for i in range(n_classes)]
    
    # 宏平均 - 每個類別的簡單平均
    macro_avg = np.mean(fprs)
    
    # 加權平均 - 基於類別支持度的加權平均
    class_counts = np.bincount(y_true_np, minlength=n_classes)
    class_weights = class_counts / class_counts.sum()
    weighted_avg = np.sum(fprs * class_weights)
    
    # 添加平均指標
    results['macro_avg'] = {'fpr_at_target_tpr': macro_avg}
    results['weighted_avg'] = {'fpr_at_target_tpr': weighted_avg}
    
    return results

def compute_metrics_at_threshold(
        y_true: np.ndarray, 
        y_proba: np.ndarray, 
        threshold: float = 0.5,
        pos_label: int = 1) -> dict:
    """
    計算特定閾值下的各種指標
    
    參數:
        y_true: 真實標籤 (0或1)
        y_proba: 正類別的預測概率
        threshold: 分類閾值 (默認0.5)
        pos_label: 正類別標籤 (默認1)
        
    返回:
        dict: 各種指標的字典
    """
    # 使用閾值將概率轉換為預測標籤
    y_pred = (y_proba >= threshold).astype(int)
    
    # 計算指標
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        'recall': recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        'f1': f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        'threshold': threshold
    }
    
    # 計算混淆矩陣
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 添加其他指標
    metrics.update({
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,  # 誤報率
        'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0,  # 真陽率 (召回率)
        'tnr': tn / (tn + fp) if (tn + fp) > 0 else 0,  # 真陰率 (特異性)
        'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0   # 漏報率
    })
    
    return metrics

def compute_roc_pr_curves(
        y_true: np.ndarray, 
        y_proba: np.ndarray, 
        pos_label: int = 1) -> dict:
    """
    計算ROC和PR曲線的數據點
    
    參數:
        y_true: 真實標籤 (0或1)
        y_proba: 正類別的預測概率
        pos_label: 正類別標籤 (默認1)
        
    返回:
        dict: 包含ROC和PR曲線數據的字典
    """
    # 計算ROC曲線
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    
    # 計算PR曲線
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba, pos_label=pos_label)
    pr_auc = auc(recall, precision)  # PR曲線下面積
    average_precision = average_precision_score(y_true, y_proba, pos_label=pos_label)
    
    return {
        'roc': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': roc_thresholds.tolist() if len(roc_thresholds) > 0 else [],
            'auc': roc_auc
        },
        'pr': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': pr_thresholds.tolist() if len(pr_thresholds) > 0 else [],
            'auc': pr_auc,
            'average_precision': average_precision
        }
    }

def find_optimal_threshold(
        y_true: np.ndarray, 
        y_proba: np.ndarray, 
        optimize_for: str = 'f1',
        pos_label: int = 1) -> dict:
    """
    尋找最佳閾值
    
    參數:
        y_true: 真實標籤 (0或1)
        y_proba: 正類別的預測概率
        optimize_for: 要最大化的指標 ('f1', 'precision', 'recall', 'accuracy', 'balanced_accuracy')
        pos_label: 正類別標籤 (默認1)
        
    返回:
        dict: 最佳閾值和對應指標
    """
    # 計算ROC用於閾值選擇
    fpr, tpr, thresholds = roc_curve(y_true, y_proba, pos_label=pos_label)
    
    # 確保thresholds不為空
    if len(thresholds) == 0:
        return {
            'threshold': 0.5,
            'method': optimize_for,
            'value': 0.0
        }
    
    # 對於每個閾值，計算所需的指標
    metrics = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        # 計算不同的指標
        if optimize_for == 'f1':
            score = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        elif optimize_for == 'precision':
            score = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        elif optimize_for == 'recall':
            score = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        elif optimize_for == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        elif optimize_for == 'balanced_accuracy':
            # 平衡準確率 = (TPR + TNR) / 2
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = (tpr + tnr) / 2
        elif optimize_for == 'g_mean':
            # G-mean = sqrt(TPR * TNR)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = np.sqrt(tpr * tnr) if tpr > 0 and tnr > 0 else 0
        else:
            score = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        
        metrics.append((threshold, score))
    
    # 找到最佳閾值
    best_threshold, best_score = max(metrics, key=lambda x: x[1])
    
    return {
        'threshold': best_threshold,
        'method': optimize_for,
        'value': best_score
    }

def evaluate_nids_metrics(
        y_true: np.ndarray, 
        y_proba: np.ndarray, 
        target_tpr_levels: List[float] = [0.90, 0.95, 0.99],
        class_names: Optional[List[str]] = None,
        optimize_for: str = 'f1',
        pos_label: int = 1) -> Dict[str, Any]:
    """
    針對網路入侵檢測系統 (NIDS) 計算全面的評估指標
    
    參數:
        y_true: 真實標籤 (0或1)
        y_proba: 預測概率
        target_tpr_levels: 目標TPR列表 (默認[0.90, 0.95, 0.99])
        class_names: 類別名稱
        optimize_for: 尋找最佳閾值時要優化的指標
        pos_label: 正類別標籤 (默認1)
    
    返回:
        dict: 全面的評價指標
    """
    # 初始化結果字典
    results = {}
    
    # 轉換為numpy數組
    y_true_np = np.array(y_true)
    
    # 處理輸入概率
    if isinstance(y_proba, torch.Tensor):
        y_proba_np = y_proba.detach().cpu().numpy()
    else:
        y_proba_np = np.array(y_proba)
    
    # 確保y_proba是二維的
    if y_proba_np.ndim == 1:
        # 如果是一維的，假設是二分類的正類別概率
        y_proba_np = np.vstack((1 - y_proba_np, y_proba_np)).T
    
    # 檢查是二分類還是多分類問題
    unique_classes = np.unique(y_true_np)
    is_binary = len(unique_classes) <= 2
    
    # 使用閾值0.5的基本指標
    if is_binary:
        # 二分類問題
        y_pred = (y_proba_np[:, pos_label] >= 0.5).astype(int)
        
        # 基本分類指標
        results['basic_metrics'] = {
            'accuracy': accuracy_score(y_true_np, y_pred),
            'precision': precision_score(y_true_np, y_pred, pos_label=pos_label, zero_division=0),
            'recall': recall_score(y_true_np, y_pred, pos_label=pos_label, zero_division=0),
            'f1': f1_score(y_true_np, y_pred, pos_label=pos_label, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true_np, y_pred).tolist()
        }
        
        # 計算不同TPR下的FPR
        fpr_at_tpr_results = {}
        for target_tpr in target_tpr_levels:
            fpr_at_tpr = compute_fpr_at_tpr(y_true_np, y_proba_np[:, pos_label], target_tpr, pos_label)
            fpr_at_tpr_results[f'fpr_at_{target_tpr}_tpr'] = fpr_at_tpr
        
        results['fpr_at_tpr'] = fpr_at_tpr_results
        
        # 計算ROC和PR曲線
        results['curves'] = compute_roc_pr_curves(y_true_np, y_proba_np[:, pos_label], pos_label)
        
        # 尋找最佳閾值
        results['optimal_threshold'] = find_optimal_threshold(y_true_np, y_proba_np[:, pos_label], optimize_for, pos_label)
        
        # 最佳閾值下的指標
        best_threshold = results['optimal_threshold']['threshold']
        results['metrics_at_optimal_threshold'] = compute_metrics_at_threshold(
            y_true_np, y_proba_np[:, pos_label], best_threshold, pos_label
        )
    else:
        # 多分類問題
        y_pred = np.argmax(y_proba_np, axis=1)
        
        # 基本分類指標
        results['basic_metrics'] = {
            'accuracy': accuracy_score(y_true_np, y_pred),
            'precision_macro': precision_score(y_true_np, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true_np, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true_np, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true_np, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true_np, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true_np, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true_np, y_pred).tolist()
        }
        
        # 添加多類別的FPR@TPR
        for target_tpr in target_tpr_levels:
            multiclass_fpr_at_tpr = compute_multiclass_fpr_at_tpr(y_true_np, y_proba_np, target_tpr)
            key = f'multiclass_fpr_at_{target_tpr}_tpr'
            results[key] = {
                'per_class': {f'class_{i}': v for i, v in enumerate(multiclass_fpr_at_tpr) if isinstance(i, int)},
                'macro_avg': multiclass_fpr_at_tpr.get('macro_avg', {'fpr_at_target_tpr': 0})['fpr_at_target_tpr'],
                'weighted_avg': multiclass_fpr_at_tpr.get('weighted_avg', {'fpr_at_target_tpr': 0})['fpr_at_target_tpr']
            }
        
        # 對於多類別，我們為每個類別計算"一對其餘"的ROC和PR
        per_class_curves = {}
        for i, class_name in enumerate(unique_classes):
            # 創建二元標籤
            binary_y_true = (y_true_np == class_name).astype(int)
            
            # 當前類別的概率
            class_prob = y_proba_np[:, i] if i < y_proba_np.shape[1] else np.zeros(len(y_true_np))
            
            # 計算曲線
            class_curves = compute_roc_pr_curves(binary_y_true, class_prob, pos_label=1)
            
            # 保存結果
            class_label = class_names[i] if class_names and i < len(class_names) else f'class_{class_name}'
            per_class_curves[class_label] = class_curves
        
        results['per_class_curves'] = per_class_curves
        
    # 添加分類報告
    report = classification_report(
        y_true_np, 
        y_pred, 
        target_names=class_names,
        zero_division=0,
        output_dict=True
    )
    results['classification_report'] = report
    
    return results

def plot_nids_metrics(metrics: Dict[str, Any], output_path: Optional[str] = None):
    """
    繪製NIDS評估指標的可視化圖表
    
    參數:
        metrics: 評估指標字典
        output_path: 輸出路徑，若為None則顯示圖表
    """
    # 創建多子圖
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. ROC曲線
    if 'curves' in metrics and 'roc' in metrics['curves']:
        roc = metrics['curves']['roc']
        axs[0, 0].plot(roc['fpr'], roc['tpr'], lw=2)
        axs[0, 0].plot([0, 1], [0, 1], 'k--', lw=1)
        axs[0, 0].set_xlim([0.0, 1.0])
        axs[0, 0].set_ylim([0.0, 1.05])
        axs[0, 0].set_xlabel('False Positive Rate')
        axs[0, 0].set_ylabel('True Positive Rate')
        axs[0, 0].set_title(f'ROC Curve (AUC = {roc["auc"]:.3f})')
        axs[0, 0].grid(True, alpha=0.3)
    
    # 2. PR曲線
    if 'curves' in metrics and 'pr' in metrics['curves']:
        pr = metrics['curves']['pr']
        axs[0, 1].plot(pr['recall'], pr['precision'], lw=2)
        axs[0, 1].set_xlim([0.0, 1.0])
        axs[0, 1].set_ylim([0.0, 1.05])
        axs[0, 1].set_xlabel('Recall')
        axs[0, 1].set_ylabel('Precision')
        axs[0, 1].set_title(f'PR Curve (AP = {pr["average_precision"]:.3f})')
        axs[0, 1].grid(True, alpha=0.3)
    
    # 3. FPR@TPR比較
    if 'fpr_at_tpr' in metrics:
        fpr_data = metrics['fpr_at_tpr']
        tpr_levels = sorted([float(k.split('_')[2]) for k in fpr_data.keys()])
        fpr_values = [fpr_data[f'fpr_at_{tpr}_tpr']['fpr_at_target_tpr'] for tpr in tpr_levels]
        
        axs[1, 0].bar(range(len(tpr_levels)), fpr_values, color='skyblue')
        axs[1, 0].set_xticks(range(len(tpr_levels)))
        axs[1, 0].set_xticklabels([f'TPR={tpr}' for tpr in tpr_levels])
        axs[1, 0].set_ylabel('False Positive Rate')
        axs[1, 0].set_title('FPR at Target TPR Levels')
        
        # 添加數值標籤
        for i, v in enumerate(fpr_values):
            axs[1, 0].text(i, v + 0.01, f'{v:.4f}', ha='center')
        
        axs[1, 0].grid(True, alpha=0.3)
    
    # 4. 混淆矩陣
    if 'basic_metrics' in metrics and 'confusion_matrix' in metrics['basic_metrics']:
        cm = np.array(metrics['basic_metrics']['confusion_matrix'])
        
        if len(cm) == 2:  # 二分類
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axs[1, 1])
            axs[1, 1].set_xlabel('Predicted Label')
            axs[1, 1].set_ylabel('True Label')
            axs[1, 1].set_title('Confusion Matrix')
            axs[1, 1].set_xticks([0.5, 1.5])
            axs[1, 1].set_yticks([0.5, 1.5])
            axs[1, 1].set_xticklabels(['0', '1'])
            axs[1, 1].set_yticklabels(['0', '1'])
        else:  # 多分類
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axs[1, 1])
            axs[1, 1].set_xlabel('Predicted Label')
            axs[1, 1].set_ylabel('True Label')
            axs[1, 1].set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"評估指標圖表已保存至: {output_path}")
        plt.close(fig)
    else:
        plt.show()

def plot_fpr_comparison(
        metrics_list: List[Dict[str, Any]], 
        labels: List[str], 
        target_tpr: float = 0.95,
        output_path: Optional[str] = None):
    """
    比較不同模型在相同TPR下的FPR
    
    參數:
        metrics_list: 多個模型的評估指標字典列表
        labels: 模型標籤
        target_tpr: 目標TPR
        output_path: 輸出路徑，若為None則顯示圖表
    """
    if not metrics_list:
        logger.warning("沒有提供評估指標")
        return
    
    # 提取FPR值
    fpr_key = f'fpr_at_{target_tpr}_tpr'
    fpr_values = []
    
    for metrics in metrics_list:
        if 'fpr_at_tpr' in metrics and fpr_key in metrics['fpr_at_tpr']:
            fpr_values.append(metrics['fpr_at_tpr'][fpr_key]['fpr_at_target_tpr'])
        else:
            logger.warning(f"無法在指標中找到 {fpr_key}")
            fpr_values.append(0)  # 使用0作為默認值
    
    # 創建圖表
    plt.figure(figsize=(10, 6))
    
    # 繪製條形圖
    bars = plt.bar(labels, fpr_values, color='skyblue')
    
    # 添加數值標籤
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.0025,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.ylabel(f'FPR at TPR={target_tpr}')
    plt.title(f'False Positive Rate Comparison at TPR={target_tpr}')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"FPR比較圖已保存至: {output_path}")
        plt.close()
    else:
        plt.show()
