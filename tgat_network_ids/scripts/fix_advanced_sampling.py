#!/usr/bin/env python
# coding: utf-8 -*-

"""
修復 advanced_sampling.py 檔案中的語法錯誤，主要是補上缺少的逗號和修復未閉合的括號
"""

import re
import sys

def fix_advanced_sampling_file(file_path):
    print(f"修復檔案: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修復缺少的逗號
    content = content.replace("current_importance.get(i 0)", "current_importance.get(i, 0)")
    content = content.replace("current_importance.get(node 0)", "current_importance.get(node, 0)")
    
    # 修復np.random.choice參數中缺少的逗號
    content = content.replace("current_nodes\n                    size=", "current_nodes,\n                    size=")
    content = content.replace("size=min(current_sample_size len(current_nodes))", 
                           "size=min(current_sample_size, len(current_nodes))")
    content = content.replace("replace=False\n                    p=", "replace=False,\n                    p=")
    
    # 修復語法錯誤：node_to_idx字典推導式
    content = content.replace("{node: i for i node in enumerate(current_nodes)}", 
                           "{node: i for i, node in enumerate(current_nodes)}")
    
    # 修復未閉合的括號
    content = re.sub(r'sample_probs = \[current_importance\.get\(node, 0\) / importance_sum for\s*$', 
                   'sample_probs = [current_importance.get(node, 0) / importance_sum for node in current_nodes]', 
                   content, flags=re.MULTILINE)
    
    # 修復其他可能的缺少逗號的情況
    content = content.replace("isinstance(current_graph nx.Graph)", "isinstance(current_graph, nx.Graph)")
    
    # 修復其他可能的缺少參數的函數調用
    content = content.replace("compute_graph_stats(G\n                include_centrality=", 
                           "compute_graph_stats(G, include_centrality=")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("檔案修復完成")
    return True

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = '../src/data/advanced_sampling.py'
    
    fix_advanced_sampling_file(file_path)
