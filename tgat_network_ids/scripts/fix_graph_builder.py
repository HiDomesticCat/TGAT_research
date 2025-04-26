#!/usr/bin/env python
# coding: utf-8 -*-

"""
修復 optimized_graph_builder.py 檔案中的語法錯誤，主要是補上缺少的逗號和修復未閉合的括號
"""

import re
import sys

def fix_graph_builder_file(file_path):
    print(f"修復檔案: {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 修復函數定義中缺少的逗號
    content = re.sub(r'def add_edges_in_batches\(self ([^,]+) ([^,]+) ([^,]+) ([^,)]+)=None ([^,)]+)=None\):',
                    r'def add_edges_in_batches(self, \1, \2, \3, \4=None, \5=None):',
                    content)
    
    # 修復未閉合的括號
    content = re.sub(r'filtered_edge_feats = \[edge_feats\[i\]',
                    r'filtered_edge_feats = [edge_feats[i] for i in valid_indices]',
                    content)
    
    # 完成 add_edges_in_batches 方法
    if "# 計算批次數量\n        total_edges" in content:
        add_edges_completion = """
        # 計算批次數量
        total_edges = len(filtered_src)
        num_batches = (total_edges + batch_size - 1) // batch_size

        logger.info(f"批量添加 {total_edges} 條有效邊，分為 {num_batches} 批次處理")

        # 分批次處理邊
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_edges)

            # 獲取當前批次
            batch_src = filtered_src[start_idx:end_idx]
            batch_dst = filtered_dst[start_idx:end_idx]
            batch_timestamps = filtered_timestamps[start_idx:end_idx]

            if filtered_edge_feats is not None:
                batch_edge_feats = filtered_edge_feats[start_idx:end_idx]
            else:
                batch_edge_feats = None

            # 添加當前批次的邊
            self.add_edges(batch_src, batch_dst, batch_timestamps, batch_edge_feats)

            # 每隔幾個批次清理一次記憶體
            if (batch_idx + 1) % 5 == 0:
                clean_memory()
                logger.info(f"已處理 {min(end_idx, total_edges)}/{total_edges} 條邊")

        logger.info(f"批量添加邊完成")"""
        
        content = content.replace("# 計算批次數量\n        total_edges", add_edges_completion)
    
    # 添加缺失的時間子圖方法
    if content and "def add_edges_in_batches" in content and "def update_temporal_graph" not in content:
        update_temporal_graph_method = """
    @memory_usage_decorator
    def update_temporal_graph(self, current_time=None):
        \"\"\"
        更新時間圖 (優化版) - 使用高效稀疏表示和採樣技術

        參數:
            current_time (float, optional): 當前時間戳記，預設使用最新時間
        \"\"\"
        if current_time is not None:
            self.current_time = current_time

        # 計算時間窗口的起始時間
        start_time = self.current_time - self.temporal_window

        # 使用高效的子圖構建技術
        if self.cache_neighbor_sampling and f"{start_time}_{self.current_time}" in self.neighbor_cache:
            # 使用緩存的結果
            self.cache_hits += 1
            logger.info(f"使用緩存的時間子圖 ({start_time} 到 {self.current_time}) 緩存命中率: {self.cache_hits/(self.cache_hits+self.cache_misses):.2f}")
            self.temporal_g = self.neighbor_cache[f"{start_time}_{self.current_time}"]
            return self.temporal_g
        else:
            self.cache_misses += 1

        # 過濾時間窗口內的邊 - 使用向量化操作加速
        temporal_src = []
        temporal_dst = []
        temporal_edge_feats = []
        temporal_edge_times = []

        # 獲取節點列表
        node_list = sorted(list(self.existing_nodes))
        node_set = set(node_list)  # 快速查找用的集合

        # 高效採樣 - 如果啟用子圖採樣且節點數量超過限制
        if self.use_subgraph_sampling and len(node_list) > self.max_nodes_per_subgraph:
            # 使用度數加權採樣而非簡單隨機採樣 - 保留高連接度節點
            if self.adaptive_pruning:
                # 計算節點權重 - 根據連接度和重要性
                node_weights = {}
                for nid in node_list:
                    in_degree = len(self.dst_to_src.get(nid, set()))
                    out_degree = len(self.src_to_dst.get(nid, set()))
                    total_degree = in_degree + out_degree
                    importance = self.node_importance.get(nid, 1.0)
                    node_weights[nid] = total_degree * importance

                # 根據權重進行採樣
                if sum(node_weights.values()) > 0:
                    sampled_nodes = random.choices(
                        node_list,
                        weights=[node_weights.get(nid, 1.0) for nid in node_list],
                        k=min(self.max_nodes_per_subgraph, len(node_list))
                    )
                    # 去除可能的重複
                    node_list = list(set(sampled_nodes))
                else:
                    node_list = random.sample(node_list, self.max_nodes_per_subgraph)
            else:
                # 簡單隨機採樣
                node_list = random.sample(node_list, self.max_nodes_per_subgraph)

            node_set = set(node_list)  # 更新節點集合
            logger.info(f"節點數量 {len(self.existing_nodes)} 超過限制 {self.max_nodes_per_subgraph}, 採樣後剩餘 {len(node_list)} 個節點")

        # 高效映射建立 - 為了快速索引查找
        node_idx_map = {nid: i for i, nid in enumerate(node_list)}

        # 高效邊過濾 - 使用字典結構和集合操作加速
        if self.edge_index_format:
            # 使用邊索引格式時的高效過濾
            src_array, dst_array = self.edge_index
            time_filtered_edges = []

            # 批量檢查時間戳
            for edge_idx, (src, dst) in enumerate(zip(src_array, dst_array)):
                edge_key = (src, dst)
                if edge_key in self.edge_timestamps:
                    timestamp = self.edge_timestamps[edge_key]
                    if timestamp >= start_time and src in node_set and dst in node_set:
                        time_filtered_edges.append((edge_idx, src, dst, timestamp))

                        # 限制邊數量
                        if self.use_subgraph_sampling and len(time_filtered_edges) >= self.max_edges_per_subgraph:
                            break

            # 從過濾後的邊中提取信息
            if time_filtered_edges:
                edge_indices, filtered_src, filtered_dst, filtered_times = zip(*time_filtered_edges)

                # 獲取新的連續索引
                temporal_src = [node_idx_map[src] for src in filtered_src]
                temporal_dst = [node_idx_map[dst] for dst in filtered_dst]
                temporal_edge_times = list(filtered_times)

                # 獲取邊特徵
                if self.edge_feat_dim is not None:
                    temporal_edge_feats = []
                    for s, d in zip(filtered_src, filtered_dst):
                        edge_key = (s, d)
                        if edge_key in self.edge_features:
                            temporal_edge_feats.append(self.edge_features[edge_key])
                        else:
                            # 對於沒有特徵的邊, 使用零向量
                            temporal_edge_feats.append([0.0] * self.edge_feat_dim)
        else:
            # 標準格式的高效過濾
            edge_count = 0
            for (src, dst), timestamp in self.edge_timestamps.items():
                if timestamp >= start_time and src in node_set and dst in node_set:
                    # 獲取節點索引
                    src_idx = node_idx_map[src]
                    dst_idx = node_idx_map[dst]

                    temporal_src.append(src_idx)
                    temporal_dst.append(dst_idx)
                    temporal_edge_times.append(timestamp)

                    # 邊特徵
                    if (src, dst) in self.edge_features:
                        temporal_edge_feats.append(self.edge_features[(src, dst)])
                    else:
                        # 對於沒有特徵的邊, 使用零向量
                        temporal_edge_feats.append([0.0] * self.edge_feat_dim if self.edge_feat_dim else [])

                    edge_count += 1

                    # 如果啟用子圖採樣且邊數量超過限制, 提前結束
                    if self.use_subgraph_sampling and edge_count >= self.max_edges_per_subgraph:
                        logger.info(f"邊數量達到限制 {self.max_edges_per_subgraph}, 提前結束邊過濾")
                        break

        # 建立時間子圖 - 使用最適合的表示方法
        if len(temporal_src) > 0:
            if self.use_sparse_representation:
                # 使用DGL的稀疏表示API
                if self.use_csr_format:
                    # 創建CSR格式的稀疏圖
                    indices = torch.tensor([temporal_src, temporal_dst], dtype=torch.int64)
                    # 創建稀疏鄰接矩陣
                    adj = torch.sparse_coo_tensor(
                        indices=indices,
                        values=torch.ones(len(temporal_src)),
                        size=(len(node_list), len(node_list))
                    )
                    # 轉換為CSR格式
                    csr = adj.to_sparse_csr()

                    # 使用DGL的from_csr構造函數
                    if hasattr(dgl, 'from_csr'):
                        indptr = csr.crow_indices()
                        indices = csr.col_indices()
                        data = csr.values()
                        self.temporal_g = dgl.from_csr(indptr, indices, data, len(node_list))
                    else:
                        # 備用方法: 仍使用標準構造
                        self.temporal_g = dgl.graph((temporal_src, temporal_dst),
                                                num_nodes=len(node_list),
                                                idtype=torch.int64,
                                                device='cpu')
                else:
                    # 使用稀疏COO表示
                    self.temporal_g = dgl.graph((temporal_src, temporal_dst),
                                            num_nodes=len(node_list),
                                            idtype=torch.int64,
                                            device='cpu')
            else:
                # 使用密集表示
                self.temporal_g = dgl.graph((temporal_src, temporal_dst),
                                        num_nodes=len(node_list),
                                        idtype=torch.int64,
                                        device='cpu')
        else:
            # 如果沒有邊, 創建空圖
            self.temporal_g = dgl.graph(([], []),
                                     num_nodes=len(node_list),
                                     idtype=torch.int64,
                                     device='cpu')

        # 獲取節點特徵 - 向量化操作
        node_features = []
        for nid in node_list:
            if nid in self.node_features:
                node_features.append(self.node_features[nid])
            else:
                # 對於沒有特徵的節點, 使用零向量
                node_features.append(torch.zeros(self.node_feat_dim))

        # 設置節點特徵
        if node_features:
            self.temporal_g.ndata['h'] = torch.stack(node_features)

        # 設置邊特徵和時間戳記
        if temporal_edge_feats:
            self.temporal_g.edata['h'] = torch.tensor(temporal_edge_feats)
        if temporal_edge_times:
            self.temporal_g.edata['time'] = torch.tensor(temporal_edge_times)

        # 設置節點標籤
        node_labels = []
        for nid in node_list:
            if nid in self.node_labels:
                node_labels.append(self.node_labels[nid])
            else:
                node_labels.append(-1)

        if node_labels:
            self.temporal_g.ndata['label'] = torch.tensor(node_labels)

        # 使用DGL的transform API優化圖
        if self.use_dgl_transform:
            # 添加自環 - 確保圖有完整連接性
            self.temporal_g = dgl.add_self_loop(self.temporal_g)

            # 如果支持, 轉換為最適合的圖格式
            try:
                if self.use_csr_format and hasattr(dgl.transforms, 'to_simple_graph'):
                    # 將圖轉換為簡單圖 - 合併多重邊
                    self.temporal_g = dgl.transforms.to_simple_graph(self.temporal_g)
            except Exception as e:
                logger.warning(f"圖轉換失敗: {str(e)}")

        # 將圖移至指定裝置
        if self.device != 'cpu':
            self.temporal_g = self.temporal_g.to(self.device)

        # 使用緩存優化
        if self.cache_neighbor_sampling:
            # 緩存當前時間窗口的子圖
            self.neighbor_cache[f"{start_time}_{self.current_time}"] = self.temporal_g

            # 限制緩存大小, 清理過舊的緩存
            if len(self.neighbor_cache) > 10:
                # 移除最舊的緩存
                oldest_key = sorted(self.neighbor_cache.keys())[0]
                del self.neighbor_cache[oldest_key]

        logger.info(f"更新時間圖: {len(temporal_src)} 條邊在時間窗口 {start_time} 到 {self.current_time}")

        return self.temporal_g

    @memory_usage_decorator
    def add_batch(self, node_ids, features, timestamps, src_nodes=None, dst_nodes=None,
                 edge_timestamps=None, edge_feats=None, labels=None):
        \"\"\"
        批次添加節點和邊 - 高效集成操作

        參數:
            node_ids (list): 節點ID列表
            features (np.ndarray): 節點特徵矩陣
            timestamps (list): 節點時間戳記列表
            src_nodes (list, optional): 源節點ID列表
            dst_nodes (list, optional): 目標節點ID列表
            edge_timestamps (list, optional): 邊時間戳記列表
            edge_feats (list, optional): 邊特徵列表
            labels (list, optional): 節點標籤列表
        \"\"\"
        # 添加節點 - 使用向量化操作
        self.add_nodes(node_ids, features, timestamps, labels)

        # 添加邊 (如果提供) - 使用批次處理提高效率
        if src_nodes is not None and dst_nodes is not None and edge_timestamps is not None:
            self.add_edges_in_batches(src_nodes, dst_nodes, edge_timestamps, edge_feats)

        # 更新時間圖 - 使用優化的子圖構建
        self.update_temporal_graph()

        return self.temporal_g

    @memory_usage_decorator
    def simulate_stream(self, node_ids, features, timestamps, labels=None):
        \"\"\"
        模擬流式資料, 自動建立時間性邊 - 優化版本

        參數:
            node_ids (list): 節點ID列表
            features (np.ndarray): 節點特徵矩陣
            timestamps (list): 節點時間戳記列表
            labels (list, optional): 節點標籤列表
        \"\"\"
        # 添加節點
        self.add_nodes(node_ids, features, timestamps, labels)

        # 自動建立時間性邊 - 使用高效數據結構
        src_nodes = []
        dst_nodes = []
        edge_timestamps = []
        edge_feats = []

        # 時間閾值 (秒), 用於決定兩個封包是否"時間接近"
        time_threshold = 1.0

        # 先按時間排序節點 - 使用NumPy/Pandas高效排序
        sorted_indices = np.argsort(timestamps)
        sorted_nodes = [(node_ids[i], timestamps[i]) for i in sorted_indices]

        # 使用滑動窗口優化邊生成 - 避免N^2複雜度
        window_size = min(100, len(sorted_nodes))  # 動態調整窗口大小

        # 對於每個新節點, 連接到時間窗口內的相關節點
        for i, (nid, timestamp) in enumerate(sorted_nodes):
            # 初始化窗口範圍
            window_start = max(0, i - window_size)
            window_end = i  # 不包括當前節點

            # 遍歷窗口中的節點
            for j in range(window_start, window_end):
                prev_nid, prev_timestamp = sorted_nodes[j]

                # 檢查是否時間接近
                if abs(timestamp - prev_timestamp) <= time_threshold:
                    # 隨機決定連接方向 - 雙向連接增加50%
                    if random.random() < 0.7:  # 70%機率建立邊
                        # 創建 prev -> current 邊
                        src_nodes.append(prev_nid)
                        dst_nodes.append(nid)
                        edge_timestamps.append(timestamp)

                        # 生成簡單的邊特徵 (時間差和隨機特性)
                        time_diff = timestamp - prev_timestamp
                        random_feat = [random.random(), time_diff, random.random()]
                        edge_feats.append(random_feat)

                        # 隨機決定是否建立反向邊
                        if random.random() < 0.5:  # 50%機率建立反向邊
                            src_nodes.append(nid)
                            dst_nodes.append(prev_nid)
                            edge_timestamps.append(timestamp)

                            # 反向邊使用相似但不同的特徵
                            rev_random_feat = [random.random(), -time_diff, random.random()]
                            edge_feats.append(rev_random_feat)

            # 優化: 動態調整窗口大小以控制邊密度
            # 如果節點多, 我們可以使用小窗口; 如果節點少, 使用大窗口
            if i % 50 == 0:
                edge_density = len(src_nodes) / max(1, len(self.existing_nodes))
                if edge_density > 10:  # 邊密度過高
                    window_size = max(10, window_size // 2)
                elif edge_density < 2:  # 邊密度過低
                    window_size = min(200, window_size * 2)

        # 批量添加所有邊
        if src_nodes:
            self.add_edges_in_batches(src_nodes, dst_nodes, edge_timestamps, edge_feats)

        # 更新時間圖
        self.update_temporal_graph()

        logger.info(f"模擬流式數據: 添加 {len(src_nodes)} 條時間性邊")

        return self.temporal_g

    def to_sparse_tensor(self):
        \"\"\"
        將圖轉換為稀疏張量表示 - 用於記憶體高效的表示和計算

        返回:
            tuple: (indices, values, shape) - PyTorch稀疏張量的組件
        \"\"\"
        if not self.g:
            logger.warning("圖為空, 無法轉換為稀疏張量")
            return None

        # 獲取圖的邊
        if self.edge_index_format:
            src, dst = self.edge_index
        else:
            src, dst = self.g.edges()

        # 創建稀疏COO格式表示
        indices = torch.stack([torch.tensor(src), torch.tensor(dst)])
        values = torch.ones(len(src))
        shape = (self.g.num_nodes(), self.g.num_nodes())

        # 使用SciPy稀疏矩陣
        if sp is not None and self.use_sparse_representation:
            try:
                # 轉換為SciPy CSR矩陣
                scipy_sparse = sp.csr_matrix(
                    (values.numpy(), (indices[0].numpy(), indices[1].numpy())),
                    shape=shape
                )
                logger.info(f"成功轉換為SciPy CSR稀疏矩陣, 大小: {shape}, 非零元素: {scipy_sparse.nnz}")
                return scipy_sparse
            except Exception as e:
                logger.warning(f"轉換為SciPy稀疏矩陣失敗: {str(e)}")

        # 使用PyTorch稀疏張量
        sparse_tensor = torch.sparse_coo_tensor(indices, values, shape)
        logger.info(f"成功轉換為PyTorch稀疏張量, 大小: {shape}, 非零元素: {len(values)}")

        return sparse_tensor"""
        
        # 在文件末尾添加缺失的方法
        content += update_temporal_graph_method
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("檔案修復完成")
    return True

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = '../src/data/optimized_graph_builder.py'
    
    fix_graph_builder_file(file_path)
