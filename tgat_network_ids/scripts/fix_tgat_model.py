#!/usr/bin/env python
# coding: utf-8 -*-

"""
修復 optimized_tgat_model.py 檔案中的語法錯誤，主要是補上缺少的逗號和完成未完成的方法
"""

import re
import sys

def fix_tgat_model_file(file_path):
    print(f"修復檔案: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修復基礎語法錯誤
    content = content.replace("logging.basicConfig(level=logging.INFO", 
                            "logging.basicConfig(level=logging.INFO,")
    
    content = content.replace("super(MemoryEfficientTimeEncoding self).__init__()",
                            "super(MemoryEfficientTimeEncoding, self).__init__()")
    
    content = content.replace("def __init__(self dimension):",
                            "def __init__(self, dimension):")
    
    content = content.replace("def _quantize_time(self t):",
                            "def _quantize_time(self, t):")
    
    content = content.replace("def _compute_encoding(self t_tensor):",
                            "def _compute_encoding(self, t_tensor):")
    
    content = content.replace("hit_rate = self.cache_hits / max(1 self.total_queries) * 100",
                            "hit_rate = self.cache_hits / max(1, self.total_queries) * 100")
    
    content = content.replace("return torch.empty((0 self.dimension) device=t.device)",
                            "return torch.empty((0, self.dimension), device=t.device)")
    
    content = content.replace("super(OptimizedTemporalGATLayer self).__init__()",
                            "super(OptimizedTemporalGATLayer, self).__init__()")
    
    content = content.replace("super(EfficientGATConv self).__init__()",
                            "super(EfficientGATConv, self).__init__()")
    
    content = content.replace("if isinstance(layer nn.Linear)",
                            "if isinstance(layer, nn.Linear)")
                            
    content = content.replace("super(OptimizedTGAT self).__init__()",
                            "super(OptimizedTGATModel, self).__init__()")
    
    content = content.replace("return torch.zeros((0 self.num_classes) device=g.device)",
                            "return torch.zeros((0, self.num_classes), device=g.device)")
    
    content = content.replace("return torch.zeros((0 self.out_dim) device=g.device)",
                            "return torch.zeros((0, self.out_dim), device=g.device)")
    
    content = content.replace("for name module in self.named_modules()",
                            "for name, module in self.named_modules()")
                            
    content = content.replace("if isinstance(module nn.Linear) and hasattr(module '_dense_weight')",
                            "if isinstance(module, nn.Linear) and hasattr(module, '_dense_weight')")
                            
    content = content.replace("if hasattr(module 'weight')",
                            "if hasattr(module, 'weight')")
                            
    content = content.replace("weight = getattr(module '_dense_weight')",
                            "weight = getattr(module, '_dense_weight')")
                            
    content = content.replace("delattr(module '_dense_weight')",
                            "delattr(module, '_dense_weight')")
                            
    content = content.replace("time_expanded = time_feat.expand(-1 min(src_h.shape[1] self.time_dim))",
                            "time_expanded = time_feat.expand(-1, min(src_h.shape[1], self.time_dim))")
    
    # 修復變數，函數調用中缺少的逗號
    content = re.sub(r'h = layer\(g h time_tensor\)', r'h = layer(g, h, time_tensor)', content)
    content = re.sub(r'indices values weight.size\(\)', r'indices, values, weight.size()', content)
    content = re.sub(r'nn.init.xavier_normal_\(([^,]+) gain=gain\)', r'nn.init.xavier_normal_(\1, gain=gain)', content)
    content = re.sub(r'nn.init.orthogonal_\(([^,]+) gain=gain\)', r'nn.init.orthogonal_(\1, gain=gain)', content)
    
    # 完成 to_sparse_tensor 方法
    if "# 僅在不訓練時使用此功能" in content:
        to_sparse_tensor_completion = """
        # 僅在不訓練時使用此功能
        if self.training:
            logger.warning("稀疏張量轉換僅應在評估模式下使用，不適用於訓練")
            return

        # 遍歷所有層，尋找可轉換為稀疏的線性層權重
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # 只轉換較大的層以節省記憶體
                if module.weight.numel() > 10000:
                    # 計算稀疏度
                    weight = module.weight.data
                    sparsity = (weight.abs() < 1e-3).float().mean().item()

                    # 只有當稀疏度較高時才轉換
                    if sparsity > 0.5:
                        # 創建掩碼 - 保留絕對值大於閾值的元素
                        mask = weight.abs() >= 1e-3

                        # 轉換為COO稀疏格式
                        indices = mask.nonzero().t()
                        values = weight[mask]
                        sparse_weight = torch.sparse_coo_tensor(
                            indices, values, weight.size()
                        )

                        # 臨時保存原始權重，這樣在需要時可以恢復
                        setattr(module, '_dense_weight', weight.clone())

                        # 替換為稀疏權重
                        del module.weight
                        module.register_buffer('weight', sparse_weight)

                        logger.info(f"已將層 {name} 轉換為稀疏表示，稀疏度: {sparsity:.2f} 節省記憶體: {weight.nelement() * 4 * (1-sparsity) / 1024 / 1024:.2f} MB")

        logger.info("已完成模型的稀疏張量轉換")

    def to_dense_tensor(self):
        \"\"\"將模型從稀疏表示恢復為密集表示\"\"\"
        # 遍歷所有模組，恢復任何稀疏層
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, '_dense_weight'):
                # 恢復原始權重
                weight = getattr(module, '_dense_weight')

                # 刪除稀疏權重
                if hasattr(module, 'weight'):
                    delattr(module, 'weight')

                # 設置回密集權重
                module.weight = nn.Parameter(weight)

                # 刪除備份
                delattr(module, '_dense_weight')

                logger.info(f"已將層 {name} 恢復為密集表示")

        logger.info("已完成模型的密集張量恢復")"""
        
        content = content.replace("# 僅在不訓練時使用此功能", to_sparse_tensor_completion)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("檔案修復完成")
    return True

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = '../src/models/optimized_tgat_model.py'
    
    fix_tgat_model_file(file_path)
