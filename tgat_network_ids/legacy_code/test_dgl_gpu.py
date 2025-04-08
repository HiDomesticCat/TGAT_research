import dgl
import torch
import sys

print(f"Python 版本: {sys.version}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"DGL 版本: {dgl.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
print(f"GPU 設備: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# 測試 DGL 在 GPU 上運行
try:
    g = dgl.graph(([0, 1], [1, 2]))
    print("成功創建 CPU 圖")
    
    if torch.cuda.is_available():
        g = g.to('cuda')
        print("成功將圖移至 CUDA 設備")
        
        # 添加一些特徵測試
        g.ndata['h'] = torch.ones(3, 5).to('cuda')
        print("成功添加 CUDA 上的節點特徵")
        
        # 驗證特徵位置
        print(f"節點特徵設備: {g.ndata['h'].device}")
    else:
        print("CUDA 不可用")
        
except Exception as e:
    print(f"錯誤: {e}")