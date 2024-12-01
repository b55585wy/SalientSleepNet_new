import torch
import torch.nn as nn

class LocalWindowTransformer(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 window_size: int,
                 dropout: float = 0.1):
        super().__init__()
        
        self.window_size = window_size
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [batch_size, seq_len, hidden_size]
        Returns:
            torch.Tensor: [batch_size, seq_len, hidden_size]
        """
        # 分割成固定大小的窗口
        B, L, H = x.shape
        P = self.window_size
        
        # 填充到窗口大小的整数倍
        pad_len = (P - L % P) % P
        if pad_len > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
            
        # 重塑为窗口形式
        windows = x.view(B, -1, P, H)
        
        # 在每个窗口内做自注意力
        windows = windows.reshape(-1, P, H)
        attended = self.attention(windows, windows, windows)[0]
        
        # 重塑回原始形状
        attended = attended.view(B, -1, P, H)
        attended = attended.reshape(B, -1, H)
        
        # 移除填充
        if pad_len > 0:
            attended = attended[:, :L]
            
        return attended