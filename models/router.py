import torch
import torch.nn as nn

class LongShortTermRouter(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
    def forward(self, global_features, local_features):
        """
        Args:
            global_features (torch.Tensor): [batch_size, seq_len, hidden_size]
            local_features (torch.Tensor): [batch_size, seq_len, hidden_size]
        Returns:
            torch.Tensor: [batch_size, seq_len, hidden_size]
        """
        # 计算重要性权重
        combined = torch.cat([global_features, local_features], dim=-1)
        importance = self.gate(combined)
        
        # 动态融合
        output = importance * global_features + (1 - importance) * local_features
        return output