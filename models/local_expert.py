import torch
import torch.nn as nn
from mamba_ssm import Mamba

class GlobalExpert(nn.Module):
    def __init__(self, 
                 d_model: int,
                 state_size: int,
                 d_conv: int = 4,
                 expand: int = 2):
        super().__init__()
        
        self.mamba = Mamba(
            d_model=d_model,
            d_state=state_size,
            d_conv=d_conv,
            expand=expand
        )
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [batch_size, seq_len, d_model]
        Returns:
            torch.Tensor: [batch_size, seq_len, d_model]
        """
        return self.mamba(x)