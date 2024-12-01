import torch
import torch.nn as nn

class MultiScalePatcher(nn.Module):
    def __init__(self, input_len, scales=[1, 2, 4, 8]):
        super().__init__()
        self.input_len = input_len
        self.scales = scales
    
    def forward(self, signal):
        """
        Args:
            signal (torch.Tensor): [batch_size, channels, time_steps]
        Returns:
            list of torch.Tensor: 不同尺度的patches
        """
        patches = []
        for scale in self.scales:
            patch_size = self.input_len // scale
            # 使用unfold进行分片
            patch = signal.unfold(dimension=-1, 
                                size=patch_size, 
                                step=patch_size)
            patches.append(patch)
        return patches