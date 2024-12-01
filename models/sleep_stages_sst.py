import torch
import torch.nn as nn
from .multi_scale_patcher import MultiScalePatcher
from .global_expert import GlobalExpert
from .local_expert import LocalWindowTransformer
from .router import LongShortTermRouter

class SleepStageSST(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # 多尺度分解
        self.patcher = MultiScalePatcher(
            input_len=config.epoch_len,
            scales=config.scales
        )
        
        # 特征提取
        self.feature_extractor = nn.Conv1d(
            in_channels=config.input_channels,
            out_channels=config.hidden_size,
            kernel_size=config.kernel_size,
            padding='same'
        )
        
        # 专家模型
        self.global_expert = GlobalExpert(
            d_model=config.hidden_size,
            state_size=config.state_size
        )
        
        self.local_expert = LocalWindowTransformer(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            window_size=config.window_size
        )
        
        # 路由器
        self.router = LongShortTermRouter(
            hidden_size=config.hidden_size
        )
        
        # 分类头
        self.classifier = nn.Linear(
            config.hidden_size,
            config.num_classes
        )
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [batch_size, channels, time_steps]
        Returns:
            torch.Tensor: [batch_size, num_classes]
        """
        # 多尺度分解
        patches = self.patcher(x)
        
        # 特征提取
        features = []
        for patch in patches:
            feat = self.feature_extractor(patch)
            features.append(feat)
        
        # 合并特征
        features = torch.cat(features, dim=1)
        
        # 专家处理
        global_features = self.global_expert(features)
        local_features = self.local_expert(features)
        
        # 路由融合
        fused = self.router(global_features, local_features)
        
        # 分类
        logits = self.classifier(fused.mean(dim=1))  # 全局平均池化
        
        return logits