import torch
import torch.nn as nn
import torch.nn.functional as F

class CRPBlock(nn.Module):
    """
    Chained Residual Pooling (CRP) Block.
    Core component of Light-Weight RefineNet used to capture multi-scale 
    contextual information efficiently with minimal computational overhead.
    """
    def __init__(self, in_channels, out_channels, n_stages=4):
        super(CRPBlock, self).__init__()
        self.n_stages = n_stages
        
        # Multi-stage max pooling and 1x1 convolutions
        self.maxpools = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=5, stride=1, padding=2) for _ in range(n_stages)]
        )
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) for _ in range(n_stages)]
        )
        
    def forward(self, x):
        out = x
        for i in range(self.n_stages):
            x = self.maxpools[i](x)
            x = self.convs[i](x)
            out = out + x  # Residual connection
        return out


class LightWeightRefineNet(nn.Module):
    """
    Light-Weight RefineNet Decoder for Semantic Segmentation.
    """
    def __init__(self, num_classes=7):
        super(LightWeightRefineNet, self).__init__()
        
        # 1. Dimensionality reduction layer: Using LazyConv2d to automatically 
        # adapt to whatever input channels the encoder outputs (e.g., 320, 1280, etc.)
        self.conv_1x1 = nn.LazyConv2d(256, kernel_size=1, bias=False)
        
        # 2. Context aggregation layer via Chained Residual Pooling
        self.crp = CRPBlock(in_channels=256, out_channels=256, n_stages=4)
        
        # 3. Final classification head
        self.clf = nn.Conv2d(256, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        # Compress channels (LazyConv2d initializes weights on the first forward pass)
        x = self.conv_1x1(x)
        x = F.relu(x, inplace=True)
        
        # Extract global context
        x = self.crp(x)
        
        # Output raw logits [B, num_classes, H, W]
        out = self.clf(x)
        return out