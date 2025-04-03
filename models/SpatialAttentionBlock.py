import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise average pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Shape: (B, 1, H, W)
        
        # Channel-wise max pooling
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Shape: (B, 1, H, W)
        
        # Concatenate along the channel dimension
        concat_out = torch.cat([avg_out, max_out], dim=1)  # Shape: (B, 2, H, W)
        
        # Apply a convolution layer followed by a sigmoid activation
        attention_map = self.conv(concat_out)  # Shape: (B, 1, H, W)
        attention_map = self.sigmoid(attention_map)  # Shape: (B, 1, H, W)
        
        # Multiply attention map with the original input feature map
        return x * attention_map  # Shape: (B, C, H, W)