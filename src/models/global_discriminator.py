import os
import yaml

import torch
import torch.nn as nn


class GlobalDiscriminator(nn.Module):
    def __init__(
        self,
        cfg_dir: str = 'cfg', 
        discriminator_yaml: str = 'net_discriminators.yaml'
    ):
        super().__init__()
        
        yaml_path = os.path.join(cfg_dir, discriminator_yaml)
        with open(yaml_path, 'r') as fd:
            params = yaml.safe_load(fd)['global']
            
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(start_dim=2),
        )
        
        self.attention = nn.MultiheadAttention(embed_dim=8, num_heads=4, batch_first=True)
        
        self.line_block = nn.Sequential(
            nn.Linear(in_features=22 * 27 * 8, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        if len(x.shape) == 3: x = x.unsqueeze(0)
        x = x.reshape(x.shape[-4], x.shape[-1], x.shape[-3], x.shape[-2])
        
        x = self.conv_block(x)        
        x = x.reshape(x.shape[-3], x.shape[-1], x.shape[-2])
        x, _ = self.attention(x, x, x, need_weights=False)
        x = x.reshape(x.shape[-3], x.shape[-1] * x.shape[-2])
        x = self.line_block(x)

        return x