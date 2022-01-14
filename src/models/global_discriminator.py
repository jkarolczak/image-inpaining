import os
import yaml

import torch
import torch.nn as nn


class Branch(nn.Module):
    def __init__(self):
            super().__init__()
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(x)
        x = x.max(axis=-1).values.max(axis=-1).values
        x = x.unsqueeze(1)
        return x


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
            
        self.branches = [Branch() for _ in range(3)]
        
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        
        self.line_block = nn.Sequential(
            nn.Linear(in_features=256, out_features=64),
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
                
        q = self.branches[0](x)
        k = self.branches[1](x)
        v = self.branches[2](x)
                        
        x, _ = self.attention(q, k, v)
        x = x.squeeze(-1)
        
        x = self.line_block(x)
        x = x.squeeze(-1)
        
        return x