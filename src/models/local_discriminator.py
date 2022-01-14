import os
import yaml

import torch
import torch.nn as nn


class LocalDiscriminator(nn.Module):
    def __init__(
        self,
        cfg_dir: str = 'cfg', 
        discriminator_yaml: str = 'net_discriminators.yaml'
    ):
        super().__init__()
        
        yaml_path = os.path.join(cfg_dir, discriminator_yaml)
        with open(yaml_path, 'r') as fd:
            params = yaml.safe_load(fd)['local']
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        self.linear_block = nn.Sequential(
            nn.Linear(192, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        if len(x.shape) == 3: x = x.unsqueeze(0)
        x = x.reshape(x.shape[-4], x.shape[-1], x.shape[-3], x.shape[-2])
        x = self.conv_block(x)
        
        x_min = torch.flatten(x, start_dim=2).min(axis=-1).values
        x_mean = torch.flatten(x, start_dim=2).mean(axis=-1)
        x_max = torch.flatten(x, start_dim=2).max(axis=-1).values

        x = torch.hstack([x_min, x_mean, x_max])
        x = self.linear_block(x)
                
        return x