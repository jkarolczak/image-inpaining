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
            
    
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        if len(x.shape) == 3: x = x.unsqueeze(0)
        x = x.reshape(x.shape[-4], x.shape[-1], x.shape[-3], x.shape[-2])
        pass
        return x