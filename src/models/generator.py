import os
import yaml

import torch
import torch.nn as nn   
    

class Generator(nn.Module):
    def __init__(
        self,
        cfg_dir: str = 'cfg', 
        generator_yaml: str = 'net_generator.yaml'
    ):
        super().__init__()
        
        yaml_path = os.path.join(cfg_dir, generator_yaml)
        with open(yaml_path, 'r') as fd:
            params = yaml.safe_load(fd)
                
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.maxunpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
                
        self.encoder_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
            
        self.encoder_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.encoder_conv_3 = nn.Sequential(        
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()   
        )
        
        self.mid = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        )

        self.decoder_conv_3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
        )
        
        self.decoder_conv_2 = nn.Sequential(
            nn.ReLU(),             
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        )
        
        self.decoder_conv_1 =  nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),
        )
    
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        x = x.reshape(x.shape[-4], x.shape[-1], x.shape[-3], x.shape[-2])
        
        x = self.encoder_conv_1(x)
        shape_1 = x.shape
        x, indices_1 = self.maxpool(x)
        x = self.encoder_conv_2(x)
        shape_2 = x.shape
        x, indices_2 = self.maxpool(x)
        x = self.encoder_conv_3(x)
        shape_3 = x.shape
        x, indices_3 = self.maxpool(x)
        
        x = self.mid(x)
        
        x = self.maxunpool(x, indices_3, output_size=shape_3)
        x = self.decoder_conv_3(x)
        x = self.maxunpool(x, indices_2, output_size=shape_2)
        x = self.decoder_conv_2(x)
        x = self.maxunpool(x, indices_1, output_size=shape_1)
        x = self.decoder_conv_1(x)
        
        x = x.reshape(x.shape[-4], x.shape[-2], x.shape[-1], x.shape[-3])
        return x
