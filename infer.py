
import argparse
import os
import yaml
from typing import Dict

import cv2
import torch

from src.data import Dataset
from src.models import Generator


def get_config(
    cfg_dir: str = 'cfg', 
    cfg_yaml: str = 'training.yaml'
) -> dict:
    yaml_path = os.path.join(cfg_dir, cfg_yaml)
    with open(yaml_path, 'r') as fd:
        config = yaml.safe_load(fd)
    return config


def get_dataloader() -> torch.utils.data.DataLoader:
    shuffle = True
    batch_size = 1
    num_workers = 1
    dataset = Dataset()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader


def get_device() -> torch.device:
    return torch.device('cpu')


def get_generator(device: torch.device) -> Generator:
    return Generator().to(device)


def prepare_directory() -> None:
    os.makedirs('inference_results', exist_ok=True)
    for file in os.listdir('inference_results'):
        path = os.path.join('inference_results', file)
        os.remove(path)


def get_img_path(idx: int) -> str:
    return os.path.join('inference_results', str(idx) + '.png')


def main(state_dict_path: str, images: int) -> None:
    prepare_directory()
    device = get_device()
    dataloader = get_dataloader()
    netG = get_generator(device)
    
    netG.train()
    with torch.no_grad():
        for idx, (img_input, img_target, coords) in enumerate(dataloader):
            if idx == images: break
            img_generated = netG(img_input)
            img_numpy = img_generated[0].numpy()
            path = get_img_path(idx)
            cv2.imwrite(path, img_numpy)
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--statedict', help="path to the saved state dict")
    parser.add_argument('--images', type=int, default=10, help="number of images to process")
    args = parser.parse_args()
    main(args.statedict, args.images)