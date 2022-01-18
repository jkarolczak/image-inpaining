import argparse
import os
import sys
import yaml
from typing import Tuple

import torch
import torch.nn as nn

from src.data import Dataset
from src.models import Generator, GlobalDiscriminator, LocalDiscriminator
from src.logger import get_run
from src.training import stage1, stage2


def suppress_stdout() -> None:
    sys.stdout = open(os.devnull, 'w')


def get_config(
    cfg_dir: str = 'cfg', 
    cfg_yaml: str = 'training.yaml'
) -> dict:
    yaml_path = os.path.join(cfg_dir, cfg_yaml)
    with open(yaml_path, 'r') as fd:
        config = yaml.safe_load(fd)
    return config


def get_dataloader(
    config: dict,
) -> torch.utils.data.DataLoader:
    shuffle = config['dataloader']['shuffle']
    batch_size = config['dataloader']['batch_size']
    num_workers = config['dataloader']['num_workers']
    dataset = Dataset()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader


def get_device(config: dict) -> torch.device:
    if torch.cuda.is_available() and config['device'] == 'cuda':
        return torch.device('cuda')
    return torch.device('cpu')


def get_generator(device: torch.device) -> Generator:
    return Generator().to(device)


def get_discriminators(
    device: torch.device
) -> Tuple[GlobalDiscriminator, LocalDiscriminator]:
    netGD = GlobalDiscriminator().to(device)
    netLD = LocalDiscriminator().to(device)
    return (netGD, netLD)


def main(debug: bool = False) -> None:
    config = get_config()
    device = get_device(config)
    dataloader = get_dataloader(config)
    netG = get_generator(device)
    netGD, netLD = get_discriminators(device)
    run = get_run(debug=debug)
    
    stage1(netG, dataloader, device, config, debug, run)
    stage2(netG, netGD, netLD, dataloader, device, config, debug, run)
    
    run.stop()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help="use to run script in debugging mode")
    args = parser.parse_args()
    if not args.debug:
        suppress_stdout()
    main(debug=args.debug)