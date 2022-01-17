import argparse
import os
import yaml

import torch
import torch.nn as nn

from src.data import Dataset
from src.models import Generator
from src.logger import get_run
from src.training import stage1, stage2


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
    generator = Generator()
    generator = generator.to(device)
    return generator


def main(debug: bool = False) -> None:
    config = get_config()
    device = get_device(config)
    dataloader = get_dataloader(config)
    generator = get_generator(device)
    run = get_run(debug=debug)
    stage1(generator, dataloader, device, config, debug, run)
    stage2()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help="use to run script in debugging mode")
    args = parser.parse_args()
    main(debug=args.debug)