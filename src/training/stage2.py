import neptune.new as neptune
import torch
import torch.nn as nn

import src.models as models
from src.models import Generator, GlobalDiscriminator, LocalDiscriminator
from src.logger import log
from src.training.common import *


def get_criterion(config: dict) -> torch.nn.Module:
    return criterion_opts[config['stage2']['loss'].lower()]()


def get_epochs(config: dict) -> int:
    return config['stage2']['epochs']


def get_optimizers(
    netG: nn.Module, 
    netGD: nn.Module, 
    netLD: nn.Module, 
    config: dict
) -> torch.nn.Module:
    optimizers = []
    for net, model in zip(['netG', 'netGD', 'netLD'], [netG, netGD, netLD]):
        optim = optimizer_opts[config['stage2'][net]['optimizer']]
        optim = optim(
            model.parameters(),
            lr=config['stage2'][net]['lr'],
            weight_decay=config['stage2'][net]['weight_decay']
        )
        optimizers.append(optim)
    return optimizers


def main(
    netG: Generator,
    netGD: GlobalDiscriminator,
    netLD: LocalDiscriminator,
    dataloader: torch.utils.data.Dataset,
    device: torch.device, 
    config: dict, 
    debug: bool, 
    run: neptune.Run
) -> None:
    criterion = get_criterion(config)
    optim_netG, optim_netGD, optim_netLD = get_optimizers(netG, netGD, netLD, config)
    epochs = get_epochs(config)
    