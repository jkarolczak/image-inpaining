import neptune.new as neptune
import torch
import torch.nn as nn

import src.models as models
from src.models import Generator
from src.logger import log
from src.training.common import cpu, cuda, mean, tensors_to_device


def get_optimizer(model: nn.Module, config: dict) -> torch.nn.Module:
    optimizer_opts = {
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'sgd': torch.optim.SGD,
    }
    config['stage1']['optimizer'].lower()
    optimizer = optimizer_opts[config['stage1']['optimizer'].lower()]
    optimizer = optimizer(
        model.parameters(),
        lr=config['stage1']['lr'],
        weight_decay=config['stage1']['weight_decay'],
    )
    return optimizer
    
    
def get_criterion(config: dict) -> torch.nn.Module:
    criterion_opts = {
        'mse': nn.MSELoss,
        'l1': nn.L1Loss,
        'mae': nn.L1Loss,
    }
    criterion = criterion_opts[config['stage1']['loss'].lower()]
    criterion = criterion()
    return criterion


def main(
    generator: Generator,
    dataloader: torch.utils.data.Dataset,
    device: torch.device, 
    config: dict, 
    debug: bool, 
    run: neptune.Run
) -> None:
    log.stage1.init(run, config)
    optimizer = get_optimizer(generator, config)
    criterion = get_criterion(config)
    
    for e in range(config['stage1']['epochs']):
        generator.train()
        loss_accum = []
        for idx, (img_input, img_target, coords) in enumerate(dataloader):
            img_input, img_target = tensors_to_device([img_input, img_target], device)
            
            img_generated = generator(img_input)
            
            loss = criterion(img_generated, img_input)
            loss.backward()
            loss_accum.append(loss.detach().to(cpu))
            
            optimizer.step()
            optimizer.zero_grad()
            
            if device == cuda:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            if idx == 2: break    
            
        models.utils.serialize(generator, e)
        log.stage1.epoch(run, {'mse': mean(loss_accum)})
        