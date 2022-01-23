import neptune.new as neptune
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.models as models
from src.data import dataloader_split
from src.models import Generator
from src.logger import log
from src.training.common import *
    
    
def get_criterion(config: dict) -> torch.nn.Module:
    return criterion_opts[config['stage1']['loss'].lower()]()


def get_epochs(config: dict) -> int:
    return config['stage1']['epochs']


def get_optimizer(model: nn.Module, config: dict) -> torch.nn.Module:
    optimizer = optimizer_opts[config['stage1']['netG']['optimizer'].lower()]
    optimizer = optimizer(
        model.parameters(),
        lr=config['stage1']['netG']['lr'],
        weight_decay=config['stage1']['netG']['weight_decay'],
    )
    return optimizer


def main(
    netG: Generator,
    dataloader: torch.utils.data.Dataset,
    device: torch.device, 
    config: dict, 
    debug: bool, 
    run: neptune.Run
) -> None:
    criterion = get_criterion(config)
    optimizer = get_optimizer(netG, config)
    epochs = get_epochs(config)
    log.stage1.init(run, optimizer, criterion, epochs)
    
    train, test = dataloader_split(dataloader, config)
        
    for e in range(epochs):
        netG.train()
        loss_accum = []
        for idx, (img_input, img_target, coords) in enumerate(train):
            if config['stage1']['limit_iters'] and idx == config['stage1']['limit_iters'] - 1:
                break
            img_input, img_target = tensors_to_device([img_input, img_target], device)
            
            img_generated = netG(img_input)
            
            loss = criterion(img_generated, img_target)
            loss.backward()
            loss_accum.append(loss.detach().to(cpu))
            
            optimizer.step()
            optimizer.zero_grad()
            
            if device == cuda:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
        models.utils.serialize(netG, e)
        log.stage1.epoch.train(run, {'mse': mean(loss_accum)})
        
        mse_accum = []
        mae_accum = []
        netG.eval()
        with torch.no_grad():
            for idx, (img_input, img_target, coords) in enumerate(test):
                img_input, img_target = tensors_to_device([img_input, img_target], device)
                img_generated = netG(img_input)
                
                mse = F.mse_loss(img_generated, img_target)
                mse_accum.append(mse.to(cpu))
                
                mae = F.l1_loss(img_generated, img_target)
                mae_accum.append(mae.to(cpu))
        
        log.stage1.epoch.test(run, {'mse': mean(mse_accum), 'mae': mean(mae_accum)})
            