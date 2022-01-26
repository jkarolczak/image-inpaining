from typing import Tuple

import neptune.new as neptune
import torch
import torch.nn as nn

import src.models as models
from src.data import dataloader_split
from src.models import Generator, GlobalDiscriminator, LocalDiscriminator
from src.logger import log
from src.training.common import *
from src.visualization import *


def get_criterion(config: dict) -> torch.nn.Module:
    return [criterion_opts[config['stage2']['discriminator_loss'].lower()]() for _ in range(2)] + [criterion_opts[config['stage2']['generator_loss'].lower()]()]


def get_epochs(config: dict) -> int:
    return config['stage2']['epochs']


def get_iter_limit(config: dict) -> int:
    return config['stage2']['limit_iters']

def get_generator_weights(config: dict) -> dict:
    return {
        'mse': float(config['stage2']['netG']['mse_weight']),
        'local': float(config['stage2']['netG']['local_weight']),
        'global': float(config['stage2']['netG']['global_weight'])
    }

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


def get_labels(
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    shape = (dataloader.batch_size, 1)
    real = torch.ones(shape, device=device)
    fake = torch.zeros(shape, device=device)
    return real, fake


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
    criterionGD, criterionLD, criterionG = get_criterion(config)
    optim_netG, optim_netGD, optim_netLD = get_optimizers(netG, netGD, netLD, config)
    epochs = get_epochs(config)
    iter_limit = get_iter_limit(config)
    generator_weights = get_generator_weights(config)
    log.stage2.init(run, optim_netG, optim_netGD, optim_netLD, criterionGD, config)

    real_label, fake_label = get_labels(dataloader, device)
    real_local_label, fake_local_label = torch.ones((1, 1), device=device), torch.zeros((1, 1), device=device)

    for e in range(epochs):
        loss_G_accum, loss_GD_accum, loss_LD_accum = [], [], []
        netG.train(); netGD.train(), netLD.train()
        for idx, (img_input, img_target, coords) in enumerate(dataloader):
            if config['stage2']['limit_iters'] and idx == config['stage2']['limit_iters']:
                break
            
            noise_bound = max(int(255 * (((2 / 3) ** e) / 4)), 1)
            img_target += torch.randint_like(img_target, -1 * noise_bound, noise_bound)
            img_target = torch.clip(img_target, 0, 255)
            
            img_input, img_target = tensors_to_device([img_input, img_target], device)
                
            # -----------------
            #  Train Global Discriminator
            # -----------------
            
            if not e % config['stage2']['netGD']['train_every']:
                netGD.zero_grad()   
                
                img_real_GD = netGD(img_target)
                loss_GD_real = criterionGD(img_real_GD, real_label)
                
                img_fake = netG(img_input)
                img_fake_GD = netGD(img_fake.detach())
                loss_GD_fake = criterionGD(img_fake_GD, fake_label)
                
                loss_GD = (loss_GD_real + loss_GD_fake) / 2
                loss_GD.backward()

                optim_netGD.step()

            # -----------------
            #  Train Local Discriminator
            # -----------------

            if not e % config['stage2']['netLD']['train_every']:
                netLD.zero_grad()

                losses_LD = []
                for (target, fake, coord) in zip(img_target, img_fake.detach(), coords):
                    x, y, w, h = coord
                    local_target = target[y: y + h, x: x + w]
                    local_fake = fake[y: y + h, x: x + w]

                    local_real_LD = netLD(local_target)
                    loss_LD_real = criterionLD(local_real_LD, real_local_label)

                    local_fake_LD = netLD(local_fake)
                    loss_LD_fake = criterionLD(local_fake_LD, fake_local_label)
                    losses_LD.append((loss_LD_real + loss_LD_fake) / 2)

                loss_LD = sum(losses_LD)
                loss_LD.backward()

                optim_netLD.step()
            
            # -----------------
            #  Train Generator
            # -----------------

            if not e % config['stage2']['netG']['train_every']:
                netG.zero_grad()
                img_fake_G = netGD(img_fake)
                loss_img_G = criterionGD(img_fake_G, real_label)

                mse_loss_G = criterionG(img_fake, img_target)

                local_losses_G = []
                for (fake, coord) in zip(img_fake, coords):
                    x, y, w, h = coord
                    local_fake = fake[y: y + h, x: x + w]

                    local_fake_G = netLD(local_fake)
                    loss_local_G = criterionLD(local_fake_G, real_local_label)
                    local_losses_G.append(loss_local_G)

                loss_local_G = sum(local_losses_G)
                loss_G = generator_weights['mse'] * mse_loss_G + generator_weights['global'] * loss_img_G + generator_weights['local'] * loss_local_G
                loss_G.backward()
        
                optim_netG.step()

                loss_G_accum.append(loss_G.to(cpu))
                
            loss_GD_accum.append(loss_GD.to(cpu))
            loss_LD_accum.append(loss_LD.to(cpu))
            
        if device == cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    
        
        models.utils.serialize(netG, e + config['stage1']['epochs'])
        models.utils.serialize(netGD, e + config['stage1']['epochs'])  
        log.stage2.epoch.train(run, {
                                   'bceG': mean(loss_G_accum), 
                                   'bceGD': mean(loss_GD_accum), 
                                   'bceLD': mean(loss_LD_accum)
                                })      
                    