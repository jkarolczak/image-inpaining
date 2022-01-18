from typing import List, Union

import torch
import torch.nn as nn

cpu = torch.device('cpu')
cuda = torch.device('cuda')

criterion_opts = {
    'bce': nn.BCELoss,
    'mae': nn.L1Loss,
    'mse': nn.MSELoss,
    'l1': nn.L1Loss
}

optimizer_opts = {
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'sgd': torch.optim.SGD,
}


def mean(
    input: List[Union[float, int, torch.Tensor]]
) -> float:
    if not len(input):
        return 0.0
    return float(sum(input) / len(input))


def tensors_to_device(
    tensors: List,
    device: torch.device
) -> List[torch.Tensor]:
    return [t.requires_grad_(True).to(device) for t in tensors]
  