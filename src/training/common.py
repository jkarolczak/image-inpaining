from typing import List, Union

import torch

cpu = torch.device('cpu')
cuda = torch.device('cuda')


def mean(
    input: List[Union[float, int, torch.Tensor]]
) -> float:
    return float(sum(input) / len(input))


def tensors_to_device(
    tensors: List,
    device: torch.device
) -> List[torch.Tensor]:
    return [t.requires_grad_(True).to(device) for t in tensors]
  