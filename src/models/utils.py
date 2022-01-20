import os
from datetime import datetime
from typing import Union

import torch
import torch.nn as nn


def serialize(
    model: nn.Module,
    epoch: int,
    directory: str = 'models'
) -> None:
    os.makedirs(directory, exist_ok=True)
    time = datetime.now().strftime('%Y-%m-%d_%H-%M')
    model_type = type(model).__name__
    file_name = f'{time}-epoch-{epoch}-{str(model_type)}.pt'
    file_path = os.path.join(directory, file_name)
    torch.save(model.state_dict(), file_path)
    
    
def deserialize(
    model: Union[nn.Module, type],
    file_name: str,
    device: Union[str, torch.device],
    directory: str = 'models'
) -> nn.Module:
    file_path = os.path.join(directory, file_name)
    state_dict = torch.load(file_path)
    if type(model, type):
        model = model()
    model.load_state_dict(state_dict)
    if type(device, str):
        device = torch.device(device)
    model.to(device)
    return model
    