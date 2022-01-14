from typing import Union

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

def transform_to_image(
    image: Union[np.ndarray, torch.Tensor]
) -> np.ndarray:
    if isinstance(image, np.ndarray):
        if len(image.shape) == 4:
            image = np.squeeze(image, axis=0)
        
    elif isinstance(image, torch.Tensor):
        if len(image.shape) == 4:
            image = image.squeeze(0)
        image = image.detach().cpu().numpy()
    else:
        raise ValueError("Image has to be np.ndarray or torch.Tensor")

    image = image.astype(np.uint8)
    return image


def imshow(
    image: Union[np.ndarray, torch.Tensor]    
) -> None:        
    image = transform_to_image(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.show()
