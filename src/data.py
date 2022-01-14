import os
from typing import Tuple, Union

import cv2
import numpy as np
from numpy.lib.arraysetops import isin
import pandas as pd
import torch

# needs to be adjusted to new data structure
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dir: str = 'data/',
        image_dir: str = 'img/',
        input_dir: str = 'input/',
        target_dir: str = 'target/',
        local_dir: str = 'local/',
        item_list: str = 'dataset.csv',
        filename_col: str = 'image_id',
        snippet_size: Union[int, Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]] = ((25, 55), (25, 55))
    ):
        super().__init__()
        itemlist_path = os.path.join(dir, item_list)
        self.dir = dir
        self.image_dir = image_dir
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.local_dir = local_dir
        self.item_list = pd.read_csv(itemlist_path)[filename_col]
        self.snippet_size = snippet_size
        
    def __len__(self) -> int:
        return self.item_list.shape[0]
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img_name = self.item_list[idx]

        input_path = os.path.join(self.dir, self.image_dir, self.input_dir, img_name)
        input_img = cv2.imread(input_path)
        input_img = torch.tensor(input_img, dtype=torch.float32, requires_grad=True)

        target_path = os.path.join(self.dir, self.image_dir, self.target_dir, img_name)
        target_img = cv2.imread(target_path)
        target_img = torch.tensor(target_img, dtype=torch.float32, requires_grad=True)
        
        local_path = os.path.join(self.dir, self.image_dir, self.local_dir, img_name)
        local_img = cv2.imread(local_path)
        local_img = torch.tensor(local_img, dtype=torch.float32, requires_grad=True)
        
        return (input_img, target_img, local_img)
    
    """
    def _remove_snippet(self, image: np.ndarray) -> np.ndarray:
        img = image.copy()
        size = self.snippet_size
        if isinstance(size, int):
            width, height = [size] * 2
        elif isinstance(size, tuple):
            if isinstance(size[0], int):
                width, height = [np.random.randint(*size)] * 2
            else:
                width = np.random.randint(*size[0])
                height = np.random.randint(*size[1])
        y = 30 + np.random.randint(img.shape[1] - height - 60)        
        x = 30 + np.random.randint(img.shape[1] - width - 60)
        img[y:y + height, x:x + width] = np.array([255, 255, 255])
        return img
    """