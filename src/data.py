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
        item_list: str = 'dataset.csv',
        filename_col: str = 'image_id',
        snippet_size: Union[int, Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]] = ((25, 55), (25, 55))
    ):
        super().__init__()
        path = os.path.join(dir, item_list)
        self.dir = dir
        self.image_dir = image_dir
        self.item_list = pd.read_csv(path)[filename_col]
        self.snippet_size = snippet_size
        
    def __len__(self) -> int:
        return self.item_list.shape[0]
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img_name = self.item_list[idx]
        img_path = os.path.join(self.dir, self.image_dir, img_name)
        img = cv2.imread(img_path)
        input = self._remove_snippet(img)
        input = torch.tensor(input, dtype=torch.float32, requires_grad=True)
        groundtruth = torch.tensor(img, dtype=torch.float32, requires_grad=True)
        
        return (input, groundtruth)
    
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