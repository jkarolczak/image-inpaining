import os
from typing import Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch


class GeneratedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dir: str = 'dataset/',
        image_dir: str = 'img/',
        item_list: str = 'dataset.csv',
        filename_col: str = 'image_id',
        x1_col: str = 'x_1',
        y1_col: str = 'y_1',
        width_col: str = 'width',
        height_col: str = 'height',
        dataset_size: int = 25000,
        snippet_size: Union[int, Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]] = ((25, 55), (25, 55)),
        seed: int = 23
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)

        path = os.path.join(dir, item_list)
        self.dir = dir
        self.image_dir = image_dir
        self.dataset_size = dataset_size
        self.item_df = pd.read_csv(path)[filename_col]
        self.item_list = self.rng.choice(self.item_df, size=self.dataset_size, replace=False, shuffle=False)
        self.snippet_size = snippet_size

        self.filename_col = filename_col
        
    def __len__(self) -> int:
        return self.dataset_size
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img_name = self.item_list[idx]
        img_path = os.path.join(self.dir, self.image_dir, img_name)
        target_img = cv2.imread(img_path)
        input_img, local_boundary = self._remove_snippet(target_img)

        return (img_name, target_img, input_img, local_boundary)
    
    def _remove_snippet(self, image: np.ndarray) -> np.ndarray:
        img = image.copy()
        size = self.snippet_size
        if isinstance(size, int):
            width, height = [size] * 2
        elif isinstance(size, tuple):
            if isinstance(size[0], int):
                width, height = [self.rng.integers(*size)] * 2
            else:
                width = self.rng.integers(*size[0])
                height = self.rng.integers(*size[1])
        y = 45 + self.rng.integers(image.shape[0] - height - 90)        
        x = 35 + self.rng.integers(image.shape[1] - width - 70)
        img[y:y + height, x:x + width] = np.array([255, 255, 255])
        return (img, (x, y, width, height))
    