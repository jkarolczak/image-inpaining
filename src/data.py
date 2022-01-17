import os
from typing import Tuple, Union

import cv2
import pandas as pd
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dir: str = 'data/',
        image_dir: str = 'img/',
        input_dir: str = 'input/',
        target_dir: str = 'target/',
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
        self.snippet_size = snippet_size
        
        df = pd.read_csv(itemlist_path)
        
        self.item_list = df[filename_col]
        self.bboxes = df[['x', 'y', 'width', 'height']]
        
    def __len__(self) -> int:
        return self.item_list.shape[0]
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img_name = self.item_list[idx]

        input_path = os.path.join(self.dir, self.image_dir, self.input_dir, img_name)
        input_img = cv2.imread(input_path)
        input_img = torch.tensor(input_img, dtype=torch.float32)

        target_path = os.path.join(self.dir, self.image_dir, self.target_dir, img_name)
        target_img = cv2.imread(target_path)
        target_img = torch.tensor(target_img, dtype=torch.float32)
        
        coords = torch.tensor(self.bboxes.iloc[idx].values, dtype=torch.int8)
        
        return (input_img, target_img, coords)
    