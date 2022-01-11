import cv2
import os
import yaml
import numpy as np
import pandas as pd

from src.data_generator import GeneratedDataset
from src.visualization import transformToImage


if __name__ == "__main__":
    
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)
    seed = params['generate']['seed']
    dataset_size = params['generate']['dataset_size']

    target_path = "data/img/target"
    input_path =  "data/img/input"
    local_path = "data/img/local"
    paths = [target_path, input_path, local_path]

    for path in paths:
        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
            if not os.path.isfile(file_path):
                continue
            os.remove(file_path)

    image_ids = {'image_id': []}

    dataGenerator = GeneratedDataset(seed = seed, dataset_size = dataset_size)

    for (img_name, target_img, input_img, local_img) in dataGenerator:
        image_ids['image_id'].append(img_name)
        imgs = [target_img, input_img, local_img]

        for (img, path) in zip(imgs, paths):
            file_path = os.path.join(path, img_name)
            img = transformToImage(img)
            cv2.imwrite(file_path, img)
    
    image_df = pd.DataFrame(image_ids)
    image_df.to_csv("data/dataset.csv")

        
