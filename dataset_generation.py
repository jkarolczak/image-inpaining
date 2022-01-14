import os
import yaml

import cv2
import pandas as pd

from src.data_generator import GeneratedDataset
from src.visualization import transform_to_image


if __name__ == "__main__":
    
    with open("cfg/dataset.yaml", 'r') as fd:
        params = yaml.safe_load(fd)
    seed = params['generate']['seed']
    dataset_size = params['generate']['dataset_size']

    target_path = "data/img/target"
    input_path = "data/img/input"
    paths = [target_path, input_path]

    os.makedirs("data/img", exist_ok=True)

    for path in paths:
        if not os.path.isdir(path):
            os.mkdir(path)
        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
            if not os.path.isfile(file_path):
                continue
            os.remove(file_path)

    image_ids = {'image_id': [], 'x': [], 'y': [], 'width': [], 'height': []}

    dataGenerator = GeneratedDataset(seed = seed, dataset_size = dataset_size)

    for (img_name, target_img, input_img, boundary) in dataGenerator:
        x, y, w, h = boundary
        image_ids['image_id'].append(img_name)
        image_ids['x'].append(x)
        image_ids['y'].append(y)
        image_ids['width'].append(w)
        image_ids['height'].append(h)

        imgs = [target_img, input_img]

        for (img, path) in zip(imgs, paths):
            file_path = os.path.join(path, img_name)
            img = transform_to_image(img)
            cv2.imwrite(file_path, img)
    
    image_df = pd.DataFrame(image_ids)
    image_df.to_csv("data/dataset.csv")
