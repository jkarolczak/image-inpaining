import os
import yaml

import torch.nn as nn

import neptune.new as neptune


def get_run(
    cfg_dir: str = 'cfg', 
    neptune_yaml: str = 'neptune.yaml',
    debug: bool = False
) -> neptune.Run:
    yaml_path = os.path.join(cfg_dir, neptune_yaml)
    with open(yaml_path, 'r') as fd:
        secrets = yaml.safe_load(fd)
    return neptune.init(
        project=secrets['project'],
        api_token=secrets['api_token'],
        mode='debug' if debug else 'async'
    )

class log:
    class stage1:
        @staticmethod
        def init(run: neptune.Run, config: dict) -> None:
            run["stage1/learning_rate"] = config['stage1']['lr']
            run["stage1/weight_decay"] = config['stage1']['weight_decay']
            run["stage1/optimizer"] = config['stage1']['optimizer'].lower()
        
        @staticmethod
        def epoch(run: neptune.Run, metrics: dict) -> None:
            for key, value in metrics.items():
                run[f"stage1/train/{key}"] = value