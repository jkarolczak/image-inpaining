import os
import yaml

import torch
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


def log_metrics(
    run: neptune.Run, 
    metrics: dict, 
    stage: str, 
    phase: str
) -> None:
    for key, value in metrics.items():
        run[f"{stage}/{phase}/{key}"] = value


class log:
    class stage1:
        @staticmethod
        def init(
            run: neptune.Run, 
            optimizer: torch.optim.Optimizer, 
            criterion: nn.Module,
            epochs: int
        ) -> None:
            run["stage1/learning_rate"] = optimizer.defaults['lr']
            run["stage1/weight_decay"] = optimizer.defaults['weight_decay']
            run["stage1/optimizer"] = type(optimizer).__name__
            run["stage1/criterion"] = type(criterion).__name__ 
            run["stage1/epochs"] = epochs
            
        class epoch:
            @staticmethod
            def train(run: neptune.Run, metrics: dict) -> None:
                log_metrics(run, metrics, "stage1", "train")
                
            @staticmethod
            def test(run: neptune.Run, metrics: dict) -> None:
                log_metrics(run, metrics, "stage1", "test")
    
    class stage2:
        @staticmethod
        def init(
            run: neptune.Run,
            *args
        ) -> None:
            pass
        
        class epoch:
            @staticmethod
            def train(run: neptune.Run, metrics: dict) -> None:
                log_metrics(run, metrics, "stage2", "train")
                
            @staticmethod
            def test(run: neptune.Run, metrics: dict) -> None:
                log_metrics(run, metrics, "stage2", "test")
                