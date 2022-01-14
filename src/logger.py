import os
import yaml

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
