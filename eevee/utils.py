import os
from typing import Dict, List

import yaml


def parse_yaml(path_to_file: os.PathLike) -> Dict:

    with open(path_to_file, "r") as fp:
        loaded_yaml : Dict[str, List[str]] = yaml.safe_load(fp)

    return loaded_yaml