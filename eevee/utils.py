from typing import Dict, List
import yaml

def parse_yaml(alias_yaml: str) -> Dict:

    with open(alias_yaml, "r") as fp:
        loaded_yaml : Dict[str, List[str]] = yaml.safe_load(fp)

    return loaded_yaml