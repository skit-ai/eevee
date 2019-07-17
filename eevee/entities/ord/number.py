from typing import Dict


def eq(a: Dict, b: Dict) -> bool:
    return a["values"] == b["values"]
