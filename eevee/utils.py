from typing import Dict, List


def match_dict(a: Dict, b: Dict, ignore_keys: List[str] = None) -> bool:
    """
    General purpose exact match for dictionaries
    """

    ignore_keys = ignore_keys or []
    return {k: a[k] for k in a if k not in ignore_keys} == {k: b[k] for k in b if k not in ignore_keys}
