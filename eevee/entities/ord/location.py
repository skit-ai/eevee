"""
Location entity mostly deals with equality. We assume types are alright in
all the components here.
"""
from typing import List, Dict

def eq(a: Dict, b: Dict) -> bool:
    return a["values"] == b["values"]


def eq_lists(a: List[Dict], b: List[Dict]) -> bool:

    if len(a) != 0 and len(b) != 0:
        return sorted([ent["values"][0]["value"] for ent in a]) == sorted([ent["values"][0]["value"] for ent in b])

    elif len(a) == len(b) == 0:
        return True

    return False
