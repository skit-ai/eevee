"""
WER evaluation functions
"""

from typing import List

import requests
from tqdm import tqdm


def get_atlas_entity(plute_url: str, text: str):
    req = requests.post(plute_url, json={"text": text})
    ents = [ent for ent in req.json()["response"]["aux_entities"] if ent["type"] == "location"]

    return ents[0] if ents else None


def atlas_error_rate(truth_texts: List[str], pred_texts: List[str], plute_url:str) -> float:
    """
    Return errors for all the cases where atlas should return something.
    """

    cases = 0
    hits = 0
    for truth, pred in tqdm(zip(truth_texts, pred_texts), total=len(truth_texts)):
        ent = get_atlas_entity(plute_url, truth)
        if ent:
            cases += 1
            pred_ent = get_atlas_entity(plute_url, pred)
            if pred_ent and pred_ent["values"][0]["value"] == ent["values"][0]["value"]:
                hits += 1

    return 1 - (hits / cases)
