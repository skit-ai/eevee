"""
People entity mostly deals with equality. We assume types are alright in
all the components here.
"""
from typing import List, Dict


def eq(a: Dict, b: Dict, match_units=True) -> bool:
    """
    If `match_units` is False, only see if the numerical values are the same.
    """

    if match_units:
        return a["values"] == b["values"]
    else:
        return a["values"][0]["value"] == b["values"][0]["value"]


def eq_lists(truth: List[Dict], pred: List[Dict], unit_sum=True) -> bool:
    """
    Tell the predictions are matching with the truth.
    """

    truth_s = truth
    pred_s = pred

    if len(truth_s) == len(pred_s):
        return all(eq(p1, p2, match_units=False) for p1, p2 in zip(truth_s, pred_s))
    else:
        return False
