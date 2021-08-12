"""
Amount of money entity mostly deals with equality. We assume types are alright in
all the components here.
"""
from typing import List

from eevee.types import Entity


def eq(a: Entity, b: Entity, match_units=False) -> bool:
    """
    If `match_units` is False, only see if the numerical values are the same.
    Keeping the match_unit to False, as default because plute is summing
    up the entities therefore, we only worry about the sum not the unit.
    """

    if match_units:
        return a["values"] == b["values"]
    else:
        return a["values"][0]["value"] == b["values"][0]["value"]


def eq_lists(truth: List[Entity], pred: List[Entity], only_first_value: bool = False) -> bool:
    """
    Tell the predictions are matching with the truth.
    only_first_value: Only checks for pred[0] in truth
    """

    truth_s = truth
    pred_s = pred

    if len(pred_s) == len(truth_s) == 0:
        return True

    if only_first_value:
        return any(eq(tr, pred_s[0]) for tr in truth_s)

    if len(truth_s) == len(pred_s):
        return all(eq(p1, p2, match_units=False) for p1, p2 in zip(truth_s, pred_s))
    else:
        return False
