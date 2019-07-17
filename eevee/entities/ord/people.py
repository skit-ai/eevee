"""
People entity mostly deals with equality. We assume types are alright in
all the components here.
"""
from typing import List, Dict

def people_sum(ents: List[Dict]):
    total_people_child = 0
    total_adult_child = 0
    total_male_female_child = 0
    total_veg_nonveg = 0

    for ent in ents:
        for val in ent["values"]:
            unit = val.get("unit")
            value = val.get("value")
            if unit:
                # 1. Add up people entity with unit person | child
                if unit in ("person", "child"):
                    total_people_child += value

                # 2. Add up people entity with unit adult | child
                if unit in ("adult", "child"):
                    total_adult_child += value

                # 3. Add up people entity with unit male | female | child
                if unit in ("male", "female", "child"):
                    total_male_female_child += value

                # 4. Add up people entity with unit veg | non veg
                if unit in ("veg", "nonveg"):
                    total_veg_nonveg += value

    total_people = max(total_people_child, total_adult_child, total_male_female_child, total_veg_nonveg)

    if total_people == 0:
        for ent in ents:
            if ent["type"] == "number":
                total_people = ent["values"][0]["value"]
                break
    return total_people


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

    if unit_sum:
        truth_sum = people_sum(truth_s)
        pred_sum = people_sum(pred_s)

        return truth_sum == pred_sum

    if len(truth_s) == len(pred_s):
        return all(eq(p1, p2, match_units=False) for p1, p2 in zip(truth_s, pred_s))
    else:
        return False
