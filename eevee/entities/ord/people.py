"""
People entity mostly deals with equality. We assume types are alright in
all the components here.
"""
from typing import List, Dict


def people_sum(ents: List[Dict]):
    max_person = max([val.get("value") for ent in ents for val in ent["values"] if val.get('unit') in ["person"]], default=0)
    max_adult = max([val.get("value") for ent in ents for val in ent["values"] if val.get('unit') in ["adult"]], default=0)
    max_child = max([val.get("value") for ent in ents for val in ent["values"] if val.get('unit') == "child"], default=0)
    max_male = max([val.get("value") for ent in ents for val in ent["values"] if val.get('unit') == "male"], default=0)
    max_female = max([val.get("value") for ent in ents for val in ent["values"] if val.get('unit') == "female"], default=0)
    max_veg = max([val.get("value") for ent in ents for val in ent["values"] if val.get('unit') == "veg"], default=0)
    max_nonveg = max([val.get("value") for ent in ents for val in ent["values"] if val.get('unit') == "nonveg"], default=0)

    total_people_child = max_person
    total_adult_child = max_adult + max_child
    total_male_female_child = max_male + max_female + max_child
    total_veg_nonveg = max_veg + max_nonveg

    total_people = max(total_people_child, total_adult_child,
                       total_male_female_child, total_veg_nonveg)

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
