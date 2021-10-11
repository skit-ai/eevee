import json
from typing import Tuple, Any

import pandas as pd
from sklearn.metrics import confusion_matrix

def parse_json_input(entity):

    if isinstance(entity, str):
        entity_list = json.loads(entity)
        if entity_list:
            for e in entity_list:
                e["type"] = e["type"].lower()
            return entity_list
    return None




def fpr_fnr(y_true, y_pred, labels: Tuple[Any, Any]) -> Tuple[Tuple[float, int], Tuple[float, int]]:
    """
    Find FPR and FNR for provided labels. Order of class in `labels` defines
    the perspective for computing these metrics. First label is read as
    positive example, second as negative. Metrics are made 0 in case of
    ZeroDivisionError.

    Return a tuple with the following two items:
    1. fpr, total negative items
    2. fnr, total positive items
    """

    if len(labels) != 2:
        raise ValueError(f"FPR and FNR can only be calculated with binary labels, {labels} was provided")

    mat = confusion_matrix(y_true, y_pred, labels=labels)

    total_neg = (mat[1, 0] + mat[1, 1])
    if total_neg > 0:
        fpr = mat[1, 0] / total_neg
    else:
        fpr = 0

    total_pos = (mat[0, 0] + mat[0, 1])

    if total_pos > 0:
        fnr = mat[0, 1] / total_pos
    else:
        fnr = 0

    return (fpr, total_neg), (fnr, total_pos)


def weighted_avg_dropna(cat_report_df: pd.DataFrame):

    wcat_df = cat_report_df.drop(index=["_"])

    support_sum = wcat_df["support"].sum()

    precision = (wcat_df["precision"] * wcat_df["support"]).sum() / support_sum
    recall = (wcat_df["recall"] * wcat_df["support"]).sum() / support_sum
    f1 = (wcat_df["f1-score"] * wcat_df["support"]).sum() / support_sum

    wad = {}
    wad["precision"] = precision
    wad["recall"] = recall
    wad["f1-score"] = f1
    wad["support"] = support_sum

    wad_df = pd.DataFrame(wad, index=["weighted average (excluding no_entity)"])

    return pd.concat([cat_report_df, wad_df])
        