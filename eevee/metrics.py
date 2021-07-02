from typing import List, Dict, Any

from sklearn.metrics import confusion_matrix
from eevee.types import SlotLabel
from itertools import groupby

import numpy as np


AlternativeMetric = Dict[str, Any]


def aggregate_metrics(alternative_metrics: List[AlternativeMetric], aggregation_fn=np.mean) -> AlternativeMetric:
    """
    Aggregate metric dictionaries from multiple alternatives using
    `aggregation_fn`.

    An alternative metric books like the following:
    {
      "base": {"metric-name": <metric-value>},
      "lemmatized": {...},
      "stopword": {...},
      "hypothesis": <str>
    }
    """
    # Assuming first alternative has all the keys that are involved.
    variants = alternative_metrics[0].keys()
    # Skipping these items. They don't make sense from aggregation standpoint.
    variant_blacklist = {"hypothesis"}

    output = {}
    for variant in variants:
        if variant in variant_blacklist:
            continue

        metric_dicts = [am[variant] for am in alternative_metrics]
        output[variant] =  {
            name: aggregation_fn([m[name] for m in metric_dicts])
            for name in metric_dicts[0].keys()
        }

    return output


def slot_capture_rate() -> float:
    ...


def slot_retry_rate() -> float:
    ...


def slot_mismatch_rate(y_true: List[SlotLabel], y_pred: List[SlotLabel]) -> float:
    ...


def top_k_slot_mismatch_rate(y_true: List[SlotLabel], y_pred: List[List[SlotLabel]], k=1) -> float:
    ...


def slot_fnr(y_true: List[SlotLabel], y_pred: List[SlotLabel]) -> float:
    """
    False negative rate for slot prediction.

    Slot type is handled outside this, so you will have to segregate the slot
    labels based on types beforehand.
    """

    _y_true = [0 if y is None else 1 for y in y_true]
    _y_pred = [0 if y is None else 1 for y in y_pred]

    mat = confusion_matrix(_y_true, _y_pred, labels=[0, 1])

    fn = mat[1, 0]
    tp = mat[1, 1]

    if (fn + tp) == 0:
        return 0
    else:
        return fn / (fn + tp)


def slot_fpr(y_true: List[SlotLabel], y_pred: List[SlotLabel]) -> float:
    ...
