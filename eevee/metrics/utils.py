"""
Common utilities for calculating metrics.
"""

from typing import Any, Tuple

from sklearn.metrics import confusion_matrix


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
