from typing import List

from sklearn.metrics import confusion_matrix

from eevee.types import SlotLabel


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
