import numpy as np
import pandas as pd
import pytest
from eevee.metrics import (intent_report, slot_capture_rate, slot_fnr,
                           slot_fpr, slot_mismatch_rate, slot_retry_rate)


@pytest.mark.parametrize("slots_predicted, expected_slot, scr", [
    (["yes", "yes", "no", None], "no", 0.25),
    (["yes", "yes", "no", None], "yes", 0.5),
])
def test_slot_scr(slots_predicted, expected_slot, scr):
    assert slot_capture_rate(slots_predicted, expected_slot) == scr


@pytest.mark.parametrize("slot_turn_counts, agg_fn, srr", [
    ([1, 2, 3, 2, 1, 1, 4, 5, None, None], np.mean, 2.375),
    ([1, 2, 3, 2, 1, 1, 4, 5, None, None], np.median, 2.0),
])
def test_slot_srr(slot_turn_counts, agg_fn, srr):
    assert slot_retry_rate(slot_turn_counts, agg_fn) == srr


@pytest.mark.parametrize("y_true, y_pred, smr", [
    ([None, None, None, None], [None, None, None, None], 0),
    ([1, 1, 1, 1, None], [1, 1, 1, 2, None], 0.25),
])
def test_slot_smr(y_true, y_pred, smr):
    assert slot_mismatch_rate(y_true, y_pred) == smr


@pytest.mark.parametrize("y_true, y_pred, fnr", [
    ([None, None, None, None], [None, None, None, None], 0),
    ([None, None, None, None], [{}, None, None, None], 0),
    ([None, {}, None, None], [None, None, None, None], 1),
    ([None, {}, {}, None], [None, None, {}, None], 0.5),
    ([None, {}, {}, None], [None, None, None, {}], 1)
])
def test_slot_fnr(y_true, y_pred, fnr):
    assert slot_fnr(y_true, y_pred) == fnr


@pytest.mark.parametrize("y_true, y_pred, fpr", [
    ([None, None, None, None], [None, None, None, None], 0),
    ([None, None, None, None], [{}, None, None, None], 0.25),
    ([None, {}, None, None], [None, None, None, None], 0),
    ([None, None, {}], [None, {}, None], 0.5),
    ([None, {}, {}, None], [None, None, None, {}], 0.5)
])
def test_slot_fpr(y_true, y_pred, fpr):
    assert slot_fpr(y_true, y_pred) == fpr


@pytest.mark.parametrize("y_true, y_pred, macro_f1", [
    ([{"id": 1, "intent": "a"}],
     [{"id": 1, "intent": "b"}, {"id": 2, "intent": "a"}],
     0.0),
    ([{"id": 1, "intent": "a"}],
     [{"id": 1, "intent": "a"}, {"id": 2, "intent": "a"}],
     1.0)
])
def test_intents(y_true, y_pred, macro_f1):
    true_labels = pd.DataFrame(y_true)
    pred_labels = pd.DataFrame(y_pred)

    report = intent_report(true_labels, pred_labels, output_dict=True)

    print(report)
    assert report["macro avg"]["f1-score"] == macro_f1
