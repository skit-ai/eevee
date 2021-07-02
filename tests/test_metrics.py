import pytest
from eevee.metrics import slot_fnr, aggregate_metrics


@pytest.mark.parametrize("y_true, y_pred, fnr", [
    ([None, None, None, None], [None, None, None, None], 0),
    ([None, None, None, None], [{}, None, None, None], 0),
    ([None, {}, None, None], [None, None, None, None], 1),
    ([None, {}, {}, None], [None, None, {}, None], 0.5),
    ([None, {}, {}, None], [None, None, None, {}], 1)
])
def test_slot_fnr(y_true, y_pred, fnr):
    assert slot_fnr(y_true, y_pred) == fnr


@pytest.mark.parametrize("ams, output", [
    ([
        {"base": {"wer": 1}, "lem": {"wer": 0}},
        {"base": {"wer": 1}, "lem": {"wer": 2}},
        {"base": {"wer": 1}, "lem": {"wer": 2}},
        {"base": {"wer": 1}, "lem": {"wer": 4}},
    ],
     {"base": {"wer": 1}, "lem": {"wer": 2}})
])
def test_aggregate_metrics(ams, output):
    assert aggregate_metrics(ams) == output


@pytest.mark.parametrize("ams, output", [
    ([
        {"base": {"wer": 1}, "lem": {"wer": 0}},
        {"base": {"wer": 1}, "lem": {"wer": 2}},
        {"base": {"wer": 1}, "lem": {"wer": 2}},
        {"base": {"wer": 1}, "lem": {"wer": 4}},
    ],
     {"base": {"wer": 1}, "lem": {"wer": 0}})
])
def test_aggregate_metrics_min(ams, output):
    assert aggregate_metrics(ams, min) == output
