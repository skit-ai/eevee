import pytest
from eevee.metrics import slot_fnr


@pytest.mark.parametrize("y_true, y_pred, fnr", [
    ([None, None, None, None], [None, None, None, None], 0),
    ([None, None, None, None], [{}, None, None, None], 0),
    ([None, {}, None, None], [None, None, None, None], 1),
    ([None, {}, {}, None], [None, None, {}, None], 0.5),
    ([None, {}, {}, None], [None, None, None, {}], 1)
])
def test_slot_fnr(y_true, y_pred, fnr):
    assert slot_fnr(y_true, y_pred) == fnr
