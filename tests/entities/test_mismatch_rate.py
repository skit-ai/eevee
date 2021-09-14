import pytest

from eevee.metrics.slot_filling import mismatch_rate

@pytest.mark.parametrize(
    "y_true, y_pred, mmr", 
    [
        (
            [            
                {
                "text": "25th April",
                "type": "date",
                "values": [{"type": "value", "value": "2019-04-25T00:00:00+05:30"}],
                },
                {
                "text": "25th April",
                "type": "date",
                "values": [{"type": "value", "value": "2019-04-25T00:00:00+05:30"}],
                },
            ],
            [
                {
                "text": "25th April",
                "type": "date",
                "values": [{"type": "value", "value": "2019-04-25T00:00:00+05:30"}],
                },
                {
                "text": "22nd April",
                "type": "date",
                "values": [{"type": "value", "value": "2019-04-22T00:00:00+05:30"}],
                },
            ],
            0.5
        )
    ]
)
def test_mmr(y_true, y_pred, mmr):
    assert mismatch_rate(y_true, y_pred) == mmr
