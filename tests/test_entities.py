import pytest

from eevee.entities.ord.people import people_sum
import eevee.entities.ord.datetime as ord_datetime


@pytest.mark.parametrize("ents, result", [
    ([
        {
            "body": "4",
            "latent": False,
            "values": [
                {
                    "value": 4,
                    "type": "value"
                }
            ],
            "type": "number",
            "range": {
                "start": 5,
                "end": 6
            },
            "parser": "duckling",
            "alternative_index": 0,
            "ambiguous": True
        },
        {
            "body": "3",
            "latent": False,
            "values": [
                {
                    "value": 3,
                    "type": "value"
                }
            ],
            "type": "number",
            "range": {
                "start": 0,
                "end": 1
            },
            "parser": "duckling",
            "alternative_index": 0,
            "ambiguous": True
        }], 4),
    ([
        {
            "body": "4 people",
            "latent": False,
            "values": [
                {
                    "value": 4,
                    "type": "value",
                    "unit": "person"
                }
            ],
            "type": "people",
            "range": {
                "start": 0,
                "end": 8
            },
            "parser": "duckling",
            "alternative_index": 0
        }], 4)
])
def test_people_sum(ents, result):

    assert people_sum(ents) == result


@pytest.mark.parametrize("truth, pred, is_equal", [
    (
        {
            "body": "day after tomorrow",
            "latent": False,
            "values": [
                {
                    "value": "2019-05-15T00:00:00+05:30",
                    "grain": "day",
                    "type": "value"
                }
            ],
            "type": "date",
            "range": {
                "start": 0,
                "end": 18
            },
            "parser": "duckling",
            "alternative_index": 0
        },
        {
            "body": "day after tomorrow 6 pm",
            "latent": False,
            "values": [
                {
                    "value": "2019-05-15T18:00:00+05:30",
                    "grain": "hour",
                    "type": "value"
                }
            ],
            "type": "datetime",
            "range": {
                "start": 0,
                "end": 23
            },
            "parser": "duckling",
            "alternative_index": 0
        },
        True
    ),
    (
        {
            "body": "today",
            "latent": False,
            "values": [
                {
                    "value": "2019-05-13T00:00:00+05:30",
                    "grain": "day",
                    "type": "value"
                }
            ],
            "type": "date",
            "range": {
                "start": 0,
                "end": 5
            },
            "parser": "duckling",
            "alternative_index": 0
        },
        {
            "body": "today dinner",
            "latent": False,
            "values": [
                {
                    "type": "interval",
                    "value": {
                        "to": "2019-05-14T00:00:00+05:30",
                        "from": "2019-05-13T18:00:00+05:30"
                    },
                    "grain": "hour"
                }
            ],
            "type": "datetime",
            "range": {
                "start": 0,
                "end": 12
            },
            "parser": "duckling",
            "alternative_index": 0
        },
        True
    ),
    (
        {
            "body": "today",
            "latent": False,
            "values": [
                {
                    "value": "2019-05-13T00:00:00+05:30",
                    "grain": "day",
                    "type": "value"
                }
            ],
            "type": "date",
            "range": {
                "start": 0,
                "end": 5
            },
            "parser": "duckling",
            "alternative_index": 0
        },
        {
            "body": "tomorow",
            "latent": False,
            "values": [
                {
                    "value": "2019-05-14T00:00:00+05:30",
                    "grain": "day",
                    "type": "value"
                }
            ],
            "type": "date",
            "range": {
                "start": 0,
                "end": 7
            },
            "parser": "duckling",
            "alternative_index": 0
        },
        False
    )
])
def test_date_eq(truth, pred, is_equal):
    assert ord_datetime.date_eq(truth, pred) == is_equal
