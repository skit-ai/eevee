import json
import pandas as pd

from eevee.metrics.entity import entity_report


def test_dfs():

    columns = ["id", "entities"]

    true = [
        [
            1,
            [{
                "text": "25th April",
                "type": "date",
                "values": [{"type": "value", "value": "2019-04-25T00:00:00+05:30"}],
            }],
        ],
        [
            2,
            [{
                "text": "25th April",
                "type": "date",
                "values": [{"type": "value", "value": "2019-04-25T00:00:00+05:30"}],
            }],
        ],
        [
            3,
            [{
                "text": "25th April",
                "type": "date",
                "values": [{"type": "value", "value": "2019-04-25T00:00:00+05:30"}],
            }],
        ],
        [
            4,
            [{
                "text": "25th April",
                "type": "date",
                "values": [{"type": "value", "value": "2019-04-25T00:00:00+05:30"}],
            }],
        ],
        [
            5,
            [{
                'text': "eight o'clock in the evening",
                'type': 'time',
                'values': [{'type': 'value', 'value': '2019-06-20T20:00:00+05:30'}]
            }]
        ],
        [
            6,
            [{
                'text': "eight o'clock in the evening",
                'type': 'time',
                'values': [{'type': 'value', 'value': '2019-06-20T20:00:00+05:30'}]
            }]
        ],
        [
            7,
            [{
                'text': "eight o'clock in the evening",
                'type': 'time',
                'values': [{'type': 'value', 'value': '2019-06-20T20:00:00+05:30'}]
            }]
        ],
        [
            8,
            [{
                'text': "eight o'clock in the evening",
                'type': 'time',
                'values': [{'type': 'value', 'value': '2019-06-20T20:00:00+05:30'}]
            }]
        ],
        [
            9,
            [{'text': '24th April noon',
            'type': 'datetime',
            'values': [{'type': 'value', 'value': '2019-04-24T12:00:00+05:30'}]}]
        ],
        [
            10,
            [{'text': '24th April noon',
            'type': 'datetime',
            'values': [{'type': 'value', 'value': '2019-04-24T12:00:00+05:30'}]}]
        ],
        [
            11,
        ],
        [
            12,
            [{'text': '24th April noon',
            'type': 'datetime',
            'values': [{'type': 'value', 'value': '2019-04-24T12:00:00+05:30'}]}]
        ],
        [
            13,
            [{'text': '24th April noon',
            'type': 'datetime',
            'values': [{'type': 'value', 'value': '2019-04-24T12:00:00+05:30'}]}]
        ],
        [
            14,
            [{'text': '6th evening',
            'type': 'time',
            'values': [{'type': 'interval',
                        'value': {'from': '2021-08-06T18:00:00.000-07:00',
                                'to': '2021-08-07T00:00:00.000-07:00'}}]}]
        ],
    ]

    pred = [
        [
            1,
            [{
                "text": "25th April",
                "type": "date",
                "values": [{"type": "value", "value": "2019-04-25T00:00:00+05:30"}],
            }],
        ],
        [
            2,
            [{
                "text": "22nd April",
                "type": "date",
                "values": [{"type": "value", "value": "2019-04-22T00:00:00+05:30"}],
            }],
        ],
        [
            3,
        ],
        [
            4,
            [{
                "text": "noon",
                "type": "time",
                "values": [{"type": "value", "value": "2019-04-25T12:00:00+05:30"}],
            }],
        ],
        [
            5,
            [{
                'text': "eight o'clock in the evening",
                'type': 'time',
                'values': [{'type': 'value', 'value': '2019-06-20T20:00:00+05:30'}]
            }]
        ],
        [
            6,
            [{
                'text': "eight o'clock in the evening",
                'type': 'time',
                'values': [{'type': 'value', 'value': '2019-06-20T08:00:00+05:30'}]
            }]
        ],
        [
            7,
        ],
        [
            8,
            [{
                'text': "eight o'clock in the evening",
                'type': 'date',
                'values': [{'type': 'value', 'value': '2019-06-08T20:00:00+05:30'}]
            }]
        ],
        [
            9,
            [{'text': '24th April noon',
            'type': 'datetime',
            'values': [{'type': 'value', 'value': '2019-04-24T12:00:00+05:30'}]}]
        ],
        [
            10,
        ],
        [
            11,
            [{'text': '24th April noon',
            'type': 'datetime',        
            'values': [{'type': 'value', 'value': '2019-04-24T12:00:00+05:30'}]}]
        ],
        [
            12,
            [{'text': '24th April noon',
            'type': 'time',
            'values': [{'type': 'value', 'value': '2019-04-24T12:00:00+05:30'}]}]
        ],
        [
            13,
            [{'text': '24th April noon',
            'type': 'date',
            'values': [{'type': 'value', 'value': '2019-04-24T12:00:00+05:30'}]}]
        ],
        [
            14,
            [{'text': '67',
            'type': 'number',
            'values': [{'type': 'value', 'value': 67}]}]
        ],
    ]

    true_labels = pd.DataFrame(true, columns=columns)
    pred_labels = pd.DataFrame(pred, columns=columns)

    true_labels["entities"] = true_labels["entities"].apply(json.dumps)
    pred_labels["entities"] = pred_labels["entities"].apply(json.dumps)

    er = entity_report(true_labels, pred_labels)

    expected_report = pd.DataFrame(
        [
            {
                "Entity": "date",
                "FPR": 2/2,
                "FNR": 4/7,
                "Mismatch Rate": 1/3,
                "Support": 4 + 4, # date + datetime
                "Positives": 3 + 4, # date + datetime
                "Negatives": 1 + 1, # date + datetime
            },
            {
                "Entity": "number",
                "FPR": 1/1,
                "FNR": 0.0,
                "Mismatch Rate": 0.0,
                "Support": 0,
                "Positives": 0,
                "Negatives": 1,
            },
            {
                "Entity": "time",
                "FPR": 2/2,
                "FNR": 5/8,
                "Mismatch Rate": 1/3,
                "Support": 4 + 5, # time + datetime
                "Positives": 3 + 5, # time + datetime
                "Negatives": 1 + 1, # date + datetime
            },
        ]
    )
    expected_report.set_index("Entity", inplace=True)

    print("entity report")
    print(er)
    print("expected report")
    print(expected_report)
    
    assert er.to_dict("records") == expected_report.to_dict("records")
