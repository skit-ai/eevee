---
layout: default
title: Entities and Slots
parent: Metrics
nav_order: 2
---
Refer to [this](https://github.com/skit-ai/onboarding/blob/master/ml/slot-reporting/slot-evaluation-and-reporting.ipynb) document to understand more about slots, entities, and their metrics.

# Slots
`TODO`

# Entities

Eevee let's you calculate all the important _turn level
metrics_ for various entities. We tag these data points using tog's, an internal
tool, region tagging method.

| Metric                    | Description                                                                                                                                                                                               |
|---------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| False Negative Rate (FNR) | Ratio of turns where we missed out predicting an entity while the utterance had it.                                                                                                                       |
| False Positive Rate (FPR) | Ratio of turns where we predicted an entity while the utterance didn't have it. This usually needs attention in normalization by clearly defining what all states are going to be sampled for evaluation. |
| Mismatch Rate (MMR)       | Within entity predictions, the ratio of cases that are differing in value. For example we predicted '3' instead of '2' for a `number` entity.                                                     |

Here is the list of entities that are supported:

| Type                       | Support remarks |
|----------------------------+-----------------|
| `datetime`, `date`, `time` | internally `datetime` (given) is broken down to `date` and `time`, therefore false positives, false negatives, true positives are considered along `date` & `time` and reported outside.          |
| `pattern`                  | not yet supported         |
| `number`                   | supported superficially only. `number` and `people` are supported interchangeably at this point.          |

## Data schema

either `true-labels.csv` or `pred-labels.csv` should have rows like these:

```
id, entities
1, '[{"type": "date", "values": [{"value": "2019-04-21T00:00:00+05:30", "type": "value"}]}]'
2, '[{"text": "6th evening", "type": "time", "values": [{"type": "interval", "value": {"from": "2021-08-06T18:00:00.000-07:00", "to": "2021-08-07T00:00:00.000-07:00"}}]}]'
3, 
4, '[{"text": "67", "type": "number", "values": [{"type": "value", "value": 67}]}]'
```

the `entities` are in `JSON` format. 

exact schema of entity looks like this:

for ordinary `value` types:

```
[
    {
        "type": "entity_type", # date, time, datetime, number, people etc...
        "values": [
            {
                "value": "entity_value", # "2019-04-21T00:00:00+05:30", 42, etc
                "type": "value",
            }
        ]
    }
]
```

for `interval` value type:

```
[
    {
        "type": "entity_type", # date, time, datetime only
        "values": [
            {
                "value": {"from": "...", "to": "..."},
                "type": "interval",
            }
        ]
    }
]
```


Three important things to note:
* we require only entity's `type`, `values` for calculating the `entity_report`, the `body` / `text` or any other key is not required as of now.
* if no-prediction / no-annotation has been made for that particular entity leave it blank, pandas will parse it as `NaN`, accordingly it'll be processed as false negative / false positive.
* Right now, we only support only one `value`, meaning we compare truth and prediction only on the first duckling prediction. Implies our comparisons right now for entities looks likes: `[{}]` vs `[{}]`, in future we'd be supporting `[{}, {}]` vs `[{}, {}, {}, {}, {}]`

## Usage

### Command Line
For using it on command line, simply call the sub-command `entity` like shown
below:

```shell
 eevee entity ./true-labels.csv ./pred-labels.csv
```

```
        FPR       FNR  Mismatch Rate  Support  Positives  Negatives
Entity                                                             
date    1.0  0.142857            0.0        7          7          1
people  0.0  0.333333            0.0        6          6          0
time    1.0  0.125000            0.0        8          8          3
```

### Python module
A common usage pattern for ML modeling is to use entity comparison functions
from `eevee.ord.entity` module.

A demonstration on how to use `date_eq` and `time_eq`:

```python
>>> from eevee.ord.entity.datetime import date_eq, time_eq
>>>
>>> true_date = {'type': 'date', 'values': [{'value': '2019-04-21T00:00:00+05:30', 'type': 'value'}]}
>>> pred_date = {'type': 'date', 'values': [{'value': '2019-04-21T00:00:00+05:30', 'type': 'value'}]}
>>> date_eq(true_date, pred_date)
True

>>> true_time = {'type': 'time', 'values': [{'value': '2019-04-21T09:00:00+05:30', 'type': 'value'}]}
>>> pred_time = {'type': 'time', 'values': [{'value': '2019-04-21T00:00:00+05:30', 'type': 'value'}]}
>>> time_eq(true_time, pred_time)
False
```

