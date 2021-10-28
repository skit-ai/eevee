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
| `number`                   | supported superficially only. `number` and `people` **are not the same**, we still debating on whether to alias all `number` to `people`, vice-versa or not.          |

## Data schema

either `true-labels.csv` or `pred-labels.csv` should have rows like these:

```
id, entities
1,"[{""text"": ""24 अक्टूबर"", ""type"": ""date"", ""score"": 0, ""value"": ""2021-10-24T00:00:00.000+05:30""}]"
2,"[{""text"": ""now"", ""type"": ""datetime"", ""score"": 0, ""value"": ""2021-10-14T13:35:17.354+05:30""}]"
3, []
4,"[{""text"":""21"",""type"":""number"",""score"":1,""value"":21}]"
5,"[{""text"":""Mira Road"",""type"":""location"",""score"":0,""value"":""mira road""}]"
6,"[{""text"":""पांच से सात"",""type"":""datetime"",""score"":0.3,""value"":{""from"":{""grain"":""hour"",""value"":""2021-10-14T05:00:00.000+05:30""},""to"":{""grain"":""hour"",""value"":""2021-10-14T08:00:00.000+05:30""},""type"":""interval""}}]"
```

the `entities` are in `JSON` format. The above ones are just few samples to demonstrate the example.

It is important to note that the csv(s) should contain columns called `id` and `entities`, as the `entities` will be merged on the common `id`. Therefore `id` is expected to be unique.


We expect `type` and `value` to be never `null`. We also expect `value` is of appropriate python-datatype. Eg: `type` of `date`, we expect `value` to be an ISO string like `"2021-10-24T00:00:00.000+05:30"` instead of `24` (an `integer`).


Three important things to note:
* we require only entity's `type`, `value` for calculating the `entity_report`, the `body` / `text` or any other key is not required as of now.
* if no-prediction / no-annotation has been made for that particular entity leave it blank or [], pandas will parse it as `NaN`, accordingly it'll be processed as false negative / false positive.
* Right now, we only support only one `value`, meaning we compare truth and prediction only on the first duckling prediction. Implies our comparisons right now for entities looks likes: `[{}]` vs `[{}]`, in future we'd be supporting `[{}, {}]` vs `[{}, {}, {}, {}, {}]`

## Usage

### Command Line
For using it on command line, simply call the sub-command `entity` like shown
below:

```shell
 eevee entity ./true-labels.csv ./pred-labels.csv
```


```
                  FPR       FNR  Mismatch Rate  Support  Negatives
Entity                                                            
date         0.081612  0.086466       0.377715     6199      18588
detail_kind  0.009233  0.840325       0.013018     5292      19495
duration     0.000444  0.750000       0.000000       36      24751
location     0.004438  0.153505       0.854647     3381      21406
number       0.052483  0.636291       0.286652     2513      22274
time         0.046771  0.098007       0.034070     1204      23583
```

The above numbers are a mix of standard entities like `date`, `time`, `number` etc but also has tagged categorical entities like `location`,  `detail_kind` etc.


only on categorical entities we can get extra `breakdown` (pass `--breakdown` flag), which will report entity mismatches on their categorical values, example:

```
$ eevee entity data/oyo-hi-datetime-truth-entities.csv data/oyo-hi-datetime-class9-entities.csv --breakdown

                                        precision    recall  f1-score  support
_                                        0.604215  0.849747  0.706250     7321
detail_kind/check_in_date                0.952381  0.909091  0.930233       44
detail_kind/check_out_date               0.906250  0.906250  0.906250       32
detail_kind/city                         0.380282  0.551020  0.450000       49
detail_kind/english                      0.250000  0.023622  0.043165      127
...                                           ...       ...       ...      ...
location/yelagiri                        0.000000  0.000000  0.000000        1
location/yelahanka                       0.000000  0.000000  0.000000        1
location/zirakpur                        0.000000  0.000000  0.000000        3
weighted average (excluding no_entity)   0.148283  0.144563  0.144732     8709
```

where `_` represents `NaN` vs `NaN` comparisons. this helps with understanding when it comes to misfiring on no-entities.
Also the last row being `weighted average (excluding no_entity)` helps in giving overall weighted average metrics on these categorical entities.

### Python module


The `entity_report` and `categorical_entity_report` can be imported using 

```python
from eevee.metrics.entity import entity_report, categorical_entity_report
```

they take true and pred dataframes as input, just like the CLI version.


A common usage pattern for ML modeling is to use entity comparison functions
from `eevee.ord.entity` module.

A demonstration on how to use `date_eq` and `time_eq`:

```python
>>> from eevee.ord.entity.datetime import date_eq, time_eq
>>>
>>> true_date = {'type': 'date', 'value': '2019-04-21T00:00:00+05:30'}
>>> pred_date = {'type': 'date', 'value': '2019-04-21T00:00:00+05:30'}
>>> 
>>> date_eq(true_date, pred_date)
True

>>> true_time = {'type': 'time', 'value': '2019-04-21T00:11:00+05:30'}
>>> pred_time = {'type': 'time', 'value': '2019-04-17T00:11:00+05:30'}
>>> 
>>> time_eq(true_time, pred_time)
True

```
