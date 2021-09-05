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

Eevee let's you calculate (work in progress
[here](https://github.com/skit-ai/eevee/pull/7)) all the important _turn level
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
| `datetime`, `date`, `time` | `TODO`          |
| `pattern`                  | `TODO`          |
| `number`                   | `TODO`          |

## Data schema
`TODO`

## Usage

### Command Line
For using it on command line, simply call the sub-command `entity` like shown
below:

```shell
 eevee entity ./true-labels.csv ./pred-labels.csv
     Entity FPR FNR Mismatch Rate Support Positives Negatives
0      date  NA  NA            NA      NA        NA        NA
1  datetime  NA  NA            NA      NA        NA        NA
2    people  NA  NA            NA      NA        NA        NA
3      time  NA  NA            NA      NA        NA        NA
```

### Python module
A common usage pattern for ML modeling is to use entity comparison functions
from `eevee.ord.entity` module.

`TODO`

