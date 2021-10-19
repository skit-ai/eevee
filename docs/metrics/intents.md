---
layout: default
title: Intents
parent: Metrics
nav_order: 1
---

# Intents
`TODO`

## Data Schema
`TODO`

## Usage

### Command Line
Call the sub-command `intent` like shown below:

```shell
 eevee intent ./true-labels.csv ./pred-labels.csv
```

that takes up the csv's merges them on `id` column, to perform [sklearn's classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report)
on the intents.

there is another feature, `alias`-ing

```
eevee intent ./true-labels.csv ./pred-labels.csv --alias-yaml=data/alias.yaml
```

This helps with aliasing/grouping intents as their respective group:

In the sample file `alias.yaml` under `data` directory, we have intents (from the `true-labels.csv` and `pred-labels.csv`) grouped under:
* smalltalk
* oos (out-of-scope)
* in_scope_intents

but you could name or group the intents according to how you wish. The remaining intents which are not
part of the groups are printed to console as a warning and grouped as `other_intents` by default.

### Python module
`TODO`

