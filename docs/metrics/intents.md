---
layout: default
title: Intents
parent: Metrics
nav_order: 1
---

# Intents

Eevee let's you calculate all the important _turn level
metrics_ (precision, recall, f1) for intents. We tag these data points using tog, an internal tool

## Data Schema

We expect the csv(s) to have `id` and `intents` columns. They will be inner-joined on `id`. 

`id` is expected from the user to be unique.
`intents` column should have values whch are of `str` type.

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

