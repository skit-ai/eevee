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

We expect the csv(s) to have `id` and `intent` columns. They will be inner-joined on `id`. 

`id` is expected from the user to be unique.
`intent` column should have values whch are of `str` type.

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

but you could name or group the intents according to how you wish. The remaining intents which are not
part of the groups are grouped as `in_scope` by default. This returns the `weighted_average` of [sklearn's precision_recall_fscore_support](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support)

Further granular analysis on aliasing/grouping is also possible. where each group has its own [sklearn's classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report) using this:

```
eevee intent ./true-labels.csv ./pred-labels.csv --alias-yaml=data/alias.yaml --breakdown
```

## JSON support

All the above mentioned commands use cases, have additional `--json` flag which will be given out in stdout and can be parsed
using tools like `jq`.

### Python module

```python
>>> import pandas as pd
>>> from eevee.metrics.classification import intent_report
:: stanza not found
>>> 
>>> true_df = pd.read_csv("data/reddoorz.tagged_data.csv")
>>> pred_df = pd.read_csv("data/reddoorz.labels.csv")
>>> 
>>> all_intents_classification_report = intent_report(true_df, pred_df)
>>> print(all_intents_classification_report)
                                 precision    recall  f1-score   support

                       _cancel_       1.00      1.00      1.00       100
                      _confirm_       1.00      1.00      1.00       161
                     _greeting_       1.00      1.00      1.00       134
                          _oos_       1.00      1.00      1.00        97
                            ...
                  refund_status       1.00      1.00      1.00         8
                  request_agent       1.00      1.00      1.00        28
                short_utterance       1.00      1.00      1.00        23

                       accuracy                           1.00      1035
                      macro avg       1.00      1.00      1.00      1035
                   weighted avg       1.00      1.00      1.00      1035

>>> all_intents_classification_report_dict = intent_report(true_df, pred_df, output_dict=True)
>>> print(all_intents_classification_report_dict)
{
    '_cancel_': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 100}, 
    '_confirm_': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 161},
    ...
    'weighted avg': {'precision': 1.0, 'recall': 1.0, 'f1-score': 1.0, 'support': 1035}
}


>>> aliased_intents = {
    'smalltalk': ['_greeting_', '_repeat_', 'what', 'hmm', '_confirm_', '_cancel_'], 
    'oos': ['_oos_', '_ood_', 'audio_silent_background_talking', 'audio_silent', 'broken_voice', 'short_utterance', 'audio_noisy']
    }

>>> grouped_weighted_average_metrics = intent_report(true_df, pred_df, intent_groups=aliased_intents)
>>> grouped_weighted_average_metrics
           precision  recall  f1-score  support
group                                          
smalltalk        1.0     1.0       1.0      398
oos              1.0     1.0       1.0      274
in_scope         1.0     1.0       1.0      363

>>> grouped_intents_classification_report = intent_report(true_df, pred_df, intent_groups=aliased_intents, breakdown=True)
>>> grouped_intents_classification_report
{
    'smalltalk':               
                    precision    recall  f1-score  support
        _greeting_     1.000000  1.000000  1.000000      134
        _repeat_       1.000000  1.000000  1.000000        3
        what           0.000000  0.000000  0.000000        0
        hmm            0.000000  0.000000  0.000000        0
        _confirm_      1.000000  1.000000  1.000000      161
        _cancel_       1.000000  1.000000  1.000000      100
        micro avg      1.000000  1.000000  1.000000      398
        macro avg      0.666667  0.666667  0.666667      398
        weighted avg   1.000000  1.000000  1.000000      398,
    
    'oos':
                                        precision    recall  f1-score  support
        _oos_                             1.000000  1.000000  1.000000       97
        _ood_                             0.000000  0.000000  0.000000        0
        audio_silent_background_talking   1.000000  1.000000  1.000000       66
        audio_silent                      1.000000  1.000000  1.000000       32
        broken_voice                      1.000000  1.000000  1.000000        1
        short_utterance                   1.000000  1.000000  1.000000       23
        audio_noisy                       1.000000  1.000000  1.000000       55
        micro avg                         1.000000  1.000000  1.000000      274
        macro avg                         0.857143  0.857143  0.857143      274
        weighted avg                      1.000000  1.000000  1.000000      274,

    'in_scope':                 
                                precision  recall  f1-score  support
        booking_cancellation        1.0     1.0       1.0       22
        booking_status              1.0     1.0       1.0       33
        booking_modification        1.0     1.0       1.0       14
        new_booking                 1.0     1.0       1.0      174
        early_checkin               1.0     1.0       1.0       12
        cancel_charges              1.0     1.0       1.0        9
        request_agent               1.0     1.0       1.0       28
        checkin_procedure           1.0     1.0       1.0       29
        location_information        1.0     1.0       1.0       14
        checkin_time                1.0     1.0       1.0        4
        price_enquiry               1.0     1.0       1.0       14
        refund_status               1.0     1.0       1.0        8
        late_checkout               1.0     1.0       1.0        2
        micro avg                   1.0     1.0       1.0      363
        macro avg                   1.0     1.0       1.0      363
        weighted avg                1.0     1.0       1.0      363
}
```

