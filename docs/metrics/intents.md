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


### aliasing


there is another feature, `alias`-ing

```
eevee intent ./true-labels.csv ./pred-labels.csv --alias-yaml=assets/alias.yaml
```

alias-yaml, helps with situations where there are different intents which are all just the same:

intents like:
```yaml
  -  _confirm_browse_
  -  _confirm_wifi_
  -  _confirm_power_indicator_
  -  _confirm_reconfirm_pincode_
  -  _confirm_next_to_device_
```

all are just representing the smalltalk intent, `_confirm_`, therefore one could replace them all with `_confirm_`. this is what the `alias.yaml` does.

alias-yaml helps replacing intents with what their mother/actual intent you want it to be. this acts as a preprocessing step.


example of an `alias.yaml`:
```yaml
_confirm_:
  -  _confirm_
  -  _flickering_
  -  _confirm_reconfirm_pincode
  -  confirm_new_connection
  -  _confirm_browse_
  -  _confirm_wifi_
  -  _confirm_power_indicator_
  -  _confirm_reconfirm_pincode_
  -  _confirm_next_to_device_

_cancel_:
  -  _cancel_
  -  _cancel_lights_steady_
  -  _cannot_
  -  _cancel_device_switched_on_
  -  _cancel_browse_
```

where `_confirm_` and `_cancel_` replaces all the intnets mentioned below them in the list, in both ground-truth and predictions.


### grouping

```
eevee intent ./true-labels.csv ./pred-labels.csv --groups-yaml=assets/groups.yaml
```

This helps with grouping intents as their respective group-name:

In the sample file `groups.yaml` under `assets` directory, we have intents (from the `true-labels.csv` and `pred-labels.csv`) grouped under:
* smalltalk_intents
* critical_intents
* oos_intents (out-of-scope)

but you could name or group the intents according to how you wish. The remaining intents which are not
part of the groups are grouped as `in_scope` by default. This returns the `weighted_average` of [sklearn's precision_recall_fscore_support](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support)

Further granular analysis on grouping is also possible. where each group has its own [sklearn's classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report) using this:

```
eevee intent ./true-labels.csv ./pred-labels.csv --groups-yaml=assets/groups.yaml --breakdown
```

### layering

```
eevee intent layers ./true-labels.csv ./pred-labels.csv --layers-yaml=assets/layers.yaml
```

layering is intended to know:
* intent & kinda group level information on relevant/specific intents.

This is mainly made to mitigate the acoustic_oos & lexical_oos differences in ground-truth, but predictions can be anything but we want them to be `_oos_`. so yeah.

There is a the sample file `layers.yaml` under `assets` directory, which we recommend you to use.

Further granular analysis on layering is also possible. where each group has its own [sklearn's classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report) using this:

```
eevee intent layers ./true-labels.csv ./pred-labels.csv --layers-yaml=assets/layers.yaml --breakdown
```


## JSON support

All the above mentioned commands use cases, have additional `--json` flag which will be given out in stdout and can be parsed
using tools like `jq`.

### Python module

```python
>>> import pandas as pd
>>> from pprint import pprint
>>> from eevee.metrics.classification import intent_report
:: stanza not found
>>> 
>>> true_df = pd.read_csv("data/labels_13_2071.csv")
>>> pred_df = pd.read_csv("data/tagged_data_13_2071.csv")
>>> 
>>> all_intents_classification_report = intent_report(true_df, pred_df)
>>> print(all_intents_classification_report)
                                 precision    recall  f1-score   support

                              _       0.00      0.00      0.00         0
                       _cancel_       0.82      0.90      0.86        80
                _cancel_browse_       0.00      0.00      0.00         6
                            ...
                 other_language       0.00      0.00      0.00         1

                       accuracy                           0.37       967
                      macro avg       0.21      0.19      0.18       967
                   weighted avg       0.30      0.37      0.33       967

>>> all_intents_classification_report_dict = intent_report(true_df, pred_df, return_output_as_dict=True)
>>> all_intents_classification_report
{
    '_': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}, 
    '_cancel_': {'precision': 0.8181818181818182, 'recall': 0.9, 'f1-score': 0.8571428571428572, 'support': 80}, 
    '_cancel_browse_': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 6}, 
    '_cancel_internet_connected_': {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 1}, 
    ...
    'weighted avg': {'precision': 0.3037583323260676, 'recall': 0.37228541882109617, 'f1-score': 0.33185064213224263, 'support': 967}}

>>> # aliasing intents
>>> aliased_intents = {
...   "_confirm_": [
...     "_confirm_", 
...     "_flickering_", 
...     "_confirm_reconfirm_pincode", 
        ...
...   ], 
...   "_cancel_": [
...     "_cancel_", 
...     "_cancel_lights_steady_", 
...     "_cannot_", 
        ...
...   ], 
      ...
... }
>>> aliased_classification_report = intent_report(true_df, pred_df, intent_aliases=aliased_intents)
>>> print(aliased_classification_report)
                                 precision    recall  f1-score   support

                              _       0.00      0.00      0.00         0
                       _cancel_       0.92      0.87      0.90        94
                      _confirm_       0.89      0.92      0.90       372
                     _greeting_       0.71      1.00      0.83         5
                            ...
                 other_language       0.00      0.00      0.00         1

                       accuracy                           0.46       967
                      macro avg       0.39      0.36      0.35       967
                   weighted avg       0.46      0.46      0.46       967


>>> # grouping intents
>>> grouped_intents = {
...   "oos_intents": [
...     "acoustic_oos", 
...     "lexical_oos"
...   ], 
...   "smalltalk_intents": [
...     "_confirm_", 
...     "_cancel_", 
...     "_repeat_", 
...     "_what_", 
...     "_greeting_", 
...     "request_agent"
...   ]
... }
>>> grouped_weighted_average_metrics = intent_report(true_df, pred_df, intent_groups=grouped_intents)
>>> print(grouped_weighted_average_metrics)
                   precision    recall  f1-score  support
group                                                    
oos_intents         0.000000  0.000000  0.000000        0
smalltalk_intents   0.723654  0.915119  0.807068      377
in_scope            0.035452  0.025424  0.028195      590


>>> grouped_weighted_average_metrics = intent_report(true_df, pred_df, intent_groups=grouped_intents, breakdown=True)
>>> pprint(grouped_weighted_average_metrics)
{
    'in_scope':                                  
                                        precision    recall  f1-score  support
        _cancel_wifi_ID_connected_        0.000000  0.000000  0.000000        3
        _request_agent_                   1.000000  0.500000  0.666667        6
        _confirm_next_to_device_          0.000000  0.000000  0.000000       28
        audio_silent                      0.000000  0.000000  0.000000      116
        _confirm_switched_on_             0.000000  0.000000  0.000000       40
        _confirm_power_indicator_         0.000000  0.000000  0.000000        4
        inform_name                       0.000000  0.000000  0.000000        4
        _cancel_internet_connected_       0.000000  0.000000  0.000000        1
        _inform_address_                  1.000000  1.000000  1.000000        4
        _cancel_browse_                   0.000000  0.000000  0.000000        6
        audio_speech_volume               0.000000  0.000000  0.000000        2
        other_language                    0.000000  0.000000  0.000000        1
        _wait_                            0.000000  0.000000  0.000000        2
        _cancel_next_to_device_           0.000000  0.000000  0.000000        0
        background_noise                  0.000000  0.000000  0.000000      272
        _                                 0.000000  0.000000  0.000000        0
        _confirm_new_connection_          0.000000  0.000000  0.000000        2
        audio_channel_noise_hold          0.000000  0.000000  0.000000        2
        _inform_old_customer_             1.000000  0.600000  0.750000        5
        audio_channel_noise               0.000000  0.000000  0.000000        1
        _confirm_wifi_ID_connected_       0.000000  0.000000  0.000000        5
        _cancel_switch_on_device_         0.000000  0.000000  0.000000        3
        internet_not_working              0.000000  0.000000  0.000000        0
        _cancel_lights_steady_            0.000000  0.000000  0.000000        1
        background_speech                 0.000000  0.000000  0.000000       49
        _confirm_browse_                  0.000000  0.000000  0.000000        1
        _hathway_plans_                   1.000000  0.500000  0.666667        2
        _ood_                             0.000000  0.000000  0.000000        0
        _inform_residential_connection_   1.000000  0.500000  0.666667        2
        _oos_                             0.333333  0.400000  0.363636        5
        _internet_not_working_            0.000000  0.000000  0.000000        3
        _where_did_you_know_              0.250000  1.000000  0.400000        1
        _inform_name_                     0.000000  0.000000  0.000000        0
        audio_speech_unclear              0.000000  0.000000  0.000000       19
        micro avg                         0.030801  0.025424  0.027855      590
        macro avg                         0.164216  0.132353  0.132754      590
        weighted avg                      0.035452  0.025424  0.028195      590,
 'oos_intents':               
                        precision  recall  f1-score  support
        acoustic_oos        0.0     0.0       0.0        0
        lexical_oos         0.0     0.0       0.0        0
        micro avg           0.0     0.0       0.0        0
        macro avg           0.0     0.0       0.0        0
        weighted avg        0.0     0.0       0.0        0,
 'smalltalk_intents':                
                        precision    recall  f1-score  support
        _confirm_       0.697917  0.917808  0.792899      292
        _cancel_        0.818182  0.900000  0.857143       80
        _repeat_        0.000000  0.000000  0.000000        0
        _what_          0.000000  0.000000  0.000000        0
        _greeting_      0.714286  1.000000  0.833333        5
        request_agent   0.000000  0.000000  0.000000        0
        micro avg       0.718750  0.915119  0.805134      377
        macro avg       0.371731  0.469635  0.413896      377
        weighted avg    0.723654  0.915119  0.807068      377
}


>>> intent_layers = {
...                 'intent_x': {
...                     'acoustic_oos': [
                            'audio_channel_noise', 'audio_channel_noise_hold', 
                            'audio_speech_unclear','audio_speech_volume', 
                            'audio_silent', 'background_noise', 
                            'background_speech', 'other_language', '_'
                          ], 
...                     'lexical_oos': ['partial', 'ood', '_oos_']
...                 }, 
...                 'intent_y': {
...                     'oos': ['oos', '_']
...                 }
...             }
>>> 
>>> intent_layers_report(true_df, pred_df, intent_layers=intent_layers)
                    precision    recall  f1-score  support
layer                                                     
layer-acoustic_oos   0.934685  0.898268  0.916115      462
layer-lexical_oos    0.000000  0.000000  0.000000        5
layer-oos            0.934685  0.888651  0.911087      467

>>> out = intent_layers_report(true_df, pred_df, intent_layers=intent_layers, breakdown=True)
>>> pprint(out)
{
  'layer-acoustic_oos':               
              precision    recall  f1-score  support
acoustic_oos   0.934685  0.898268  0.916115      462
micro avg      0.934685  0.898268  0.916115      462
macro avg      0.934685  0.898268  0.916115      462
weighted avg   0.934685  0.898268  0.916115      462,

 'layer-lexical_oos':               
              precision  recall  f1-score  support
lexical_oos         0.0     0.0       0.0        5
micro avg           0.0     0.0       0.0        5
macro avg           0.0     0.0       0.0        5
weighted avg        0.0     0.0       0.0        5,

 'layer-oos':               
              precision    recall  f1-score  support
oos            0.934685  0.888651  0.911087      467
micro avg      0.934685  0.888651  0.911087      467
macro avg      0.934685  0.888651  0.911087      467
weighted avg   0.934685  0.888651  0.911087      467
}
```

