---
layout: default
title: Speech Recognition
parent: Metrics
nav_order: 3
---

# Speech Recognition

`TODO`

| Metric | Description |
|--------------------------------------+---------------------------------------------------------------|
| WER | Word Error Rate |
| Utterance False Positive Rate (uFPR) | Ratio of cases where non speech utterances were transcribed. |
| Utterance False Negative Rate (uFNR) | Ratio of cases where utterances where transcribed as silence. |
| SER | Sentence Error Rate |

## Data schema

We expect,

- tagged.transcriptions.csv to have columns called `id` and `transcription`, where `transcription` can have only one string as value for each row, if not present leave it empty as it is, it'll get parsed as `NaN`.
- predicted.transcriptions.csv to have columns called `id` and `utterances`, where **each value** in the `utterances` column looks like this:

```
'[[
    {"confidence": 0.94847125, "transcript": "iya iya iya iya iya"},
    {"confidence": 0.9672866, "transcript": "iya iya iya iya"},
    {"confidence": 0.8149829, "transcript": "iya iya iya iya iya iya"}
]]'
```

as you might have noticed it is expected to be in `JSON` format. each `transcript` represents each alternative from the ASR, and `confidence` represents ASR's confidence for that particular alternative. If no such `utterances` present for that particular `id`, leave it as `'[]'` (`json.dumps` of empty list `[]`)

Note: Please remove `transcription` column from predicted.transcriptions.csv (if it exists) before using `eevee`.

## Usage

### Command Line

Use the sub-command `asr` like shown below:

```shell
eevee asr ./data/tagged.transcriptions.csv ./data/predicted.transcriptions.csv
```

```
                  Value  Support
Metric                          
WER            0.571429        6
Utterance FPR  0.500000        2
Utterance FNR  0.250000        4
SER            0.666667        6
```

For users who want utternace level metrics, add the "--dump" flag like:

```shell
eevee asr ./data/tagged.transcriptions.csv ./data/predicted.transcriptions.csv --dump
```

This will add a csv file called **predicted.transcriptions-dump.csv**. The filename is based on the prediction filename given by the user

### Python module

```python
>>> import pandas as pd
>>> from eevee.metrics.asr import asr_report
>>> 
>>> true_df = pd.read_csv("data/tagged.transcriptions.csv", usecols=["id", "transcription"])
>>> pred_df = pd.read_csv("data/predicted.transcriptions.csv", usecols=["id", "utterances"])
>>> 
>>> asr_report(true_df, pred_df)
                  Value  Support
Metric                          
WER            0.571429        6
Utterance FPR  0.500000        2
Utterance FNR  0.250000        4
SER            0.666667        6
```
