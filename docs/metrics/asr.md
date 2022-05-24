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
| Min 3 WER | The minimum Word Error Rate when considering the first three alternatives only |
| Min WER | The minimum Word Error Rate out of all the alternatives |
| Short Utterance WER | WER of utterance with ground truth length of 1 or 2 words |
| Long Utterance WER | WER of utterances with at least 3 words in ground truth |

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
WER                  0.571429        6
Utterance FPR        0.500000        2
Utterance FNR        0.250000        4
SER                  0.666667        6
Min 3 WER            0.571429        6
Min WER              0.571429        6
Short Utterance WER  0.000000        1
Long Utterance WER   0.809524        3

```

For users who want utterance level metrics or edit operations, add the "--dump" flag like:

```shell
eevee asr ./data/tagged.transcriptions.csv ./data/predicted.transcriptions.csv --dump
```

This will add two csv files

- **predicted.transcriptions-dump.csv** : File containing utterance level metrics
- **predicted.transcriptions-ops.csv** : File containing dataset level edit operations.

The filename is based on the prediction filename given by the user

----                                                                                                                                                                                     

For users who want ASR metrics reported separately on `noisy` and `non-noisy` subsets of audios, 
use the "--noisy" flag like:

```shell
eevee asr ./data/tagged.transcriptions.csv ./data/predicted.transcriptions.csv --noisy
```

Results are in two DataFrames - for each of `noisy` and `non-noisy` subsets, in order. An important note 
here, is that the transcriptions in `tagged.transcriptions.csv` are expected to contain informational tags,
like - `<audio_silent>`, `<inaudible>`, etc - which are normally removed when not using the "--noisy" flag.

### Python module

```python
>>> import pandas as pd
>>> from eevee.metrics.asr import asr_report
>>>
>>> true_df = pd.read_csv("data/tagged.transcriptions.csv", usecols=["id", "transcription"])
>>> pred_df = pd.read_csv("data/predicted.transcriptions.csv", usecols=["id", "utterances"])
>>>
>>> asr_report(true_df, pred_df)
                    Value   Support
Metric
WER                  0.571429        6
Utterance FPR        0.500000        2
Utterance FNR        0.250000        4
SER                  0.666667        6
Min 3 WER            0.571429        6
Min WER              0.571429        6
Short Utterance WER  0.000000        1
Long Utterance WER   0.809524        3

```
