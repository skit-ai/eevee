---
layout: default
title: Speech Recognition
parent: Metrics
nav_order: 3
---

# Speech Recognition
`TODO`

| Metric                               | Description                                                   |
|--------------------------------------+---------------------------------------------------------------|
| WER                                  | Word Error Rate                                               |
| Utterance False Positive Rate (uFPR) | Ratio of cases where non speech utterances were transcribed.  |
| Utterance False Negative Rate (uFNR) | Ratio of cases where utterances where transcribed as silence. |

## Data schema
`TODO`: Note about the `<>` situation in transcription.

## Usage

### Command Line
Use the sub-command `asr` like shown below:

```shell
eevee asr ./data/tagged.transcriptions.csv ./data/predicted.transcriptions.csv

               Value     Support
Metric
WER            0.571429        6
Utterance FPR  0.500000        2
Utterance FNR  0.250000        4
```

### Python module
`TODO`
