---
layout: default
title: Data Structures
nav_order: 2
---

# Data Structures

Eevee works with CSV label dataframes with items as per [these
definitions](https://github.com/skit-ai/dataframes/blob/master/protos/labels.proto).
Since label dataframes have `id` for referring back to the data, we just focus
on labels in this tool. These labels can be true labels of various kind or
coming from predictions of different models.

This page documents a few notes about the label representation. Specific details
are in the pages for different kinds of metrics [here](./metrics).

## Serialization
Each row in the label dataframe CSV is of one of the types defined
[here](https://github.com/skit-ai/dataframes/blob/master/protos/labels.proto).
In cases where the field type primitive, we serialize items in JSON. In Python,
this looks like the following:

```python
import pandas as pd

# Assuming each item in `items` is a list of entities
rows = [{"id": i, "entities": json.dumps(it)} for i, it in enumerate(items)]

pd.DataFrame(rows).to_csv("./predictions.csv", index=False)
```

The following is how correctly serialized structure looks like in a labels CSV:

```
"[[{""am_score"": -278.4794, ""confidence"": 0.9739978, ""lm_score"": 13.827044, ""transcript"": ""no""}]]"
```

If you skip JSON dumping, tools like pandas might still serialize like following:
```
"[[{'am_score': -278.4794, 'confidence': 0.9739978, 'lm_score': 13.827044, 'transcript': 'no'}]]"
```

But this won't be read back in `eevee` and you will get a `JSONDecodeError`

```
JSONDecodeError: Expecting property name enclosed in double quotes: line 1 column 4 (char 3)
```
