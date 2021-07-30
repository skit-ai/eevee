# eevee

![](https://img.shields.io/github/v/tag/Vernacular-ai/eevee.svg?style=flat-square) ![GitHub Workflow Status](https://img.shields.io/github/workflow/status/Vernacular-ai/eevee/CI?style=flat-square)

`eevee` is a set of standard evaluation utilities for problems that we work on.
You can use `eevee` both as a python module or as a CLI tool. General CLI
pattern is to point to a dataset and ask for a metric. Few metrics, like Speech
Recognition ones, by default, provide a sliced breakdown report.

`eevee` is supposed to work with files using structures from
[dataframes](https://github.com/Vernacular-ai/dataframes), which contains
standard datatype definitions.

## Installation

For now, you have to install eevee using Github release URLs. The current
version can be installed by using the following:

```bash
pip install https://github.com/Vernacular-ai/eevee/releases/download/0.5.1/eevee-0.5.1-py3-none-any.whl
```

## Usage

Once installed, the most common usage pattern involves passing a reference and
predicted label dataframes and get report either for human viewing, or get a
json for further machine consumption. Here is how you use it for intents:

```bash
eevee intent ./tagged.intent.csv ./predicted.intent.csv
```
