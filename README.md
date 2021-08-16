# eevee

![](https://img.shields.io/github/v/tag/Vernacular-ai/eevee.svg?style=flat-square) ![GitHub Workflow Status](https://img.shields.io/github/workflow/status/Vernacular-ai/eevee/CI?style=flat-square)

`eevee` is a set of standard evaluation utilities for problems that we work on.
You can use `eevee` both as a python module or as a CLI tool. It works on data
files with label structures from
[dataframes](https://github.com/Vernacular-ai/dataframes) that has standard
datatype definitions. See `./data` directory for example files.

## Installation

For now, you have to install eevee using Github release URLs. The current
version can be installed by using the following:

```bash
pip install https://github.com/Vernacular-ai/eevee/releases/download/0.5.2/eevee-0.5.2-py3-none-any.whl
```

## Usage

Once installed, the most common usage pattern involves passing a reference and
predicted label dataframes and get report either for human viewing, or get a
json for further machine consumption. Here is how you use it for intents:

```bash
eevee intent ./tagged.intent.csv ./predicted.intent.csv
```

Similarly, for WER report you can do this:

```bash
eevee asr ./data/tagged.transcriptions.csv ./data/predicted.transcriptions.csv --json

# {
# "Value":{
#     "WER":0.4761904762
#   }
# }
```

There are a few advanced unexposed metrics related to ASR. Since they are still
work in progress, we have kept a few dependencies from there as _extras_. If you
need those, you should install the package in development mode and do `poetry
install -E asr`. Then follow the scripts in `./scripts`.