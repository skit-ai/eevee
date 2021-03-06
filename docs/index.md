---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
title: Home
nav_order: 1
---

# Eevee
![](https://img.shields.io/github/v/tag/skit-ai/eevee.svg?style=flat-square) ![GitHub Workflow Status](https://img.shields.io/github/workflow/status/skit-ai/eevee/CI?style=flat-square)

`eevee` is a set of standard evaluation utilities for problems that we work on
at [Skit](https://skit.ai).

You can use `eevee` both as a python module or as a CLI tool. It works on data
files with label structures from
[dataframes](https://github.com/skit-ai/dataframes) that has standard datatype
definitions. See `./data` directory for example files.

## Installation

For now, you have to install eevee using Github release URLs. The current
version can be installed by using the following:

```bash
pip install https://github.com/skit-ai/eevee/releases/download/1.3.0/eevee-1.3.0-py3-none-any.whl
```

With `poetry`

```bash
poetry add "git+https://github.com/skit-ai/eevee.git@1.3.0"
```

or directly add this in `pyproject.toml`
```toml
[tool.poetry.dependencies]
eevee = {git = "https://github.com/skit-ai/eevee.git", rev = "1.3.0"}
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
eevee asr ./data/tagged.transcriptions.csv ./data/predicted.transcriptions.csv
```

```
               Value     Support
Metric
WER            0.571429        6
Utterance FPR  0.500000        2
Utterance FNR  0.250000        4
```

There are a few advanced unexposed metrics related to ASR. Since they are still
work in progress, we have kept a few dependencies from there as _extras_. If you
need those, you should install the package in development mode and do `poetry
install -E asr`. Then follow the scripts in `./scripts`.
