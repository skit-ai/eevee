# eevee

![](https://img.shields.io/github/v/tag/Vernacular-ai/eevee.svg?style=flat-square)

`eevee` is a set of standard evaluation utilities for problems that we work on.
The key abstractions are datasets, slices, and metrics.

You can use `eevee` both as a python module or as a CLI tool. General CLI
pattern is to point to a dataset and ask for a metric. Few metrics, like Speech
Recognition ones, by default, provide a sliced breakdown report.

We work with many dataset formats, but the most common one is
[tog's](https://github.com/Vernacular-ai/tog-cli/).
