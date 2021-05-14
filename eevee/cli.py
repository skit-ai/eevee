"""
eevee

Usage:
  eevee slots <dataset> [--json]

Options:
  --json                    If true, dump the report in json format instead of
                            pretty printing.

Arguments:
  <dataset>                 Path to the dataset file with annotated items
"""

from docopt import docopt

from eevee import __version__


def main():
    args = docopt(__doc__, version=__version__)
