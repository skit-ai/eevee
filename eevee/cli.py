"""
eevee

Usage:
  eevee intent <true-labels> <pred-labels> [--json]
  eevee entity <true-labels> <pred-labels> [--json]

Options:
  --json                    If true, dump the report in json format for machine
                            consumption instead of pretty printing.

Arguments:
  <true-labels>             Path to file with true labels with our dataframe
                            definitions.
  <pred-labels>             Path to file with predicted labels with our
                            dataframe definitions.
"""

import json

import pandas as pd
from docopt import docopt

from eevee import __version__
from eevee.metrics import entity_report, multi_class_classification_report


def main():
    args = docopt(__doc__, version=__version__)

    if args["intent"]:
        true_labels = pd.read_csv(args["<true-labels>"])
        pred_labels = pd.read_csv(args["<pred-labels>"])

        if args["--json"]:
            output = multi_class_classification_report(
                true_labels, pred_labels, output_dict=True
            )
            output = json.dumps(output, indent=2)
        else:
            output = multi_class_classification_report(true_labels, pred_labels)

        print(output)

    elif args["entity"]:
        true_labels = pd.read_csv(args["<true-labels>"])
        pred_labels = pd.read_csv(args["<pred-labels>"])

        output = entity_report(true_labels, pred_labels)

        if args["--json"]:
            output = output.to_json(indent=2)

        print(output)
