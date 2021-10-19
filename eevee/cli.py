"""
eevee

Usage:
  eevee intent <true-labels> <pred-labels> [--json] [--alias-yaml=<alias_yaml_path>]
  eevee asr <true-labels> <pred-labels> [--json]
  eevee entity <true-labels> <pred-labels> [--json] [--breakdown] [--dump]

Options:
  --json                            If true, dump the report in json format for machine
                                    consumption instead of pretty printing.
  --breakdown                       If true, breaksdown the categorical entities for entities.
  --dump                            If true, dumps the prediction fp, fn, mm errors as csvs.
  --alias-yaml=<alias_yaml_path>    Path to aliasing yaml for intents.

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
from eevee.metrics import intent_report
from eevee.metrics.asr import asr_report
from eevee.metrics.entity import categorical_entity_report, entity_report


def main():
    args = docopt(__doc__, version=__version__)

    if args["intent"]:
        true_labels = pd.read_csv(args["<true-labels>"])
        pred_labels = pd.read_csv(args["<pred-labels>"])

        breakdown = True if args["--breakdown"] else False
        alias_yaml = args["--alias-yaml"]

        if args["--json"]:
            output = intent_report(
                true_labels,
                pred_labels,
                output_dict=True,
                breakdown=breakdown,
                alias_yaml=alias_yaml,
            )
            output = json.dumps(output, indent=2)
        else:
            output = intent_report(
                true_labels, pred_labels, alias_yaml=alias_yaml
            )

        print(output)

    elif args["asr"]:
        true_labels = pd.read_csv(args["<true-labels>"])
        pred_labels = pd.read_csv(args["<pred-labels>"])

        output = asr_report(true_labels, pred_labels)

        if args["--json"]:
            print(output.to_json(indent=2))
        else:
            print(output)

    elif args["entity"]:
        true_labels = pd.read_csv(args["<true-labels>"])
        pred_labels = pd.read_csv(args["<pred-labels>"])

        breakdown = True if args["--breakdown"] else False
        dump = True if args["--dump"] else False

        if breakdown:
            output = categorical_entity_report(true_labels, pred_labels)
        else:
            output = entity_report(true_labels, pred_labels, dump)

        if args["--json"]:
            print(output.to_json(indent=2))
        else:
            print(output)
