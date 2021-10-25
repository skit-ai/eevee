"""
eevee

Usage:
  eevee intent <true-labels> <pred-labels> [--json] [--alias-yaml=<alias_yaml_path>] [--breakdown]
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
from eevee.utils import parse_yaml


def main():
    args = docopt(__doc__, version=__version__)

    if args["intent"]:
        true_labels = pd.read_csv(args["<true-labels>"])
        pred_labels = pd.read_csv(args["<pred-labels>"])

        breakdown = True if args["--breakdown"] else False
        alias_yaml = args["--alias-yaml"]
        return_output_as_dict = False
        intent_groups = None

        if alias_yaml:
            intent_groups = parse_yaml(alias_yaml)

        if args["--json"] or breakdown:
            return_output_as_dict = True

        output = intent_report(
            true_labels,
            pred_labels,
            return_output_as_dict=return_output_as_dict,
            intent_groups=intent_groups,
            breakdown=breakdown,
        )
            
        # output can be str when return_output_as_dict=False, intent_groups is None and breakdown=False
        # output can be dict when return_output_as_dict=True (intent_groups can be present or absent)
            # can be dict of dict, or dict of str, pd.DataFrames
        # output can be pd.DataFrame when return_output_as_dict=False, intent_groups is present and breakdown=False            

        if args["--json"]:

            # grouping is present and breakdown is asked for
            # output : Dict[str, pd.DataFrame] -> output : Dict[str, Dict[str, Dict]]
            if alias_yaml is not None and isinstance(output, dict):
                for alias_intent, group_intent_metrics_df in output.items():
                    output[alias_intent] = group_intent_metrics_df.to_dict("index")

            # grouping is present but no breakdown
            elif isinstance(output, pd.DataFrame):
                output = output.to_dict("index")

            output = json.dumps(output, indent=2)

            print(output)

        else:

            # when alias.yaml is given, and one expects a breakdown of group's classification
            # report as per each group
            if alias_yaml is not None and breakdown:
                # output : Dict[str, pd.DataFrame]
                for alias_intent, group_intent_metrics_df in output.items():
                    print("\n")
                    print("group :", alias_intent)
                    print(group_intent_metrics_df)

            else:
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
