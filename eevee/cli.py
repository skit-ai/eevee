"""
eevee

Usage:
  eevee intent <true-labels> <pred-labels> [--json] [--alias-yaml=<alias_yaml_path>] [--groups-yaml=<groups-yaml_path>] [--breakdown]
  eevee intent layers <true-labels> <pred-labels> --layers-yaml=<layers_yaml_path> [--breakdown]
  eevee asr <true-labels> <pred-labels> [--json]
  eevee entity <true-labels> <pred-labels> [--json] [--breakdown] [--dump]

Options:
  --json                            If true, dump the report in json format for machine
                                    consumption instead of pretty printing.
  --breakdown                       If true, breaksdown the 
                                        * categorical entities for entities (or)
                                        * grouped intents when --alias-yaml  is provided.
  --dump                            If true, dumps the prediction fp, fn, mm errors as csvs.
  --alias-yaml=<alias_yaml_path>    Path to aliasing yaml for intents.
  --groups-yaml=<groups_yaml_path>  Path to intent groups yaml for batched evaluation.

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
from eevee.metrics.classification import intent_layers_report
from eevee.metrics.asr import asr_report
from eevee.metrics.entity import categorical_entity_report, entity_report
from eevee.utils import parse_yaml


def main():
    args = docopt(__doc__, version=__version__)

    if args["intent"]:
        true_labels = pd.read_csv(args["<true-labels>"])
        pred_labels = pd.read_csv(args["<pred-labels>"])

        if args["layers"]:
            breakdown = True if args["--breakdown"] else False
            layers_yaml = args["--layers-yaml"]
            intent_layers = None

            if layers_yaml:
                intent_layers = parse_yaml(layers_yaml)

            output = intent_layers_report(
                true_labels,
                pred_labels,
                intent_layers=intent_layers,
                breakdown=breakdown,
            )
        else:
            breakdown = True if args["--breakdown"] else False
            alias_yaml = args["--alias-yaml"]
            groups_yaml = args["--groups-yaml"]
            return_output_as_dict = False
            intent_aliases = None
            intent_groups = None

            if not groups_yaml and breakdown:
                raise ValueError("--breakdown requires, --groups-yaml along with it.")

            if alias_yaml:
                intent_aliases = parse_yaml(alias_yaml)

            if groups_yaml:
                intent_groups = parse_yaml(groups_yaml)

            if args["--json"]:
                return_output_as_dict = True

            output = intent_report(
                true_labels,
                pred_labels,
                return_output_as_dict=return_output_as_dict,
                intent_aliases=intent_aliases,
                intent_groups=intent_groups,
                breakdown=breakdown,
            )
            
        # output can be str when return_output_as_dict=False, intent_groups is None and breakdown=False
        # output can be dict when return_output_as_dict=True (intent_groups can be present or absent)
            # can be dict of dict, or dict of str, pd.DataFrames
        # output can be pd.DataFrame when return_output_as_dict=False, intent_groups is present and breakdown=False            

        if args["--json"] or args["layers"]:

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
            if groups_yaml is not None and breakdown:
                # output : Dict[str, pd.DataFrame]
                for group_intent, group_intent_metrics_df in output.items():
                    print("\n")
                    print("group :", group_intent)
                    print(group_intent_metrics_df)

            else:
                print(output)

    elif args["asr"]:
        true_labels = pd.read_csv(args["<true-labels>"], usecols=["id", "transcription"])
        pred_labels = pd.read_csv(args["<pred-labels>"], usecols=["id", "utterances"])

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


            # TODO: handle really large reports
            # if isinstance(output, pd.DataFrame):

            #     with pd.option_context(
            #         'display.max_rows', None, 
            #         'display.max_columns', None
            #     ):
            #         print(output)

            # else:
            #     print(output)
