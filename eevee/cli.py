"""
eevee

Usage:
  eevee intent <true-labels> <pred-labels> [--json] [--alias-yaml=<alias_yaml_path>] [--groups-yaml=<groups-yaml_path>] [--breakdown]
  eevee intent layers <true-labels> <pred-labels> --layers-yaml=<layers_yaml_path> [--breakdown] [--json]
  eevee asr <true-labels> <pred-labels> [--json] [--dump] [--noisy]
  eevee entity <true-labels> <pred-labels> [--json] [--breakdown] [--dump]

Options:
  --json                            If true, dump the report in json format for machine
                                    consumption instead of pretty printing.
  --breakdown                       If true, breaksdown the 
                                        * categorical entities for entities (or)
                                        * grouped intents when --alias-yaml  is provided.
  --dump                            If true, 
                                    * dumps the prediction fp, fn, mm errors as csvs.
                                    * ASR metrics on an utterance level
  --noisy                           If true,
                                        * splits the dataset into noisy and non-noisy subsets
                                          and returns results for both separately
                                        * expects uncleaned asr alternatives, with informational tags
  --alias-yaml=<alias_yaml_path>    Path to aliasing yaml for intents.
  --groups-yaml=<groups_yaml_path>  Path to intent groups yaml for batched evaluation.
  --layers-yaml=<layers_yaml_path>  Path to intent layers yaml for evaluation of sub layers.

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
from eevee.metrics.asr import asr_report, process_noise_info
from eevee.metrics.entity import categorical_entity_report, entity_report
from eevee.utils import parse_yaml


def main():
    args = docopt(__doc__, version=__version__)

    if args["intent"]:
        true_labels = pd.read_csv(args["<true-labels>"])
        pred_labels = pd.read_csv(args["<pred-labels>"])
        breakdown = True if args["--breakdown"] else False
        alias_yaml = args["--alias-yaml"]
        groups_yaml = args["--groups-yaml"]
        layers_yaml = args["--layers-yaml"]
        return_output_as_dict = False
        intent_aliases = None
        intent_groups = None
        intent_layers = None

        if args["layers"]:

            if layers_yaml:
                intent_layers = parse_yaml(layers_yaml)

            output = intent_layers_report(
                true_labels,
                pred_labels,
                intent_layers=intent_layers,
                breakdown=breakdown,
            )
        else:

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

        if args["--json"]:

            # grouping is present and breakdown is asked for
            # output : Dict[str, pd.DataFrame] -> output : Dict[str, Dict[str, Dict]]
            if (groups_yaml is not None or args["layers"]) and isinstance(output, dict):
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
            if (groups_yaml is not None or args["layers"]) and breakdown:
                # output : Dict[str, pd.DataFrame]
                for group_intent, group_intent_metrics_df in output.items():
                    print("\n")
                    print("group :", group_intent)
                    print(group_intent_metrics_df)

            else:
                print(output)

    elif args["asr"]:
        true_labels = pd.read_csv(
            args["<true-labels>"], usecols=["id", "transcription"]
        )
        pred_labels = pd.read_csv(args["<pred-labels>"], usecols=["id", "utterances"])

        dump = True if args["--dump"] else False

        if args["--noisy"]:

            noisy_dict, not_noisy_dict = process_noise_info(true_labels, pred_labels)
            input_dict = {"noisy": noisy_dict, "not-noisy": not_noisy_dict}
            output_dict = {}

            if dump:
                for key, subset in input_dict.items():
                    output, breakdown, ops = asr_report(
                        subset["true"], subset["pred"], dump
                    )
                    output_dict[key] = output
                    breakdown.to_csv(
                        f'{key}-{args["<pred-labels>"].replace(".csv", "")}-dump.csv',
                        index=False,
                    )
                    ops.to_csv(
                        f'{key}-{args["<pred-labels>"].replace(".csv", "")}-ops.csv',
                        index=False,
                    )
            else:
                for key, subset in input_dict.items():
                    output = asr_report(subset["true"], subset["pred"])
                    output_dict[key] = output

            if args["--json"]:
                for key, output in output_dict.items():
                    print(key)
                    print(output.to_json(indent=2))
            else:
                for key, output in output_dict.items():
                    print(key)
                    print(output)

        else:
            if dump:
                output, breakdown, ops = asr_report(true_labels, pred_labels, dump)
                breakdown.to_csv(
                    f'{args["<pred-labels>"].replace(".csv", "")}-dump.csv', index=False
                )
                ops.to_csv(
                    f'{args["<pred-labels>"].replace(".csv", "")}-ops.csv', index=False
                )
            else:
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
