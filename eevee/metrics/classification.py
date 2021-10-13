from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import classification_report


def process_yaml(alias_yaml, unique_intents) -> Dict:

    with open(alias_yaml, "r") as fp:
        loaded_yaml = yaml.safe_load(fp)
    
    aliasing_dict = {}

    given_intents = set()

    for aliased_intent, tagged_intents in loaded_yaml.items():

        given_intents.update(tagged_intents)
        for tagged_intent in tagged_intents:
            aliasing_dict[tagged_intent] = aliased_intent

    remaining_intents = unique_intents - given_intents
    if remaining_intents:
        print(f":: [WARN] {remaining_intents} are being considered as `other_intents`.")

    for rem_intent in remaining_intents:
        aliasing_dict[rem_intent] = "other_intents"

    return aliasing_dict


def intent_report(
    true_labels: pd.DataFrame,
    pred_labels: pd.DataFrame,
    output_dict=False,
    breakdown=False,
    alias_yaml=None,
):
    """
    Make an intent report from given label dataframes. We only support single
    intent dataframes as of now.
    TODO:
    - Check type of labels (we are not supporting rich labels right now)
    - Handle 'null' labels.
    """
    df = pd.merge(true_labels, pred_labels, on="id", how="inner")

    if breakdown and alias_yaml is not None:

        unique_intents = set(df["intent_x"].unique()).union(set(df["intent_y"].unique()))
        alias = process_yaml(alias_yaml, unique_intents)
        
        df["intent_x"] = df["intent_x"].replace(alias)
        df["intent_y"] = df["intent_y"].replace(alias)

    return classification_report(
        df["intent_x"], df["intent_y"], output_dict=output_dict, zero_division=0
    )
