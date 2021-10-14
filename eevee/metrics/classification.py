from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd
import yaml
from sklearn.metrics import classification_report


def map_intents_to_their_alias(unique_intents : Set[str], alias_yaml: str) -> Dict:

    with open(alias_yaml, "r") as fp:
        loaded_yaml : Dict[str, List[str]] = yaml.safe_load(fp)
    
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

    if alias_yaml is not None:

        unique_intents = set(df["intent_x"].unique()).union(set(df["intent_y"].unique()))
        alias = map_intents_to_their_alias(unique_intents, alias_yaml)

        df["intent_x"] = df["intent_x"].replace(alias)
        df["intent_y"] = df["intent_y"].replace(alias)

    return classification_report(
        df["intent_x"], df["intent_y"], output_dict=output_dict, zero_division=0
    )
