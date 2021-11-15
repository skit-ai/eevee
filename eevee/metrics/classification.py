from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support


def intent_report(
    true_labels: pd.DataFrame,
    pred_labels: pd.DataFrame,
    return_output_as_dict : bool=False,
    intent_aliases: Optional[Dict[str, List[str]]] = None,
    intent_groups: Optional[Dict[str, List[str]]]=None,
    breakdown=False,
):

    df = pd.merge(true_labels, pred_labels, on="id", how="inner")

    # for cases where we are seeing NaN values popping up.
    df[['intent_x', 'intent_y']] = df[['intent_x', 'intent_y']].fillna(value="_")

    # aliasing intents
    if intent_aliases is not None:
        alias_dict = {intent: alias for alias, intent_list in intent_aliases.items() for intent in intent_list}
        for col in ["intent_x", "intent_y"]:
            df[col] = df[col].apply(lambda intent: alias_dict.get(intent, intent))

    # vanilla case, where just ordinary classification report is required.
    # it goes out as str or dict, depending on `return_output_as_dict`
    if intent_groups is None and not breakdown:

        return classification_report(
        df["intent_x"], df["intent_y"], output_dict=return_output_as_dict, zero_division=0
        )

    # grouping is required
    # to give out pd.DataFrame or Dict[str, pd.DataFrame] only in case of grouping.
    if intent_groups is not None:

        unique_intents = set(df["intent_x"]).union(set(df["intent_y"]))
        given_intents = set()

        for tagged_intents in intent_groups.values():
            given_intents.update(tagged_intents)

        inscope_intents = unique_intents - given_intents
        intent_groups["in_scope"] = list(inscope_intents)

        # where each intent group is having its own classification_report
        if breakdown:

            return_output_as_dict = True
            grouped_classification_reports = {}

            for group_intent, tagged_intents in intent_groups.items():

                group_classification_report = classification_report(
                    df["intent_x"], df["intent_y"], output_dict=return_output_as_dict, zero_division=0, labels=tagged_intents
                )
                group_classification_report_df = pd.DataFrame(group_classification_report).transpose()
                group_classification_report_df["support"] = group_classification_report_df["support"].astype('int32')
                grouped_classification_reports[group_intent] = group_classification_report_df

            return grouped_classification_reports

        # where each intent group just requires weighted average of precision, recall, f1, support
        else:

            weighted_group_intents_numbers : List[Dict] = []

            for group_intent, tagged_intents in intent_groups.items():

                p, r, f, _ = precision_recall_fscore_support(
                                    df["intent_x"], df["intent_y"], 
                                    labels=tagged_intents, zero_division=0, 
                                    average="weighted"
                                    )


                # since support is None, on average='weighted' on precision_recall_fscore_support
                support = df["intent_x"].isin(tagged_intents).sum()

                wgin = {
                    "group": group_intent,
                    "precision": p,
                    "recall": r,
                    "f1-score": f,
                    "support": support
                }
                weighted_group_intents_numbers.append(wgin)

            weighted_group_df = pd.DataFrame(weighted_group_intents_numbers)
            weighted_group_df.set_index('group', inplace=True)
            
            return weighted_group_df
