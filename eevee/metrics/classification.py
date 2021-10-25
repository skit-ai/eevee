from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd
from sklearn.metrics import classification_report
from eevee.metrics.utils import convert_classification_report_dict_into_dataframe, weighted_avg_of_labels


def intent_report(
    true_labels: pd.DataFrame,
    pred_labels: pd.DataFrame,
    output_dict=False,
    intent_groups: Optional[Dict[str, List[str]]]=None,
    breakdown=False,
):

    df = pd.merge(true_labels, pred_labels, on="id", how="inner")

    # vanilla case, where just ordinary classification report is required.
    # it goes out as str or dict, depending on `output_dict`
    if intent_groups is None and not breakdown:

        return classification_report(
        df["intent_x"], df["intent_y"], output_dict=output_dict, zero_division=0
        )

    # grouping is required
    # to give out pd.DataFrame or Dict[str, pd.DataFrame] only in case of grouping.
    if intent_groups is not None:

        unique_intents = set(df["intent_x"].unique()).union(set(df["intent_y"].unique()))
        given_intents = set()

        for _, tagged_intents in intent_groups.items():
            given_intents.update(tagged_intents)

        inscope_intents = unique_intents - given_intents
        intent_groups["in_scope"] = list(inscope_intents)

        # where each intent group is having its own classification_report
        if breakdown:

            grouped_classification_reports = {}

            for alias_intent, tagged_intents in intent_groups.items():

                group_classification_report = classification_report(
                    df["intent_x"], df["intent_y"], output_dict=True, zero_division=0, labels=tagged_intents
                )
                group_classification_report_df = convert_classification_report_dict_into_dataframe(group_classification_report)
                group_classification_report_df["support"] = group_classification_report_df["support"].astype('int32')
                grouped_classification_reports[alias_intent] = group_classification_report_df

            return grouped_classification_reports

        # where each intent group just requires weighted average of precision, recall, f1, support
        else:

            weighted_group_intents_numbers : Dict[str, Any] = []

            for alias_intent, tagged_intents in intent_groups.items():

                p, r, f, _ = weighted_avg_of_labels(df["intent_x"], df["intent_y"], labels=tagged_intents)

                # since support is None, on average='weighted' on precision_recall_fscore_support
                support = df["intent_x"].isin(tagged_intents).sum()

                wgin = {
                    "group": alias_intent,
                    "precision": p,
                    "recall": r,
                    "f1-score": f,
                    "support": support
                }
                weighted_group_intents_numbers.append(wgin)

            weighted_group_df = pd.DataFrame(weighted_group_intents_numbers)
            weighted_group_df.set_index('group', inplace=True)
            
            return weighted_group_df



