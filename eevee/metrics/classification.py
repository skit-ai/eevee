from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support

TRUE_COL = "intent_x"
PREDICTED_COL = "intent_y"

ALIAS_SUFFIX = "{}-alias"
LAYER_PREFIX = "layer-{}"

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
    df[[TRUE_COL, PREDICTED_COL]] = df[[TRUE_COL, PREDICTED_COL]].fillna(value="_")

    # aliasing intents
    if intent_aliases is not None:
        alias_dict = {intent: alias for alias, intent_list in intent_aliases.items() for intent in intent_list}
        for col in [TRUE_COL, PREDICTED_COL]:
            df[col] = df[col].apply(lambda intent: alias_dict.get(intent, intent))

    # vanilla case, where just ordinary classification report is required.
    # it goes out as str or dict, depending on `return_output_as_dict`
    if intent_groups is None and not breakdown:

        return classification_report(
        df[TRUE_COL], df[PREDICTED_COL], output_dict=return_output_as_dict, zero_division=0
        )

    # grouping is required
    # to give out pd.DataFrame or Dict[str, pd.DataFrame] only in case of grouping.
    if intent_groups is not None:

        # intent_groups copy, since `intent_groups` is getting mutated and giving 
        # odd behavior on even trials. 
        ig_replica = {k: v for k, v in intent_groups.items()}

        unique_intents = set(df[TRUE_COL]).union(set(df[PREDICTED_COL]))
        given_intents = set()

        for tagged_intents in ig_replica.values():
            given_intents.update(tagged_intents)

        inscope_intents = unique_intents - given_intents
        ig_replica["in_scope"] = list(inscope_intents)

        # where each intent group is having its own classification_report
        if breakdown:

            return_output_as_dict = True
            grouped_classification_reports = {}

            for group_intent, tagged_intents in ig_replica.items():

                group_classification_report = classification_report(
                    df[TRUE_COL], df[PREDICTED_COL], output_dict=return_output_as_dict, zero_division=0, labels=tagged_intents
                )
                group_classification_report_df = pd.DataFrame(group_classification_report).transpose()
                group_classification_report_df["support"] = group_classification_report_df["support"].astype('int32')
                grouped_classification_reports[group_intent] = group_classification_report_df

            return grouped_classification_reports

        # where each intent group just requires weighted average of precision, recall, f1, support
        else:

            weighted_group_intents_numbers : List[Dict] = []

            for group_intent, tagged_intents in ig_replica.items():

                p, r, f, _ = precision_recall_fscore_support(
                                    df[TRUE_COL], df[PREDICTED_COL], 
                                    labels=tagged_intents, zero_division=0, 
                                    average="weighted"
                                    )


                # since support is None, on average='weighted' on precision_recall_fscore_support
                support = df[TRUE_COL].isin(tagged_intents).sum()

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




def create_group_classification_report(
    trues: pd.Series, preds: pd.Series, 
    labels: List[str], output_dict: bool) -> pd.DataFrame:
    """
    classification report on breakdown for groups.
    """

    group_classification_report = classification_report(trues, preds, 
                                                        labels=labels,
                                                        output_dict=output_dict,
                                                        zero_division=0
                                                        )

    group_classification_report_df = pd.DataFrame(group_classification_report).transpose()
    group_classification_report_df["support"] = group_classification_report_df["support"].astype('int32')
    return group_classification_report_df


def create_wgin(
    trues: pd.Series, preds: pd.Series, 
    label: List[str], support: int) -> Dict[str, Any]:
    """
    weighted group intent numbers, otherwise
    """

    # since support is None, on average='weighted' on precision_recall_fscore_support
    p, r, f, _ = precision_recall_fscore_support(trues, preds,
                                                 labels=[label],
                                                 average="weighted",
                                                 zero_division=0
                                                 )

    wgin = {
        "layer": LAYER_PREFIX.format(label),
        "precision": p,
        "recall": r,
        "f1-score": f,
        "support": support
    }
    
    return wgin


def intent_layers_report(
        true_labels: pd.DataFrame,
        pred_labels: pd.DataFrame,
        intent_layers: Optional[Dict[str, Dict[str, List[str]]]] = None,
        breakdown=False,
):

    df = pd.merge(true_labels, pred_labels, on="id", how="inner")

    # for cases where we are seeing NaN values popping up.
    df[[TRUE_COL, PREDICTED_COL]] = df[[TRUE_COL, PREDICTED_COL]].fillna(value="_")

    # aliasing predicted column with values provided
    col = PREDICTED_COL
    intents_dict = {value: key for key, values in intent_layers.get(col).items() for value in values}
    df[col] = df[col].apply(lambda intent: intents_dict.get(intent, intent))

    #aliasing true column with values provided.
    col = TRUE_COL
    intents_dict = {value: key for key, values in intent_layers.get(col).items() for value in values}
    df[ALIAS_SUFFIX.format(col)] = df[col].apply(lambda intent: intents_dict.get(intent, intent))

    # first element is taken as the name of the original layer
    predicted_layer = list(intent_layers.get(PREDICTED_COL).keys())[0]

    # reverse aliasing dictionary - maps sublayers to original layer
    reverse_oos_dict = {sub_layer: predicted_layer for sub_layer in intent_layers.get(TRUE_COL)}

    # where each intent group is having its own classification_report
    if breakdown:

        return_output_as_dict = True
        grouped_classification_reports = {}

        for sub_layer in intent_layers.get(TRUE_COL):

            col = PREDICTED_COL
            df[ALIAS_SUFFIX.format(col)] = df[col].apply(lambda intent: {predicted_layer: sub_layer}.get(intent, intent))
            grouped_classification_reports[LAYER_PREFIX.format(sub_layer)] = create_group_classification_report(
                df[ALIAS_SUFFIX.format(TRUE_COL)],
                df[ALIAS_SUFFIX.format(PREDICTED_COL)],
                labels=[sub_layer], output_dict=return_output_as_dict
            )

        # normal oos calculations
        col = TRUE_COL
        df[ALIAS_SUFFIX.format(col)] = df[ALIAS_SUFFIX.format(col)].apply(lambda intent: reverse_oos_dict.get(intent, intent))
        grouped_classification_reports[LAYER_PREFIX.format(predicted_layer)] = create_group_classification_report(
            df[ALIAS_SUFFIX.format(TRUE_COL)],
            df[PREDICTED_COL],
            labels=[predicted_layer], output_dict=return_output_as_dict
        )

        return grouped_classification_reports

    # where each intent group just requires weighted average of precision, recall, f1, support
    else:

        weighted_group_intents_numbers: List[Dict] = []

        for sub_layer in intent_layers.get(TRUE_COL):
            
            col = PREDICTED_COL
            df[ALIAS_SUFFIX.format(col)] = df[col].apply(lambda intent: {predicted_layer: sub_layer}.get(intent, intent))
            wgin = create_wgin(
                df[ALIAS_SUFFIX.format(TRUE_COL)],
                df[ALIAS_SUFFIX.format(PREDICTED_COL)],
                label=sub_layer,
                support=df[ALIAS_SUFFIX.format(TRUE_COL)].isin([sub_layer]).sum()
            )
            weighted_group_intents_numbers.append(wgin)

        col = TRUE_COL
        df[ALIAS_SUFFIX.format(col)] = df[ALIAS_SUFFIX.format(col)].apply(lambda intent: reverse_oos_dict.get(intent, intent))
        wgin = create_wgin(
            df[ALIAS_SUFFIX.format(TRUE_COL)],
            df[PREDICTED_COL],
            label=predicted_layer,
            support=df[ALIAS_SUFFIX.format(TRUE_COL)].isin([predicted_layer]).sum()
        )
        weighted_group_intents_numbers.append(wgin)

        weighted_group_df = pd.DataFrame(weighted_group_intents_numbers)
        weighted_group_df.set_index('layer', inplace=True)

        return weighted_group_df
