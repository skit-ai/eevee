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

        # intent_groups copy, since `intent_groups` is getting mutated and giving 
        # odd behavior on even trials. 
        ig_replica = {k: v for k, v in intent_groups.items()}

        unique_intents = set(df["intent_x"]).union(set(df["intent_y"]))
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
                    df["intent_x"], df["intent_y"], output_dict=return_output_as_dict, zero_division=0, labels=tagged_intents
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


def intent_layers_report(
        true_labels: pd.DataFrame,
        pred_labels: pd.DataFrame,
        intent_layers: Optional[Dict[str, Dict[str, List[str]]]] = None,
        breakdown=False,
):
    df = pd.merge(true_labels, pred_labels, on="id", how="inner")

    # for cases where we are seeing NaN values popping up.
    df[['intent_x', 'intent_y']] = df[['intent_x', 'intent_y']].fillna(value="_")

    # aliasing preds
    col = "intent_y"
    intents_dict = {value: key for key, values in intent_layers.get(col).items() for value in values}
    df[col] = df[col].apply(lambda intent: intents_dict.get(intent, intent))

    #aliasing trues
    col = "intent_x"
    intents_dict = {value: key for key, values in intent_layers.get(col).items() for value in values}
    df["{}-alias".format(col)] = df[col].apply(lambda intent: intents_dict.get(intent, intent))

    PREDICTED_LAYER = list(intent_layers.get("intent_y").keys())[0]

    # where each intent group is having its own classification_report
    if breakdown:

        return_output_as_dict = True
        grouped_classification_reports = {}

        for sub_layer in intent_layers.get("intent_x"):
            col = "intent_y"
            df["{}-alias".format(col)] = df[col].apply(lambda intent: {PREDICTED_LAYER: sub_layer}.get(intent, intent))
            group_classification_report = classification_report(df["{}-alias".format("intent_x")],
                                                                df["{}-alias".format("intent_y")],
                                                                labels=[sub_layer], output_dict=return_output_as_dict, zero_division=0)
            group_classification_report_df = pd.DataFrame(group_classification_report).transpose()
            group_classification_report_df["support"] = group_classification_report_df["support"].astype('int32')
            grouped_classification_reports["layer-{}".format(sub_layer)] = group_classification_report_df

        # normal oos calculations
        reverse_oos = {sub_layer: PREDICTED_LAYER for sub_layer in intent_layers.get("intent_x")}
        col = "intent_x"
        df["{}-alias".format(col)] = df["{}-alias".format(col)].apply(lambda intent: reverse_oos.get(intent, intent))
        group_classification_report = classification_report(df["{}-alias".format("intent_x")],
                                                            df["intent_y"],
                                                            labels=[PREDICTED_LAYER], output_dict=return_output_as_dict, zero_division=0)
        group_classification_report_df = pd.DataFrame(group_classification_report).transpose()
        group_classification_report_df["support"] = group_classification_report_df["support"].astype('int32')
        grouped_classification_reports["layer-{}".format(PREDICTED_LAYER)] = group_classification_report_df

        return grouped_classification_reports

    # where each intent group just requires weighted average of precision, recall, f1, support
    else:

        weighted_group_intents_numbers: List[Dict] = []

        for sub_layer in intent_layers.get("intent_x"):
            col = "intent_y"
            df["{}-alias".format(col)] = df[col].apply(lambda intent: {PREDICTED_LAYER: sub_layer}.get(intent, intent))
            p, r, f, _ = precision_recall_fscore_support(
                df["{}-alias".format("intent_x")],
                df["{}-alias".format("intent_y")],
                labels=[sub_layer], zero_division=0,average="weighted"
            )

            # since support is None, on average='weighted' on precision_recall_fscore_support
            support = df["{}-alias".format("intent_x")].isin([sub_layer]).sum()

            wgin = {
                "layer": "layer-{}".format(sub_layer),
                "precision": p,
                "recall": r,
                "f1-score": f,
                "support": support
            }
            weighted_group_intents_numbers.append(wgin)

        reverse_oos = {sub_layer: PREDICTED_LAYER for sub_layer in intent_layers.get("intent_x")}
        col = "intent_x"
        df["{}-alias".format(col)] = df["{}-alias".format(col)].apply(lambda intent: reverse_oos.get(intent, intent))
        p, r, f, _ = precision_recall_fscore_support(
            df["{}-alias".format("intent_x")],
            df["intent_y"],
            labels=[PREDICTED_LAYER], zero_division=0, average="weighted"
        )

        # since support is None, on average='weighted' on precision_recall_fscore_support
        support = df["{}-alias".format("intent_x")].isin([PREDICTED_LAYER]).sum()

        wgin = {
            "layer": "layer-{}".format(PREDICTED_LAYER),
            "precision": p,
            "recall": r,
            "f1-score": f,
            "support": support
        }
        weighted_group_intents_numbers.append(wgin)

        weighted_group_df = pd.DataFrame(weighted_group_intents_numbers)
        weighted_group_df.set_index('layer', inplace=True)

        return weighted_group_df
