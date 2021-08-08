from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def multi_class_classification_report(
    feat_df: pd.DataFrame,
    label_df: pd.DataFrame = pd.DataFrame(),
    data_id: str = "id",
    label_col: str = "labels",
    predicted_col: str = "preds",
    subsets: Optional[Dict[str, List[str]]] = None,
    output_dict: bool = False,
    zero_division: int = 0,
) -> Union[Dict[str, Any], str]:
    """
    Report for multi-class classification.

    We produce classification report for dataframes `feat_df` and `label_df`.
    `label_df` is not required, if the labels are present within the `feat_df`.

    We can produce subsets of this report by passing a `subsets` argument. 
    An example subset looks like:

    ```python
    {
        "small-talk": ["_confirm_", "_cancel_"],
        "unhandled": ["_oos_"]
    }
    ```

    This automatically sets the `output_dict` to `True`, 
    ... and we return a dict like so:

    ```python
    {
        "__main": {
            "label": {
                "precision": float,
                "recall": float,
                "f1-score": float,
                "support": int
            }
        },
        "subset_name": {
            "label": {
                "precision": float,
                "recall": float,
                "f1-score": float,
                "support": int
            }
        }
    }
    ```

    :param feat_df: DataFrame containing prediction value for features and the identifier. Possibly, the true-labels too.
    :type feat_df: pd.DataFrame
    :param label_df: DataFrame containing identifier to link with feat_df and true-labels, defaults to None
    :type label_df: Optional[pd.DataFrame], optional
    :param data_id: Identifier field that is common to feat_df and label_df, defaults to "id"
    :type data_id: str, optional
    :param label_col: Name of the true label column, defaults to "labels"
    :type label_col: str, optional
    :param predicted_col: Name of the predicted label column, defaults to "preds"
    :type predicted_col: str, optional
    :param subsets: A dict of subset name as keys and , defaults to None
    :type subsets: Optional[Dict[str, List[str]]], optional
    :param output_dict: If True, return output of classification report as a dict., defaults to False
    :type output_dict: bool, optional
    :param zero_division: Sets the value to return when there is a zero division. If set to “warn”, this acts as 0, but warnings are also raised, defaults to 0
    :type zero_division: int, optional
    :return: The classification report as a string if output_dict is False otherwise a dict.
    :rtype: Union[Dict[str, Any], str]
    """
    reports = {}
    main_report_key = "__main"
    if subsets:
        output_dict = True

    df = (
        feat_df
        if label_df.empty
        else pd.merge(label_df, feat_df, on=data_id, how="inner")
    )

    reports[main_report_key] = classification_report(
        df[predicted_col],
        df[label_col],
        output_dict=output_dict,
        zero_division=zero_division,
    )

    if subsets:
        for subset_title, labels in subsets.items():
            if not labels:
                continue
            df_ = df[df[label_col].isin(labels) | df[predicted_col].isin(labels)]
            reports[subset_title] = classification_report(
                df_[predicted_col],
                df_[label_col],
                output_dict=output_dict,
                zero_division=zero_division,
            )

    return reports if subsets else reports[main_report_key]


def get_confusions(
    feat_df: pd.DataFrame,
    label_df: Optional[pd.DataFrame] = None,
    data_id: str = "data_id",
    label_col: str = "labels",
    predicted_col: str = "preds",
    threshold: int = 0,
):
    """
    Get a map of true labels that are being predicted incorrectly.

    :param feat_df: DataFrame containing prediction value for features and the identifier. Possibly, the true-labels too.
    :type feat_df: pd.DataFrame
    :param label_df: DataFrame containing identifier to link with feat_df and true-labels, defaults to None
    :type label_df: Optional[pd.DataFrame], optional
    :param data_id: Identifier field that is common to feat_df and label_df, defaults to "id"
    :type data_id: str, optional
    :param label_col: Name of the true label column, defaults to "labels"
    :type label_col: str, optional
    :param predicted_col: Name of the predicted label column, defaults to "preds"
    :type predicted_col: str, optional
    :param threshold: Only confusions above this value will be returned, defaults to 0
    :type threshold: int, optional
    :return: [description]
    :rtype: [type]
    """
    df = (
        feat_df
        if not label_df
        else pd.merge(label_df, feat_df, on=data_id, how="inner")
    )
    labels = sorted(
        set(df[predicted_col].unique().tolist() + df[label_col].unique().tolist())
    )

    matrix = confusion_matrix(df[predicted_col], df[label_col], labels=labels)

    confusions = {}
    for true_idx, true_label in enumerate(labels):
        confusion = [
            {labels[pred_idx]: matrix[true_idx][pred_idx]}
            for pred_idx in np.where(matrix[true_idx] > threshold)[0]
            if pred_idx != true_idx
        ]
        if confusion:
            confusions[true_label] = confusion
    return confusions
