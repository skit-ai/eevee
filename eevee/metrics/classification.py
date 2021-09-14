from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def intent_report(true_labels: pd.DataFrame, pred_labels: pd.DataFrame, output_dict=False):
    """
    Make an intent report from given label dataframes. We only support single
    intent dataframes as of now.
    TODO:
    - Check type of labels (we are not supporting rich labels right now)
    - Handle 'null' labels.
    """
    df = pd.merge(true_labels, pred_labels, on="id", how="inner")

    return classification_report(df["intent_x"], df["intent_y"], output_dict=output_dict, zero_division=0)
