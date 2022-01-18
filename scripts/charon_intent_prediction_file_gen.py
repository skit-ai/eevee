"""
Script for extracting true prediction from the tagged_data.csv and dumping it as separate .csv

Usage:
    charon_intent_prediction_file_gen.py <input_tagged_data_file> <output_csv_file>
"""

import json
from typing import Optional, Any

import pandas as pd
import numpy as np
from docopt import docopt


def extract_predicted_intent(json_string: Any) -> Optional[str]:

    if isinstance(json_string, str):

        prediction = json.loads(json_string)
        if isinstance(prediction, dict) and "name" in prediction:
            return prediction["name"]

    return np.nan


def convert(df: pd.DataFrame):

    df["predicted_intent"] = df["prediction"].apply(extract_predicted_intent)
    df = df.rename(columns={"predicted_intent": "intent"})
    df = df[["id", "intent"]]
    return df


if __name__ == "__main__":

    args = docopt(__doc__)    
    df = pd.read_csv(args["<input_tagged_data_file>"], usecols=["id", "prediction"])
    df = convert(df)
    df.to_csv(args["<output_csv_file>"], index=False)
