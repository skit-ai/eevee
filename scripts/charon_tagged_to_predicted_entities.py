"""
Script for picking predicted_entities (along with other columns that help with EGA) from charon's tagged_data.csv,
and saving it in a different file.

Usage:
    charon_tagged_to_predicted_entities.py <input_csv_file> <output_csv_file>

Arguments:
  <input_csv_file>          Path to charon's tagged_data.csv
  <output_csv_file>         Path to save predicted entities in a csv
"""


import pandas as pd
from docopt import docopt

def convert(input_path, output_path):

    usecols = [
        "id", "predicted_entities", 
        "call_uuid", "conversation_uuid",
        "alternatives", "audio_url", "prediction", "state"
    ]

    df = pd.read_csv(input_path, usecols=usecols)
    df = df.rename(columns={"predicted_entities": "entities"})
    df.to_csv(output_path, index=False)


if __name__ == "__main__":

    args = docopt(__doc__)
    convert(args["<input_csv_file>"], args["<output_csv_file>"])
