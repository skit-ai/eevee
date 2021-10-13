"""
Script for converting the json file into the format required by eevee

Usage:
    get_csv_from_json.py <input_json_file> <output_csv_file>
"""

# read a json file
import json
import pandas as pd
from docopt import docopt


def get_csv_from_json(json_file) -> pd.DataFrame:
    """
    :param json_file: json file
    :return: df 
    """

    # create a pandas dataframe with "utterances" as its column
    with open(json_file) as f:
        data = json.load(f)

    df = pd.DataFrame(data.values(), columns=['utterances'])
    df.index.name = 'id'

    return df


if __name__ == "__main__":
    args = docopt(__doc__)
    json_file = args["<input_json_file>"]
    df = get_csv_from_json(json_file)
    df.to_csv(args["<output_csv_file>"])
