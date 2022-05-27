"""
Script for picking predicted_entities (along with other columns that help with EGA) from charon's tagged_data.csv,
and saving it in a different file.

Usage:
    charon_tagged_to_categorical_entities.py <input_csv_file> <truth_csv_file> <pred_csv_file> <config_yaml>

Arguments:
  <input_csv_file>          Path to charon's tagged_data.csv
  <truth_csv_file>          Path to save true entities in a csv after processing
  <pred_csv_file>           Path to save predicted entities in a csv after processing
  <config_yaml>             Config yaml mentioning aliasing and entity type-value mappings, format defined below


Format of config yaml:

type:
# (Entity type to entity value mapping)
    product_kind:
        - debit_card
        - credit_card
    language:
        - en
        - hi
alias:
# (Aliasing true and tagged values. Might be needed to maintain homogeneity between true and pred tags)
  en: english
  hi: hindi
"""


import json
import numpy as np
import pandas as pd
import yaml
from docopt import docopt

def get_truth(tag, alias_dict):
    ## Need this to work with json errors
    try:
        tag = json.loads(tag.replace('\'', '\"').replace("None", "null"))
    except:
        return None
    if len(tag) == 0:
        return "no_entity"
    else:
        tag = tag[0]["type"]
        if tag in alias_dict:
            return alias_dict[tag]
        else:
            return tag

def parse_predicted_entity(json_data, alias_dict):
    loaded_dict = json.loads(json_data)
    if len(loaded_dict) == 0:
        return None
    else:
        op = loaded_dict[0]
        op["values"] = [{"value": op["value"]}]
        val = op["value"]
        if isinstance(val, dict) and val['type'] == 'interval':
            op["values"][0]["vale"] = val['from']['value']
            op["value"] = val['from']['value']
        elif val in alias_dict:
            op["values"][0]["value"] = alias_dict[val]
            op['value'] = alias_dict[val]
        return [op]

def insert_modified_truth(truth_value, value_type_combos):
    truth_entity_type = value_type_combos[truth_value] if truth_value in value_type_combos else None
    if truth_entity_type:
        op = {}
        op["type"] = truth_entity_type
        op["value"] = truth_value
        op["values"] = [{"value": truth_value}]
        return [op]
    return None

def process(input_path, true_output_path, pred_output_path, config):
    df = pd.read_csv(input_path)
    with open(config, 'r') as f:
        entities_dict = yaml.load(f, yaml.BaseLoader)
    value_type_combos = {value:ent_type for ent_type, ent_values in entities_dict['type'].items() for value in ent_values}
    alias_dict = entities_dict['alias']

    df["truth"] = df["tag"].apply(get_truth, args=(alias_dict,))
    df["pred"] = df["entities"].apply(parse_predicted_entity, args=(alias_dict,))
    df["modified_truth"] = df["truth"].apply(insert_modified_truth, args=(value_type_combos,))

    df["modified_truth"] = df["modified_truth"].apply(lambda x: json.dumps(x) if x else None)
    df["pred"] = df["pred"].apply(lambda x: json.dumps(x) if x else None)
    df = df.fillna(value=np.nan)

    truth_cat_df = df[["modified_truth"]]
    truth_cat_df = truth_cat_df.reset_index()
    truth_cat_df.rename(columns={"index": "id", "modified_truth": "entities"}, inplace=True)

    pred_cat_df = df[["pred"]]
    pred_cat_df = pred_cat_df.reset_index()
    pred_cat_df.rename(columns={"index": "id", "pred": "entities"}, inplace=True)

    truth_cat_df.to_csv(true_output_path, index=False)
    pred_cat_df.to_csv(pred_output_path, index=False)

    
if __name__ == "__main__":
    args = docopt(__doc__)
    process(args["<input_csv_file>"], args["<truth_csv_file>"], args["<pred_csv_file>"], args["<config_yaml>"])