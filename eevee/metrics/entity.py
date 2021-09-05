"""
Entity comparison and reporting functions.
"""

import json

import pandas as pd
from pydash import py_


def entity_report(true_labels: pd.DataFrame, pred_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Generate entity report based on true and predicted labels.

    Items follow `EntityLabel` protobuf definition.
    """

    df = pd.merge(true_labels, pred_labels, on="id", how="inner")
    df["true"] = df["entities_x"].apply(lambda it: json.loads(it))
    df["pred"] = df["entities_y"].apply(lambda it: json.loads(it))

    # All the unique entity types in the dataset
    entity_types = sorted(set([ent["type"] for ent in py_.flatten(df["true"].tolist() + df["pred"].tolist())]))

    # TODO: Handle compositional entities like datetime
    report = []

    for entity_type in entity_types:
        report.append({
            "Entity": entity_type,
            "FPR": "NA",
            "FNR": "NA",
            "Mismatch Rate": "NA",
            "Support": "NA",
            "Positives": "NA",
            "Negatives": "NA"
        })

    report = pd.DataFrame(report)
    report.set_index("Entity", inplace=True)
    return report
