"""
Entity comparison and reporting functions.
"""

from eevee.metrics.slot_filling import (mismatch_rate, slot_fnr, 
                                        slot_fpr, slot_positives, slot_negatives
                                        )
import json

import numpy as np
import pandas as pd
from pydash import py_

import eevee.ord.entity.datetime as ord_datetime
import eevee.ord.entity.people as ord_people
import eevee.ord.entity.number as ord_number

EQ_TYPES = {
    "time": ["datetime", "time"],
    "date": ["datetime", "date"],
    "people": ["people", "number"],
}

# NOTE: not supporting multiple value comparisons presently.
# EQ_LIST_FNS = {
#     "date": ord_datetime.date_eq_lists,
#     "time": ord_datetime.time_eq_lists,
#     "people": ord_people.eq_lists,
# }


ENTITY_EQ_FNS = {
    "date": ord_datetime.date_eq,
    "time": ord_datetime.time_eq,
    "people": ord_people.eq,
    "number": ord_number.eq
}


ENTITY_EQ_ALIAS = {
    "date": "datetime",
    "time": "datetime",
    "datetime": "datetime",
    "number": "people",
    "people": "people",
}


def are_these_types_equal(true_ent_type, pred_ent_type):

    if (true_ent_type in ENTITY_EQ_ALIAS and 
        pred_ent_type in ENTITY_EQ_ALIAS and
        (ENTITY_EQ_ALIAS[true_ent_type] == ENTITY_EQ_ALIAS[pred_ent_type])
    ):
        return True
    return False



def entity_report(true_labels: pd.DataFrame, pred_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Generate entity report based on true and predicted labels.

    Items follow `EntityLabel` protobuf definition.
    """

    df = pd.merge(true_labels, pred_labels, on="id", how="inner")
    df.dropna(how="all", subset=["entities_x", "entities_y"], inplace=True)
    df["true"] = df["entities_x"].apply(lambda it: json.loads(it) if isinstance(it, str) else None)
    df["pred"] = df["entities_y"].apply(lambda it: json.loads(it) if isinstance(it, str) else None)

    # assuming there will be only one entity type and value 
    df["true_ent_type"] = df["true"].apply(lambda it: it[0].get("type") if it else None)
    df["pred_ent_type"] = df["pred"].apply(lambda it: it[0].get("type") if it else None)

    # All the unique entity types in the dataset
    entity_types = sorted(set([ent["type"] for ent in py_.flatten(df["true"].dropna().tolist() + df["pred"].dropna().tolist())]))

    # TODO: Handle compositional entities like datetime
    report = []

    for entity_type in entity_types:

        if entity_type in ENTITY_EQ_FNS:

            entity_type_df = df[(df["true_ent_type"] == entity_type) | (df["pred_ent_type"] == entity_type)]

            y_true = []
            y_pred = []

            y_true_mmr = []
            y_pred_mmr = []

            eq_fn_for_this_entity = ENTITY_EQ_FNS[entity_type]

            for _, row in entity_type_df.iterrows():

                if row["true"] is None:
                    true_ent = None
                else:
                    true_ent = row["true"][0]

                if row["pred"] is None:
                    pred_ent = None
                else:
                    pred_ent = row["pred"][0]

                if are_these_types_equal(row["true_ent_type"], row["pred_ent_type"]):
                    is_this_entity_type_value_equal = eq_fn_for_this_entity(true_ent, pred_ent)
                    if is_this_entity_type_value_equal:
                        y_true.append(True)
                        y_pred.append(True)

                # false negative, we expected a prediction but didn't happen.
                elif row["true_ent_type"] == entity_type and row["pred_ent_type"] != entity_type:
                    y_true.append(True)
                    y_pred.append(None)

                # false positive, no prediction should have happened
                elif row["true_ent_type"] != entity_type and row["pred_ent_type"] == entity_type:
                    y_true.append(None)
                    y_pred.append(True)

                y_true_mmr.append(true_ent)
                y_pred_mmr.append(pred_ent)

            ent_fpr = slot_fpr(y_true, y_pred)
            ent_fnr = slot_fnr(y_true, y_pred)
            ent_mmr = mismatch_rate(y_true_mmr, y_pred_mmr)

            ent_pos = slot_positives(y_true, y_pred)
            ent_neg = slot_negatives(y_true, y_pred)

            report.append({
                "Entity": entity_type,
                "FPR": ent_fpr,
                "FNR": ent_fnr,
                "Mismatch Rate": ent_mmr,
                "Support": len(y_true),
                "Positives": ent_pos,
                "Negatives": ent_neg
            })

    return pd.DataFrame(report)
