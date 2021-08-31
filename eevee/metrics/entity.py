"""
Entity comparison and reporting functions.
"""

from eevee.metrics.slot_filling import (mismatch_rate, slot_fnr, 
                                        slot_fpr, slot_positives, slot_negatives
                                        )
import json

import pandas as pd
from pydash import py_

import eevee.ord.entity.datetime as ord_datetime
import eevee.ord.entity.people as ord_people

EQ_TYPES = {
    "time": ["datetime", "time"],
    "date": ["datetime", "date"],
    "people": ["people", "number"],
}

EQ_LIST_FNS = {
    "date": ord_datetime.date_eq_lists,
    "time": ord_datetime.time_eq_lists,
    "people": ord_people.eq_lists,
}


ENTITY_EQ_FNS = {
    "date": ord_datetime.date_eq,
    "time": ord_datetime.time_eq,
    "people": ord_people.eq
}


def entity_report(true_labels: pd.DataFrame, pred_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Generate entity report based on true and predicted labels.

    Items follow `EntityLabel` protobuf definition.
    """

    df = pd.merge(true_labels, pred_labels, on="id", how="inner")
    df["true"] = df["entities_x"].apply(lambda it: json.loads(it))
    df["pred"] = df["entities_y"].apply(lambda it: json.loads(it))

    # assuming there will be only one entity type and value 
    df["true_ent_type"] = df["true"].apply(lambda it: it[0].get("type"))
    df["pred_ent_type"] = df["pred"].apply(lambda it: it[0].get("type"))

    df["true_ent_value"] = df["true"].apply(lambda it: it[0]["values"][0].get("value"))
    df["pred_ent_value"] = df["pred"].apply(lambda it: it[0]["values"][0].get("value"))

    # All the unique entity types in the dataset
    entity_types = sorted(set([ent["type"] for ent in py_.flatten(df["true"].tolist() + df["pred"].tolist())]))

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

                true_ent = row["true"][0]
                pred_ent = row["pred"][0]

                if row["true_ent_type"] == entity_type and row["pred_ent_type"] == entity_type:

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

                # else:
                #     # TODO: don't know how to handle. types matching, but values not matching.
                #     # that is mismatch rate?
                #     pass
                #     # y_true.append(None)
                #     # y_pred.append(None)

                y_true_mmr.append(row["true"][0])
                y_pred_mmr.append(row["pred"][0])

            ent_fpr = slot_fpr(y_true, y_pred)
            ent_fnr = slot_fnr(y_true, y_pred)
            ent_mmr = mismatch_rate(y_true_mmr, y_pred_mmr)

            ent_pos = slot_positives(y_true, y_pred)
            ent_neg = slot_negatives(y_true, y_pred)

            report.append({
                "Entity": entity_type,
                "FPR": f"{ent_fpr:.2f}",
                "FNR": f"{ent_fnr:.2f}",
                "Mismatch Rate": f"{ent_mmr:.2f}",
                "Support": len(y_true),
                "Positives": f"{ent_pos:.2f}",
                "Negatives": f"{ent_neg:.2f}"
            })

    return pd.DataFrame(report)
