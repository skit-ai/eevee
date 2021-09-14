"""
Entity comparison and reporting functions.
"""


import json
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
from pydash import py_

import eevee.ord.entity.datetime as ord_datetime
import eevee.ord.entity.people as ord_people
import eevee.ord.entity.number as ord_number

from eevee.metrics.slot_filling import (mismatch_rate, slot_fnr, 
                                        slot_fpr, slot_positives, slot_negatives
                                        )


ENTITY_EQ_FNS = {
    "date": ord_datetime.date_eq,
    "time": ord_datetime.time_eq,
    "people": ord_people.eq,
    "number": ord_number.eq
}


ENTITY_EQ_ALIAS = {
    "number": "number",
    "people": "number",
}


@dataclass
class EntityComparisonResult:
    tp: Dict
    fp: Dict
    fn: Dict
    mm: Dict


def are_these_types_equal(true_ent_type, pred_ent_type):

    if (true_ent_type in ENTITY_EQ_ALIAS and 
        pred_ent_type in ENTITY_EQ_ALIAS and
        (ENTITY_EQ_ALIAS[true_ent_type] == ENTITY_EQ_ALIAS[pred_ent_type])
    ) or (true_ent_type == pred_ent_type and true_ent_type is not None):
        return True
    return False



def compare_datetime_special_entities(row) -> EntityComparisonResult:

    tp = {}
    fp = {}
    fn = {}
    mm = {}

    if row["true"] is None:
        true_ent = None
    else:
        true_ent = row["true"][0]

    if row["pred"] is None:
        pred_ent = None
    else:
        pred_ent = row["pred"][0]

    if true_ent is None and pred_ent is None:
        return None

    pred_ent_type = row["pred_ent_type"]
    true_ent_type = row["true_ent_type"]

    if true_ent_type == "datetime" and pred_ent_type == "datetime":

        datetime_two_types = ["date", "time"]

        for dtt in datetime_two_types:

            eq_fn_for_this_entity = ENTITY_EQ_FNS[dtt]
            is_this_entity_type_value_equal = eq_fn_for_this_entity(true_ent, pred_ent)

            if is_this_entity_type_value_equal:
                tp[dtt] = 1
            else:
                mm[dtt] = 1


    elif true_ent_type == "datetime":
        
        if pred_ent_type in ["time", "date"]:

            eq_fn_for_this_entity = ENTITY_EQ_FNS[pred_ent_type]

            is_this_entity_type_value_equal = eq_fn_for_this_entity(true_ent, pred_ent)

            if is_this_entity_type_value_equal:
                tp[pred_ent_type] = 1
            else:
                mm[pred_ent_type] = 1
            
            if pred_ent_type == "time":
                fn["date"] = 1
            elif pred_ent_type == "date":
                fn["time"] = 1

        else:
            fn["date"] = 1
            fn["time"] = 1
    
    elif pred_ent_type == "datetime":

        if true_ent_type in ["time", "date"]:

            eq_fn_for_this_entity = ENTITY_EQ_FNS[true_ent_type]

            is_this_entity_type_value_equal = eq_fn_for_this_entity(true_ent, pred_ent)
            if is_this_entity_type_value_equal:
                tp[true_ent_type] = 1
            else:
                mm[true_ent_type] = 1
            
            if true_ent_type == "time":
                fp["date"] = 1
            elif true_ent_type == "date":
                fp["time"] = 1

        else:
            fp["date"] = 1
            fp["time"] = 1


    ecr = EntityComparisonResult(tp=tp, fp=fp, fn=fn, mm=mm)
    return ecr



def compare_row_level_entities(row) -> Optional[EntityComparisonResult]:
    """
    if truth's entity type == prediction's entity type:
        if truth's entity value == predictions's entity value:
            true positive for entity type
        else:
            mismatch on value for this entity type.
    else:
        # type didn't match here
        true entity type should be predicted but it didn't happen, 
            therefore false negative for true entity type

        unexpected entity type got predicted happened,
            therefore false positive for predicted entity type
    """


    tp = {}
    fp = {}
    fn = {}
    mm = {}

    if row["true"] is None:
        true_ent = None
    else:
        true_ent = row["true"][0]

    if row["pred"] is None:
        pred_ent = None
    else:
        pred_ent = row["pred"][0]

    if true_ent is None and pred_ent is None:
        return None


    # special case handling where one entity is `datetime`
    if "datetime" in [row["true_ent_type"], row["pred_ent_type"]]:
        ecr = compare_datetime_special_entities(row)
        return ecr


    if are_these_types_equal(row["true_ent_type"], row["pred_ent_type"]):
        # type matched here
        
        eq_fn_for_this_entity = ENTITY_EQ_FNS[row["true_ent_type"]]
        is_this_entity_type_value_equal = eq_fn_for_this_entity(true_ent, pred_ent)
        if is_this_entity_type_value_equal:
            # value matched here
            if row["true_ent_type"] in tp:
                tp[row["true_ent_type"]] += 1
            else:
                tp[row["true_ent_type"]] = 1
        else:
            # mismatch on value for this entity type.
            if row["true_ent_type"] in mm:
                mm[row["true_ent_type"]] += 1
            else:
                mm[row["true_ent_type"]] = 1
            
    else:
        # type didn't match here
        # true entity type should be predicted but it didn't happen, 
        #     therefore false negative for true entity type
        if row["true_ent_type"] in fn:
            fn[row["true_ent_type"]] += 1
        else:
            fn[row["true_ent_type"]] = 1

        # unexpected entity type got predicted happened,
        #     therefore false positive for predicted entity type
        if row["pred_ent_type"] in fp:
            fp[row["pred_ent_type"]] += 1
        else:
            fp[row["pred_ent_type"]] = 1

    ecr = EntityComparisonResult(tp=tp, fp=fp, fn=fn, mm=mm)
    return ecr



def get_entity_df_by_ecr(df: pd.DataFrame, entity_type: str):

    entity_idxs = []

    for idx, row in df.iterrows():

        ecr = row["entity_comp_results"]

        if ecr is None:
            continue

        if entity_type in ecr.tp or entity_type in ecr.fn or entity_type in ecr.fp or entity_type in ecr.mm:
            entity_idxs.append(idx)

    return df.loc[df.index[entity_idxs]]


def entity_report(true_labels: pd.DataFrame, pred_labels: pd.DataFrame) -> pd.DataFrame:


    df = pd.merge(true_labels, pred_labels, on="id", how="inner")
    df.dropna(how="all", subset=["entities_x", "entities_y"], inplace=True)
    df["true"] = df["entities_x"].apply(lambda it: json.loads(it) if isinstance(it, str) else None)
    df["pred"] = df["entities_y"].apply(lambda it: json.loads(it) if isinstance(it, str) else None)

    # assuming there will be only one entity type and value 
    df["true_ent_type"] = df["true"].apply(lambda it: it[0].get("type") if it else None)
    df["pred_ent_type"] = df["pred"].apply(lambda it: it[0].get("type") if it else None)

    df.reset_index(inplace=True)

    # All the unique entity types in the dataset
    entity_types = sorted(set([ent["type"] for ent in py_.flatten(df["true"].dropna().tolist() + df["pred"].dropna().tolist())]))

    # TODO: Handle compositional entities like datetime
    report = []

    df["entity_comp_results"] = df.apply(compare_row_level_entities, axis=1)

    for entity_type in entity_types:

        if entity_type in ENTITY_EQ_FNS:

            # entity type df needs to consider `entity_type` in tp, fp, fn, mm in "entity_comp_results"
            entity_type_df = get_entity_df_by_ecr(df, entity_type)
            entity_support = 0

            y_true = []
            y_pred = []

            y_true_mmr = []
            y_pred_mmr = []


            for _, row in entity_type_df.iterrows():

                true_ent = None if row["true"] is None else row["true"][0]
                pred_ent = None if row["pred"] is None else row["pred"][0]

                if row["entity_comp_results"] is None:
                    continue
                else:
                    ecr = row["entity_comp_results"]

                if (
                    (row["true_ent_type"] == entity_type) or # for ordinary support
                    (ENTITY_EQ_ALIAS.get(row["true_ent_type"]) == entity_type) or # for number vs people
                    (entity_type in ecr.tp or entity_type in ecr.fn or entity_type in ecr.mm) # for datetime, date, time mess
                ):
                    entity_support += 1

                if entity_type in ecr.tp:
                    y_true.append(True)
                    y_pred.append(True)
                
                # false negative, we expected a prediction but didn't happen.
                if entity_type in ecr.fn:
                    y_true.append(True)
                    y_pred.append(None)

                # false positive, no prediction should have happened
                if entity_type in ecr.fp:
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
                "Support": entity_support,
                "Positives": ent_pos,
                "Negatives": ent_neg
            })

    return pd.DataFrame(report)