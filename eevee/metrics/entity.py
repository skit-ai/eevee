"""
Entity comparison and reporting functions.
"""


from eevee.metrics.utils import weighted_avg_dropna
import json
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from pydash import py_
from sklearn.metrics import classification_report

import eevee.ord.entity.datetime as ord_datetime
import eevee.ord.entity.people as ord_people
import eevee.ord.entity.number as ord_number


# legacy plute.ord equality functions for entities
# derived from : https://gitlab.com/vernacularai/ai/clients/plute/-/tree/master/plute/ord/entities
ENTITY_EQ_FNS = {
    "date": ord_datetime.date_eq,
    "time": ord_datetime.time_eq,
    "people": ord_people.eq,
    "number": ord_number.eq
}


# ENTITY_EQ_ALIAS = {
#     "number": "number",
#     "people": "number",
# }


@dataclass
class EntityComparisonResult:
    tp: Dict
    fp: Dict
    fn: Dict
    mm: Dict


def dump_error_reports(df: pd.DataFrame, fp_error_idxs, fn_error_idxs, mm_errror_idxs):
    """
    dumps the .csv files for fp, fn, mm of all entities
    for deeper analysis.
    """

    df.drop(labels=["index", "true", "pred", "entity_comp_results"], axis=1, inplace=True)
    df.rename(columns={"entities_x": "true_entities", "entities_y": "pred_entities"}, inplace=True)

    fp_df = df.loc[df.index[fp_error_idxs]]
    fn_df = df.loc[df.index[fn_error_idxs]]
    mm_df = df.loc[df.index[mm_errror_idxs]]

    fp_df.to_csv("./fp.csv", index=False)
    fn_df.to_csv("./fn.csv", index=False)
    mm_df.to_csv("./mm.csv", index=False)


def are_generic_entity_type_and_value_equal(true_ent, pred_ent):
    """
    checks values for two given entities are same or not.
    """

    true_ent_value = true_ent["values"][0]["value"]
    pred_ent_value = pred_ent["values"][0]["value"]
    return true_ent_value == pred_ent_value


def are_these_types_equal(true_ent_type, pred_ent_type):
    """
    compares if two entities are of same type.

    in future it may support scenes where number and people
    are interchange-able entities.
    """

    # TODO: decide upon aliasing or not for number
    # people 

    # if (true_ent_type in ENTITY_EQ_ALIAS and 
    #     pred_ent_type in ENTITY_EQ_ALIAS and
    #     (ENTITY_EQ_ALIAS[true_ent_type] == ENTITY_EQ_ALIAS[pred_ent_type]):

    if (true_ent_type == pred_ent_type and true_ent_type is not None):
        return True
    return False


def compare_datetime_special_entities(row) -> Optional[EntityComparisonResult]:
    """
    datetime is a different category, datetime needs to be broken
    down into date & time. 

    accordingly their entities will be compared with date_eq/time_eq
    not datetime_eq

    therefore these comparisons (where atleast one entity is datetime),
    will result in date or time, being part of tp/fp/fn/mm.
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

    # when both truth and prediction is absent (NaN) for this turn id
    # we go ahead and return None for ECR
    if true_ent is None and pred_ent is None:
        return None

    pred_ent_type = row["pred_ent_type"]
    true_ent_type = row["true_ent_type"]

    # when both truth and predicted entities are datetime
    if true_ent_type == "datetime" and pred_ent_type == "datetime":

        # we have to compare both their dates & times separately
        # and act accordingly with date/time's tp/fp/fn/mm
        datetime_two_types = ["date", "time"]

        for dtt in datetime_two_types:

            eq_fn_for_this_entity = ENTITY_EQ_FNS[dtt]
            is_this_entity_type_value_equal = eq_fn_for_this_entity(true_ent, pred_ent)

            if is_this_entity_type_value_equal:
                tp[dtt] = 1
            else:
                mm[dtt] = 1


    elif true_ent_type == "datetime":
        
        # situation where truth is datetime,
        # but prediction is either time/date
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

            # situation where the truth is datetime
            # but prediction is neither a date/time
            # therefore they are false negatives for date
            # and time.
            # and false positive for predicted entity.
            fn["date"] = 1
            fn["time"] = 1
            if pred_ent is not None:
                fp[pred_ent_type] = 1
    
    elif pred_ent_type == "datetime":

        # situation where prediction is datetime,
        # but truth is either time/date
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
            # situation where the prediction is datetime
            # but truth is neither a date/time
            # therefore they are false positives for date
            # and time.
            # and false negative for true entity that didn't
            # get predicted.
            fp["date"] = 1
            fp["time"] = 1

            if true_ent is not None:
                fn[true_ent_type] = 1


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


    tp : Dict[str, int] = {}
    fp : Dict[str, int] = {}
    fn : Dict[str, int] = {}
    mm : Dict[str, int] = {}

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

    true_ent_type = row["true_ent_type"]
    pred_ent_type = row["pred_ent_type"]

    # special case handling where one entity is `datetime`
    if "datetime" in [true_ent_type, pred_ent_type]:
        ecr = compare_datetime_special_entities(row)
        return ecr


    if are_these_types_equal(true_ent_type, pred_ent_type):
        # type matched here
        
        if true_ent_type in ENTITY_EQ_FNS:
            eq_fn_for_this_entity : Callable = ENTITY_EQ_FNS[true_ent_type]
            is_this_entity_type_value_equal = eq_fn_for_this_entity(true_ent, pred_ent)
        else:
            # entity_type outside datetime, date, time, number, people
            is_this_entity_type_value_equal = are_generic_entity_type_and_value_equal(true_ent, pred_ent)

        # afaik the += 1 sitaution has no need for now
        # but for situations in future where there are going to multiple
        # entities in the list.

        if is_this_entity_type_value_equal:
            # value matched here
            if true_ent_type in tp:
                tp[true_ent_type] += 1
            else:
                tp[true_ent_type] = 1
        else:
            # mismatch on value for this entity type.
            if true_ent_type in mm:
                mm[true_ent_type] += 1
            else:
                mm[true_ent_type] = 1
            
    else:
        # type didn't match here
        # true entity type should be predicted but it didn't happen, 
        #     therefore false negative for true entity type
        if true_ent_type in fn:
            fn[true_ent_type] += 1
        else:
            fn[true_ent_type] = 1

        # unexpected entity type got predicted happened,
        #     therefore false positive for predicted entity type
        if pred_ent_type in fp:
            fp[pred_ent_type] += 1
        else:
            fp[pred_ent_type] = 1

    ecr = EntityComparisonResult(tp=tp, fp=fp, fn=fn, mm=mm)
    return ecr


def get_entity_df_by_ecr(df: pd.DataFrame, entity_type: str):
    """
    when you have column of EntityComparisonResult,
    but you need all the entity rows which have this particular
    `entity_type` in EntityComparisonResult's tp/fp/fn/mm.

    therefore finds the index of rows where it happens
    and returns that particular selection of df for this
    given `entity_type`
    """

    entity_idxs = []

    for idx, row in df.iterrows():

        ecr = row["entity_comp_results"]

        if ecr is None:
            continue

        if any([entity_type in ecr_d for ecr_d in [ecr.tp, ecr.fn, ecr.fp, ecr.mm]]):
            entity_idxs.append(idx)

    return df.loc[df.index[entity_idxs]]


def categorical_entity_report(true_labels: pd.DataFrame, pred_labels: pd.DataFrame) -> Optional[pd.DataFrame]:

    df = pd.merge(true_labels, pred_labels, on="id", how="inner")
    df["true"] = df["entities_x"].apply(lambda it: json.loads(it) if isinstance(it, str) else None)
    df["pred"] = df["entities_y"].apply(lambda it: json.loads(it) if isinstance(it, str) else None)

    # assuming there will be only one entity type and value 
    df["true_ent_type"] = df["true"].apply(lambda it: it[0].get("type") if it else None)
    df["pred_ent_type"] = df["pred"].apply(lambda it: it[0].get("type") if it else None)

    df.reset_index(inplace=True)

    # we don't want classification report on standard entities like: date, time, number, people etc
    # we want it only on other entities.
    to_be_filtered = list(ENTITY_EQ_FNS.keys()) + ["datetime"]

    entity_types = sorted(set([ent["type"] for ent in py_.flatten(df["true"].dropna().tolist() + df["pred"].dropna().tolist())]))
    filtered_entity_types = sorted(list(set(entity_types) - set(to_be_filtered)))

    # including NaN here, because we want NaN vs other-entity/NaN for comparison as well.
    filtered_entity_df = df[df["true_ent_type"].isin(filtered_entity_types) | df["true_ent_type"].isna()]
    
    if not filtered_entity_df.empty:

        y_true = []
        y_pred = []

        for _, row in filtered_entity_df.iterrows():

            true_ent = None if row["true"] is None else row["true"][0]
            pred_ent = None if row["pred"] is None else row["pred"][0]

            if true_ent:
                true_ent_type = true_ent["type"] # eg: product_kind
                true_ent_value = true_ent["values"][0]["value"] # eg: credit_card

                # we don't want to include entity types like `duration`, ordinal etc.
                # that is why we imposing rule for them to custom entities which are called
                # categorical.
                if true_ent["values"][0]["type"] != "categorical":
                    continue
                else:
                    true_ent_value_mod = f"{true_ent_type}/{true_ent_value}" # eg: product_kind/credit_card
                    y_true.append(true_ent_value_mod)

            else:
                y_true.append(np.nan) # for None/no_entity present


            if pred_ent:
                pred_ent_type = pred_ent["type"]
                pred_ent_value = pred_ent["values"][0]["value"]

                if pred_ent["values"][0]["type"] == "categorical":
                    pred_ent_value_mod = f"{pred_ent_type}/{pred_ent_value}"
                    y_pred.append(pred_ent_value_mod)
                else:
                    y_pred.append(pred_ent_value)

            else:
                y_pred.append(np.nan) # for None/no_entity being predicted


    if y_true and y_pred:
        cls_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        # nan is being replaced with `_`
        if "nan" in cls_report:
            cls_report["_"] = cls_report.pop("nan")

        # we don't want "accuracy", "macro avg", "weighted avg"
        entries_to_remove = ("accuracy", "macro avg", "weighted avg")
        for entry_to_remove in entries_to_remove:
            cls_report.pop(entry_to_remove)

        cat_report_df = pd.DataFrame(cls_report).transpose()
        cat_report_df["support"] = cat_report_df["support"].astype(int)
        cat_report_df = cat_report_df[cat_report_df["support"] > 0]
        cat_report_df.sort_index(inplace=True)
        cat_report_df.index.name = "Categorical Entity"
        cat_report_df = weighted_avg_dropna(cat_report_df)

        return cat_report_df


def entity_report(true_labels: pd.DataFrame, pred_labels: pd.DataFrame, dump=False) -> pd.DataFrame:
    """
    given a true entity dataframe
    along with pred entity dataframe, we can get
    the False Positive Rate, False Negataive Rate, Mismatch Rate
    for each of the entity types mentioned
    in truth/prediction.
    """


    df = pd.merge(true_labels, pred_labels, on="id", how="inner")
    df["true"] = df["entities_x"].apply(lambda it: json.loads(it) if isinstance(it, str) else None)
    df["pred"] = df["entities_y"].apply(lambda it: json.loads(it) if isinstance(it, str) else None)

    # assuming there will be only one entity type and value 
    df["true_ent_type"] = df["true"].apply(lambda it: it[0].get("type") if it else None)
    df["pred_ent_type"] = df["pred"].apply(lambda it: it[0].get("type") if it else None)

    df.reset_index(inplace=True)

    # All the unique entity types in the dataset
    entity_types = sorted(set([ent["type"] for ent in py_.flatten(df["true"].dropna().tolist() + df["pred"].dropna().tolist())]))

    report = []

    df["entity_comp_results"] = df.apply(compare_row_level_entities, axis=1)

    dt_filtered_entity_types = list(filter(lambda x: x!="datetime", entity_types))

    # for dump
    fp_error_idxs = []
    fn_error_idxs = []
    mm_errror_idxs = []

    for entity_type in dt_filtered_entity_types:

        # entity type df needs to consider `entity_type` in tp, fp, fn, mm in "entity_comp_results"
        entity_type_df = get_entity_df_by_ecr(df, entity_type)

        # entity_support: entity's support refers to sitatuion where entity_type present in
        # the true_labels dataframe.
        entity_support = 0

        # entity_fp : entity_type prediction happened unexpectedly, 
        # i.e., truth didn't have same
        # entity type as prediction for that particular `id`
        entity_fp = 0

        # entity_fn: entity_type prediction didn't happen even though it was expected
        # i.e., prediction didn't have same
        # entity type as truth for that particular `id`
        entity_fn = 0

        # entity_tp: true and prediction entity type match
        # and their values match as well.
        entity_tp = 0

        # entity_mm: true and prediction entity type match
        # but their values don't match.
        entity_mm = 0

        for entity_row_idx, row in entity_type_df.iterrows():

            # situation of true negative (for entire df) where both true and prediction
            # entity types are of `NaN` and `NaN` => no tagging and no predictions
            # for that particular turn id, which results in ecr being None.
            if row["entity_comp_results"] is None:
                continue
            else:
                ecr = row["entity_comp_results"]

            if (
                (row["true_ent_type"] == entity_type) or # for ordinary support
                # (ENTITY_EQ_ALIAS.get(row["true_ent_type"]) == entity_type) or # for number vs people
                (entity_type in ecr.tp or entity_type in ecr.fn or entity_type in ecr.mm) # for datetime, date, time mess
            ):
                entity_support += 1

            if entity_type in ecr.tp:
                entity_tp += 1
            
            # false negative, we expected a prediction but didn't happen.
            if entity_type in ecr.fn:
                entity_fn += 1
                fn_error_idxs.append(entity_row_idx)

            # false positive, no prediction should have happened
            if entity_type in ecr.fp:
                entity_fp += 1
                fp_error_idxs.append(entity_row_idx)

            if entity_type in ecr.mm:
                entity_mm += 1
                mm_errror_idxs.append(entity_row_idx)

        # we are trying to find true negatives for this particular entity type
        # true negatives of entity type = remaining rows which don't have
        # entity_type in true.
        entity_neg = df.shape[0] - entity_support

        # fpr is defined as := fp/negatives
        if entity_neg == 0:
            entity_fpr = 0.0
        else:
            entity_fpr = entity_fp / entity_neg

        # fnr is defined as := fn / (fn + tp +mm)
        # not to be confused with sklearn's fn / (fn + tp)
        # since eevee's tp != sklearn's tp
        if (entity_fn + entity_tp + entity_mm) == 0:
            entity_fnr = 0.0
        else:
            entity_fnr = entity_fn / (entity_fn + entity_tp + entity_mm)
        
        # mm is defined as := mm / (tp + mm)
        # rate of type-matched-but-value-mismatched for this entity
        if (entity_tp + entity_mm) == 0:
            entity_mmr = 0.0
        else:
            entity_mmr = entity_mm / (entity_tp + entity_mm)

        report.append({
            "Entity": entity_type,
            "FPR": entity_fpr,
            "FNR": entity_fnr,
            "Mismatch Rate": entity_mmr,
            "Support": entity_support,
            "Negatives": entity_neg
        })

    report = pd.DataFrame(report)

    if not report.empty:
        report.set_index("Entity", inplace=True)

        # dumps the .csv files for fp, fn, mm for deeper analysis.
        if dump:
            dump_error_reports(df, fp_error_idxs, fn_error_idxs, mm_errror_idxs)

    return report