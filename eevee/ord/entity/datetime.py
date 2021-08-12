from typing import List, Optional

from pydash import py_

from eevee.types import Entity
from eevee.ord.utils import (merge_date_and_time_entities,
                                  parse_datetime_objects)


def datetime_eq(a: Entity, b: Entity) -> bool:
    return [v["value"] for v in a["values"]] == [v["value"] for v in b["values"]]


def datetime_neq_strict(a: Entity, b: Entity) -> bool:
    """
    Strict non-equality for datetime only when date and time both mismatch.

    This is needed separately since the main datetime_eq function does an `or`
    match instead of an `and`.
    """

    if date_eq(a, b):
        return False

    if time_eq(a, b):
        return False

    return True


def datetime_neq_strict_lists(truth: List[Entity], pred: List[Entity]) -> bool:
    """
    Tell if datetime lists are strictly _not_ matching.

    TODO: Make this a partial comparison.
    """

    if pred and (pred[0]["type"] == "datetime"):
        datetime_entity: Optional[Entity] = pred[0]
    else:
        # We try to see if we can compose first `date` and `time` entities in a
        # single `datetime`
        time_entity = py_.find(pred, lambda it: it["type"] == "time")
        date_entity = py_.find(pred, lambda it: it["type"] == "date")
        if time_entity and date_entity:
            datetime_entity = merge_date_and_time_entities(time_entity, date_entity)
        else:
            datetime_entity = None

    if truth:
        return bool(not datetime_entity or datetime_neq_strict(truth[0], datetime_entity))
    else:
        return bool(datetime_entity)


def datetime_eq_lists(truth: List[Entity], pred: List[Entity]) -> bool:
    """
    TODO: Make this a partial comparison
    """

    if pred and (pred[0]["type"] == "datetime"):
        datetime_entity: Optional[Entity] = pred[0]
    else:
        # We try to see if we can compose first `date` and `time` entities in a
        # single `datetime`
        time_entity = py_.find(pred, lambda it: it["type"] == "time")
        date_entity = py_.find(pred, lambda it: it["type"] == "date")
        if time_entity and date_entity:
            datetime_entity = merge_date_and_time_entities(time_entity, date_entity)
        else:
            datetime_entity = None

    if truth:
        return bool(datetime_entity and datetime_eq(truth[0], datetime_entity))
    else:
        return not datetime_entity


def date_eq(truth: Entity, pred: Entity) -> bool:
    """
    We assume at least one date present in truth and look for similar set of
    dates in pred.
    """

    true_dates = parse_datetime_objects(truth, to_date=True)
    pred_dates = parse_datetime_objects(pred, to_date=True)

    if len(true_dates) != len(pred_dates):
        return False

    def _same_date(x, y) -> Optional[bool]:
        if x and y:
            if x == y:
                return x
            else:
                return None
        else:
            return x or y

    def _date_interval_match(xd, ydi) -> bool:
        yd = _same_date(*ydi)
        if yd:
            return xd == yd
        else:
            # TODO: Here we might need to understand the severity of the issue.
            #       To be safe we return False
            return False

    def _match(td, pd) -> bool:
        if isinstance(td, tuple):
            if isinstance(pd, tuple):
                return td == pd
            else:
                return _date_interval_match(pd, td)
        else:
            if isinstance(pd, tuple):
                return _date_interval_match(td, pd)
            else:
                return td == pd

    for td, pd in zip(true_dates, pred_dates):
        if not _match(td, pd):
            return False

    return True


def date_eq_lists(truth: List[Entity], pred: List[Entity]) -> bool:
    """
    Compare date parts of truth and prediction. Truth is supposed to have one
    thing and so.
    """

    date_entity = py_.find(pred, lambda it: it["type"] in ["datetime", "date"])
    if truth:
        return date_entity and date_eq(truth[0], date_entity)
    else:
        return not date_entity


def time_eq(truth: Entity, pred: Entity) -> bool:
    """
    We assume at least one time present in truth and look for similar set of
    dates in pred.
    """

    true_times = parse_datetime_objects(truth, to_time=True)
    pred_times = parse_datetime_objects(pred, to_time=True)
    return true_times == pred_times


def time_eq_lists(truth: List[Entity], pred: List[Entity]) -> bool:
    """
    Full comparison between truth and pred.
    Here we compare the interval and values separately.
    """

    def _extract_time_objects(ents: List[Entity]):
        values = []
        intervals = []
        for ent in ents:
            if ent["type"] in ["datetime", "time"]:
                if ent["values"][0]["type"] == "value":
                    values.extend(parse_datetime_objects(ent, to_time=True))
                elif ent["values"][0]["type"] == "interval":
                    intervals.extend(parse_datetime_objects(ent, to_time=True))
                else:
                    raise NotImplementedError

        return list(set(values)), list(set(intervals))

    truth_values, truth_intervals = _extract_time_objects(truth)
    pred_values, pred_intervals = _extract_time_objects(pred)

    if truth_values or truth_intervals:
        if len(pred_values) != len(truth_values):
            return False

        if len(pred_intervals) != len(truth_intervals):
            return False

        return all(p_val in truth_values for p_val in pred_values) and all(p_interval in truth_intervals for p_interval in pred_intervals)
    else:
        return not pred_values and not pred_intervals


def time_superset_list(superset: List[Entity], subset: List[Entity]) -> bool:
    """
    Tell if all time entities in subset is present in superset.
    """
    subset_time_entities = [ent for ent in subset if ent["type"] in ["datetime", "time"]]
    if superset:
        return all(any([time_eq(sb_ent, sp_ent) for sp_ent in superset]) for sb_ent in subset_time_entities)
    else:
        return not subset_time_entities
