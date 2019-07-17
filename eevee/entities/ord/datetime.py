import datetime
from typing import List, Optional, Dict

import dateutil.parser
from pydash import py_


def parse_datetime_objects(ent: Dict, to_date=False, to_time=False) -> List:
    """
    Parse datetime type entity and return a list of datetimes or intervals.
    """

    def _parser(text: str):
        dt = dateutil.parser.parse(text)
        if to_date == to_time:
            return dt
        else:
            return dt.date() if to_date else dt.time()

    # NOTE: We assume there is no mixing of value type within an entity
    value_type = ent["values"][0]["type"]
    if value_type == "value":
        return [_parser(v["value"]) for v in ent["values"]]
    elif value_type == "interval":
        def _parse_interval_value(v):
            from_dt = _parser(v["value"]["from"]) if "from" in \
                v["value"] else None
            to_dt = _parser(v["value"]["to"]) if "to" in v["value"] else None
            return (from_dt, to_dt)

        return [_parse_interval_value(v) for v in ent["values"]]
    else:
        raise ValueError(f"Unknown type {value_type} for the truth")


def replace_date(iso_string: str, date: datetime.date) -> str:
    """
    Apply date in the iso_string and return new iso string
    """

    parsed = dateutil.parser.parse(iso_string)
    return parsed.replace(year=date.year, month=date.month,
                          day=date.day).isoformat()


def merge_time_and_date(time_ent: Dict, date_ent: Dict) -> Dict:
    """
    Attach date information in the time entity (note the order)
    """

    dates = parse_datetime_objects(date_ent, to_date=True)
    # We just pick the first valid date object
    if dates:
        date = dates[0][0] if isinstance(dates[0], tuple) else dates[0]
    else:
        raise RuntimeError(f"No dates found in {date}")

    value_type = time_ent["values"][0]["type"]
    if value_type == "value":
        # NOTE: We don't change grain etc. so don't use this for anything
        #       serious.
        return {
            **time_ent,
            "values": [
                {**v, "value": replace_date(v["value"], date)}
                for v in time_ent["values"]
            ]
        }
    elif value_type == "interval":
        return {
            **time_ent,
            "values": [
                {**v, "value": {
                    k: replace_date(v["value"][k], date)
                    for k in ["from", "to"] if k in v["value"]
                }} for v in time_ent["values"]
            ]
        }
    else:
        raise ValueError(f"Unknown type {value_type} for the truth")


def datetime_eq(a: Dict, b: Dict) -> bool:
    return [v["value"] for v in a["values"]] == \
        [v["value"] for v in b["values"]]


def datetime_eq_lists(truth: List[Dict], pred: List[Dict]) -> bool:
    """
    TODO: Make this a partial comparison
    """

    if pred and (pred[0]["type"] == "datetime"):
        datetime_entity: Optional[Dict] = pred[0]
    else:
        # We try to see if we can compose first `date` and `time` entities in a
        # single `datetime`
        time_entity = py_.find(pred, lambda it: it["type"] == "time")
        date_entity = py_.find(pred, lambda it: it["type"] == "date")
        if time_entity and date_entity:
            datetime_entity = merge_time_and_date(time_entity, date_entity)
        else:
            datetime_entity = None

    if truth:
        return bool(datetime_entity and datetime_eq(truth[0], pred[0]))
    else:
        return not datetime_entity


def date_eq(truth: Dict, pred: Dict) -> bool:
    """
    We assume at least one date present in truth and look for similar set of
    dates in pred.
    """

    true_dates = parse_datetime_objects(truth, to_date=True)
    pred_dates = parse_datetime_objects(pred, to_date=True)

    if len(true_dates) != len(pred_dates):
        return False

    for td, pd in zip(true_dates, pred_dates):
        if isinstance(pd, tuple) and isinstance(td, datetime.date):
            if pd[0] != td:
                return False
        else:
            if pd != td:
                return False

    return True


def date_eq_lists(truth: List[Dict], pred: List[Dict]) -> bool:
    """
    Compare date parts of truth and prediction. Truth is supposed to have one
    thing and so.
    """

    date_entity = py_.find(pred, lambda it: it["type"] in ["datetime", "date"])
    if truth:
        return date_entity and date_eq(truth[0], date_entity)
    else:
        return not date_entity


def time_eq(truth: Dict, pred: Dict) -> bool:
    """
    We assume at least one time present in truth and look for similar set of
    dates in pred.
    """

    true_times = parse_datetime_objects(truth, to_time=True)
    pred_times = parse_datetime_objects(pred, to_time=True)
    return true_times == pred_times


def time_eq_lists(truth: List[Dict], pred: List[Dict]) -> bool:
    """
    TODO: Make this a partial comparison
    """

    time_entity = py_.find(pred, lambda it: it["type"] in ["datetime", "time"])
    if truth:
        return time_entity and time_eq(truth[0], time_entity)
    else:
        return not time_entity
