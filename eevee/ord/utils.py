"""
General purpose utilities for working with entity types.
"""

from typing import List

import dateutil.parser

from eevee.types import Entity


def replace_date(iso_string: str, date) -> str:
    """
    Apply date in the iso_string and return new iso string.
    """

    parsed = dateutil.parser.parse(iso_string)
    return parsed.replace(year=date.year, month=date.month, day=date.day).isoformat()



def parse_datetime_objects(ent: Entity, to_date=False, to_time=False) -> List:
    """
    Parse datetime type entity and return a list of datetimes or intervals.
    """

    def _parser(text: str):
        dt = dateutil.parser.parse(text)
        if to_date == to_time:
            return dt
        else:
            return dt.date() if to_date else dt.time()

    def _is_midnight(text: str) -> bool:
        dt = dateutil.parser.parse(text).time()
        return dt.hour == dt.minute == 0

    # NOTE: We assume there is no mixing of value type within an entity

    if isinstance(ent["value"], dict) and ("from" in ent["value"] or "to" in ent["value"]):
        value_type = "interval"
    else:
        value_type = "value"

    if value_type == "value":
        return _parser(ent["value"])
    elif value_type == "interval":
        def _parse_interval_value(v):
            from_dt = _parser(v["from"]["value"]) if "from" in v else None
            to_dt = _parser(v["to"]["value"]) if "to" in v else None

            if to_date:
                # In case we are parsing a date range, we need to consider the fact
                # that parser gives to value for next day with time set to 00:00.
                if from_dt and to_dt and (to_dt - from_dt).days == 1 and _is_midnight(v["to"]["value"]):
                    return (from_dt, from_dt)
                else:
                    return (from_dt, to_dt)

            return (from_dt, to_dt)

        return _parse_interval_value(ent["value"])
    else:
        raise ValueError(f"Unknown type {value_type} for the truth")


def merge_date_and_time_entities(time_ent: Entity, date_ent: Entity) -> Entity:
    """
    Attach date information in the time entity (note the order)
    """

    dates = parse_datetime_objects(date_ent, to_date=True)
    # We just pick the first valid date object
    if dates:
        date = dates[0][0] if isinstance(dates[0], tuple) else dates[0]
    else:
        raise RuntimeError(f"No dates found in {dates}")

    value_type = time_ent["values"][0]["type"]
    if value_type == "value":
        # NOTE: We don't change grain etc. so don't use this for anything
        #       serious.
        return Entity({
            **time_ent,
            "values": [
                {**v, "value": replace_date(v["value"], date)}
                for v in time_ent["values"]
            ]
        })
    elif value_type == "interval":
        return Entity({
            **time_ent,
            "values": [
                {**v, "value": {
                    k: replace_date(v["value"][k], date)
                    for k in ["from", "to"] if k in v["value"]
                }} for v in time_ent["values"]
            ]
        })
    else:
        raise ValueError(f"Unknown type {value_type} for the truth")
