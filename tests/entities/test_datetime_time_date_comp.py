import pytest
import pandas as pd

from eevee.ord.entity.datetime import date_eq, time_eq
from eevee.metrics.entity import EntityComparisonResult, compare_datetime_special_entities, compare_row_level_entities


@pytest.mark.parametrize(
    # truth and pred are Entity/Dict, same is boolean
    "truth, pred, same",
    [
        (
            {'type': 'date', 'values': [{'value': '2019-04-21T00:00:00+05:30', 'type': 'value'}]}, 
            {'type': 'date', 'values': [{'value': '2019-04-21T00:00:00+05:30', 'type': 'value'}]}, 
            True,
        ),
        (
            {'type': 'datetime', 'values': [{'value': '2019-04-21T00:00:00+05:30', 'type': 'value'}]}, 
            {'type': 'date', 'values': [{'value': '2019-04-21T00:00:00+05:30', 'type': 'value'}]}, 
            True,
        ),
        (
            {'type': 'datetime', 'values': [{'value': '2019-04-21T00:00:00+05:30', 'type': 'value'}]}, 
            {'type': 'date', 'values': [{'value': '2019-04-21T09:00:00+05:30', 'type': 'value'}]}, 
            True,
        ),
        (
            {'type': 'date', 'values': [{'value': '2019-04-21T00:00:00+05:30', 'type': 'value'}]}, 
            {'type': 'datetime', 'values': [{'value': '2019-04-21T00:00:00+05:30', 'type': 'value'}]}, 
            True,
        ),
        (
            {'type': 'date', 'values': [{'value': '2019-04-21T09:00:00+05:30', 'type': 'value'}]}, 
            {'type': 'datetime', 'values': [{'value': '2019-04-21T00:00:00+05:30', 'type': 'value'}]}, 
            True,
        ),
        (
            {'type': 'datetime', 'values': [{'value': '2019-04-22T00:00:00+05:30', 'type': 'value'}]}, 
            {'type': 'date', 'values': [{'value': '2019-04-21T00:00:00+05:30', 'type': 'value'}]}, 
            False,
        ),

        # since 1 vs 3 length mismatch, we don't know which one is correct.
        # therefore putting in mismatch, definitely not the expected behavior.
        (
            {'type': 'date', 'values': [{'value': '2021-09-18T00:00:00+05:30', 'type': 'value'}]}, 
            {'type': 'date', 'values': [
                    {'value': '2021-09-18T00:00:00+05:30', 'type': 'value'},
                    {'value': '2022-09-18T00:00:00+05:30', 'type': 'value'},
                    {'value': '2023-09-18T00:00:00+05:30', 'type': 'value'}]}, 
            False,
        ),
    ],
)
def test_datetime_date_eq(truth, pred, same):
    assert date_eq(truth, pred) == same



@pytest.mark.parametrize(
    # truth and pred are Entity/Dict, same as boolean
    "truth, pred, same",
    [
        (
            {'type': 'time', 'values': [{'value': '2019-04-21T00:11:00+05:30', 'type': 'value'}]}, 
            {'type': 'time', 'values': [{'value': '2019-04-17T00:11:00+05:30', 'type': 'value'}]}, 
            True,
        ),
        (
            {'type': 'datetime', 'values': [{'value': '2019-04-21T00:00:00+05:30', 'type': 'value'}]}, 
            {'type': 'time', 'values': [{'value': '2019-04-21T00:00:00+05:30', 'type': 'value'}]}, 
            True,
        ),
        (
            {'type': 'datetime', 'values': [{'value': '2019-04-21T00:00:00+05:30', 'type': 'value'}]}, 
            {'type': 'time', 'values': [{'value': '2019-04-21T09:00:00+05:30', 'type': 'value'}]}, 
            False,
        ),
        (
            {'type': 'time', 'values': [{'value': '2019-04-21T00:00:00+05:30', 'type': 'value'}]}, 
            {'type': 'datetime', 'values': [{'value': '2019-04-21T00:00:00+05:30', 'type': 'value'}]}, 
            True,
        ),
        (
            {'type': 'time', 'values': [{'value': '2019-04-21T09:00:00+05:30', 'type': 'value'}]}, 
            {'type': 'datetime', 'values': [{'value': '2019-04-21T00:00:00+05:30', 'type': 'value'}]}, 
            False,
        ),
        (
            {'type': 'datetime', 'values': [{'value': '2019-04-22T00:00:00+05:30', 'type': 'value'}]}, 
            {'type': 'time', 'values': [{'value': '2019-04-21T00:00:00+05:30', 'type': 'value'}]}, 
            True,
        ),
    ],
)
def test_datetime_time_eq(truth, pred, same):
    assert time_eq(truth, pred) == same


@pytest.mark.parametrize(
    "row, ecr",
    [

        # date-datetime, where date type & value should match
        (
            pd.DataFrame([[
                [{'type': 'date', 'values': [{'value': '2019-04-21T00:00:00+05:30', 'type': 'value'}]}],
                [{'type': 'datetime', 'values': [{'value': '2019-04-21T17:00:00+05:30', 'type': 'value'}]}],
                "date",
                "datetime",
            ]], columns=["true", "pred", "true_ent_type", "pred_ent_type"]).iloc[0]
            , 
            EntityComparisonResult(tp={"date": 1}, fp={"time": 1}, fn={}, mm={}),
        ),

        # date-datetime, where date type should match, but their value shouldn't
        (
            pd.DataFrame([[
                [{'type': 'date', 'values': [{'value': '2019-04-30T00:00:00+05:30', 'type': 'value'}]}],
                [{'type': 'datetime', 'values': [{'value': '2019-04-21T00:00:00+05:30', 'type': 'value'}]}],
                "date",
                "datetime",
            ]], columns=["true", "pred", "true_ent_type", "pred_ent_type"]).iloc[0]
            , 
            EntityComparisonResult(tp={}, fp={"time": 1}, fn={}, mm={"date": 1}),
        ),

        # time-datetime, where time type & value should match
        (
            pd.DataFrame([[
                [{'type': 'time', 'values': [{'value': '2019-04-21T17:00:00+05:30', 'type': 'value'}]}],
                [{'type': 'datetime', 'values': [{'value': '2019-04-21T17:00:00+05:30', 'type': 'value'}]}],
                "time",
                "datetime",
            ]], columns=["true", "pred", "true_ent_type", "pred_ent_type"]).iloc[0]
            , 
            EntityComparisonResult(tp={"time": 1}, fp={"date": 1}, fn={}, mm={}),
        ),

        # time-datetime, where time type should match, but their value shouldn't
        (
            pd.DataFrame([[
                [{'type': 'time', 'values': [{'value': '2019-04-21T00:00:00+05:30', 'type': 'value'}]}],
                [{'type': 'datetime', 'values': [{'value': '2019-04-21T17:00:00+05:30', 'type': 'value'}]}],
                "time",
                "datetime",
            ]], columns=["true", "pred", "true_ent_type", "pred_ent_type"]).iloc[0]
            , 
            EntityComparisonResult(tp={}, fp={"date": 1}, fn={}, mm={"time": 1}),
        ),


        # datetime-None
        (
            pd.DataFrame([[
                [{'type': 'datetime', 'values': [{'value': '2019-04-21T17:00:00+05:30', 'type': 'value'}]}],
                None,
                "datetime",
                None,
            ]], columns=["true", "pred", "true_ent_type", "pred_ent_type"]).iloc[0]
            , 
            EntityComparisonResult(tp={}, fp={}, fn={"date": 1, "time": 1}, mm={}),
        ),

        # datetime-time, time should match
        (
            pd.DataFrame([[
                [{'type': 'datetime', 'values': [{'value': '2019-04-21T17:00:00+05:30', 'type': 'value'}]}],
                [{'type': 'time', 'values': [{'value': '2019-04-21T17:00:00+05:30', 'type': 'value'}]}],
                "datetime",
                "time",
            ]], columns=["true", "pred", "true_ent_type", "pred_ent_type"]).iloc[0]
            , 
            EntityComparisonResult(tp={"time": 1}, fp={}, fn={"date": 1}, mm={}),
        ),

        # datetime-time, time should mismatch
        (
            pd.DataFrame([[
                [{'type': 'datetime', 'values': [{'value': '2019-04-21T17:00:00+05:30', 'type': 'value'}]}],
                [{'type': 'time', 'values': [{'value': '2019-04-21T00:00:00+05:30', 'type': 'value'}]}],
                "datetime",
                "time",
            ]], columns=["true", "pred", "true_ent_type", "pred_ent_type"]).iloc[0]
            , 
            EntityComparisonResult(tp={}, fp={}, fn={"date": 1}, mm={"time": 1}),
        ),

        # datetime-date, date should match
        (
            pd.DataFrame([[
                [{'type': 'datetime', 'values': [{'value': '2019-04-21T17:00:00+05:30', 'type': 'value'}]}],
                [{'type': 'date', 'values': [{'value': '2019-04-21T00:00:00+05:30', 'type': 'value'}]}],
                "datetime",
                "date",
            ]], columns=["true", "pred", "true_ent_type", "pred_ent_type"]).iloc[0]
            , 
            EntityComparisonResult(tp={"date": 1}, fp={}, fn={"time": 1}, mm={}),
        ),

        # datetime-date, date should mismatch
        (
            pd.DataFrame([[
                [{'type': 'datetime', 'values': [{'value': '2019-04-22T17:00:00+05:30', 'type': 'value'}]}],
                [{'type': 'date', 'values': [{'value': '2019-04-21T00:00:00+05:30', 'type': 'value'}]}],
                "datetime",
                "date",
            ]], columns=["true", "pred", "true_ent_type", "pred_ent_type"]).iloc[0]
            , 
            EntityComparisonResult(tp={}, fp={}, fn={"time": 1}, mm={"date": 1}),
        ),

    ],
)
def test_datetime_time_eq(row, ecr):
    assert compare_datetime_special_entities(row) == ecr



@pytest.mark.parametrize(
    "row, ecr",
    [
        # datetime-datetime, datetime should match
        (
            pd.DataFrame([[
                [{'type': 'datetime', 'values': [{'value': '2019-04-21T17:00:00+05:30', 'type': 'value'}]}],
                [{'type': 'datetime', 'values': [{'value': '2019-04-21T17:00:00+05:30', 'type': 'value'}]}],
                "datetime",
                "datetime",
            ]], columns=["true", "pred", "true_ent_type", "pred_ent_type"]).iloc[0]
            , 
            EntityComparisonResult(tp={"date": 1, "time": 1}, fp={}, fn={}, mm={}),
        ),

        # datetime-datetime, datetime should mismatch
        (
            pd.DataFrame([[
                [{'type': 'datetime', 'values': [{'value': '2019-04-22T17:00:00+05:30', 'type': 'value'}]}],
                [{'type': 'datetime', 'values': [{'value': '2019-04-21T12:00:00+05:30', 'type': 'value'}]}],
                "datetime",
                "datetime",
            ]], columns=["true", "pred", "true_ent_type", "pred_ent_type"]).iloc[0]
            , 
            EntityComparisonResult(tp={}, fp={}, fn={}, mm={"date": 1, "time": 1}),
        ),

        # datetime-datetime, datetime should mismatch
        (
            pd.DataFrame([[
                [{'type': 'datetime', 'values': [{'value': '2019-04-22T17:00:00+05:30', 'type': 'value'}]}],
                [{'type': 'datetime', 'values': [{'value': '2019-04-22T12:00:00+05:30', 'type': 'value'}]}],
                "datetime",
                "datetime",
            ]], columns=["true", "pred", "true_ent_type", "pred_ent_type"]).iloc[0]
            , 
            EntityComparisonResult(tp={"date": 1}, fp={}, fn={}, mm={"time": 1}),
        ),

        # datetime-datetime, datetime should mismatch
        (
            pd.DataFrame([[
                [{'type': 'datetime', 'values': [{'value': '2019-04-22T17:00:00+05:30', 'type': 'value'}]}],
                [{'type': 'datetime', 'values': [{'value': '2019-04-21T17:00:00+05:30', 'type': 'value'}]}],
                "datetime",
                "datetime",
            ]], columns=["true", "pred", "true_ent_type", "pred_ent_type"]).iloc[0]
            , 
            EntityComparisonResult(tp={"time": 1}, fp={}, fn={}, mm={"date": 1}),
        ),

        # true date vs several predicted dates
        # even though truth is present in predicted, there still 1 vs 3, not sure
        # which one is correct.
        (
            pd.DataFrame([[

                # truth
                [{'type': 'date', 'values': [{'value': '2021-09-18T00:00:00+05:30', 'type': 'value'}]}],

                # multiple values predicted
                [{'type': 'date', 'values': [
                    {'value': '2021-09-18T00:00:00+05:30', 'type': 'value'},
                    {'value': '2022-09-18T00:00:00+05:30', 'type': 'value'},
                    {'value': '2023-09-18T00:00:00+05:30', 'type': 'value'}]}],
                "date",
                "date",
            ]], columns=["true", "pred", "true_ent_type", "pred_ent_type"]).iloc[0]
            , 
            EntityComparisonResult(tp={}, fp={}, fn={}, mm={"date": 1}),
        ),
    ],
)
def test_datetime_datetime_eq(row, ecr):
    assert compare_row_level_entities(row) == ecr




@pytest.mark.parametrize(
    "truth, pred, same",
    [
        # checks only the time interval 18:00 to 00:00 not day/year.
        (
        {'text': '24th July night',
        'type': 'time',
        'values': [{'type': 'interval',
                    'value': {'from': '2021-07-24T18:00:00.000-07:00',
                            'to': '2021-07-25T00:00:00.000-07:00'}}]},
        {'text': '24th July night',
        'type': 'time',
        'values': [{'type': 'interval',
                    'value': {'from': '2022-07-24T18:00:00.000-07:00',
                                'to': '2022-07-25T00:00:00.000-07:00'}}]},
        True,
        ),
        # a time given and interval given for time, doesn't work right now.
        (
        {"text": "12pm", 
        "type": "time", 
        'values': [{'type': 'value', 'value': '2021-07-22T12:00:00.000+05:30'}]},
        {'text': 'twelve pm to eleven am',
        'type': 'time',
        "values": [{"type": "interval", 
                    "value": {"from": "2021-07-22T12:00:00.000+05:30", 
                            "to": "2021-07-23T12:00:00.000+05:30"}}]},
        False,
        ),
        # TODO: handle ambiguous intervals. with only `from` or `to`
        # a time given and interval given for time, doesn't work right now.
        (
        # truth
        {"text": "12pm", 
        "type": "time", 
        'values': [{'type': 'value', 'value': '2021-07-22T12:00:00.000+05:30'}]},

        # pred
        {'text': 'twelve pm to eleven am',
        'type': 'time',
        "values": [{"type": "interval", 
                    "value": {"from": "2021-07-22T00:00:00.000+05:30", 
                            }}]},
        False,
        ),
        # a time given and interval given for time, doesn't work right now.
        (
        # truth
        {"text": "12pm", 
        "type": "time", 
        'values': [{'type': 'value', 'value': '2021-07-22T12:00:00.000+05:30'}]},

        # pred
        {'text': 'twelve pm to eleven am',
        'type': 'time',
        "values": [{"type": "interval", 
                    "value": {"to": "2021-07-22T11:00:00.000+05:30"}}]},
        False,
        ),
    ],
)
def test_interval_eq(truth, pred, same):
    assert time_eq(truth, pred) == same
