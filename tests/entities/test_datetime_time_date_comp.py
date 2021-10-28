import pytest
import pandas as pd

from eevee.ord.entity.datetime import date_eq, time_eq
from eevee.metrics.entity import EntityComparisonResult, compare_datetime_special_entities, compare_row_level_entities


@pytest.mark.parametrize(
    # truth and pred are Entity/Dict, same is boolean
    "truth, pred, same",
    [
        (
            {'type': 'date', 'value': '2019-04-21T00:00:00+05:30'}, 
            {'type': 'date', 'value': '2019-04-21T00:00:00+05:30'}, 
            True,
        ),
        (
            {'type': 'datetime', 'value': '2019-04-21T00:00:00+05:30', }, 
            {'type': 'date', 'value': '2019-04-21T00:00:00+05:30', }, 
            True,
        ),
        (
            {'type': 'datetime', 'value': '2019-04-21T00:00:00+05:30'}, 
            {'type': 'date', 'value': '2019-04-21T09:00:00+05:30'}, 
            True,
        ),
        (
            {'type': 'date', 'value': '2019-04-21T00:00:00+05:30'}, 
            {'type': 'datetime', 'value': '2019-04-21T00:00:00+05:30'}, 
            True,
        ),
        (
            {'type': 'date', 'value': '2019-04-21T09:00:00+05:30'}, 
            {'type': 'datetime', 'value': '2019-04-21T00:00:00+05:30'}, 
            True,
        ),
        (
            {'type': 'datetime', 'value': '2019-04-22T00:00:00+05:30'}, 
            {'type': 'date', 'value': '2019-04-21T00:00:00+05:30'}, 
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
            {'type': 'time', 'value': '2019-04-21T00:11:00+05:30'}, 
            {'type': 'time', 'value': '2019-04-17T00:11:00+05:30'}, 
            True,
        ),
        (
            {'type': 'datetime', 'value': '2019-04-21T00:00:00+05:30'}, 
            {'type': 'time', 'value': '2019-04-21T00:00:00+05:30'}, 
            True,
        ),
        (
            {'type': 'datetime', 'value': '2019-04-21T00:00:00+05:30'}, 
            {'type': 'time', 'value': '2019-04-21T09:00:00+05:30'}, 
            False,
        ),
        (
            {'type': 'time', 'value': '2019-04-21T00:00:00+05:30', }, 
            {'type': 'datetime', 'value': '2019-04-21T00:00:00+05:30', }, 
            True,
        ),
        (
            {'type': 'time', 'value': '2019-04-21T09:00:00+05:30', }, 
            {'type': 'datetime', 'value': '2019-04-21T00:00:00+05:30', }, 
            False,
        ),
        (
            {'type': 'datetime', 'value': '2019-04-22T00:00:00+05:30', }, 
            {'type': 'time', 'value': '2019-04-21T00:00:00+05:30', }, 
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
                [{'type': 'date', 'value': '2019-04-21T00:00:00+05:30'}],
                [{'type': 'datetime', 'value': '2019-04-21T17:00:00+05:30'}],
                "date",
                "datetime",
            ]], columns=["true", "pred", "true_ent_type", "pred_ent_type"]).iloc[0]
            , 
            EntityComparisonResult(tp={"date": 1}, fp={"time": 1}, fn={}, mm={}),
        ),

        # date-datetime, where date type should match, but their value shouldn't
        (
            pd.DataFrame([[
                [{'type': 'date', 'value': '2019-04-30T00:00:00+05:30', }],
                [{'type': 'datetime', 'value': '2019-04-21T00:00:00+05:30', }],
                "date",
                "datetime",
            ]], columns=["true", "pred", "true_ent_type", "pred_ent_type"]).iloc[0]
            , 
            EntityComparisonResult(tp={}, fp={"time": 1}, fn={}, mm={"date": 1}),
        ),

        # time-datetime, where time type & value should match
        (
            pd.DataFrame([[
                [{'type': 'time', 'value': '2019-04-21T17:00:00+05:30', }],
                [{'type': 'datetime', 'value': '2019-04-21T17:00:00+05:30', }],
                "time",
                "datetime",
            ]], columns=["true", "pred", "true_ent_type", "pred_ent_type"]).iloc[0]
            , 
            EntityComparisonResult(tp={"time": 1}, fp={"date": 1}, fn={}, mm={}),
        ),

        # time-datetime, where time type should match, but their value shouldn't
        (
            pd.DataFrame([[
                [{'type': 'time', 'value': '2019-04-21T00:00:00+05:30', }],
                [{'type': 'datetime', 'value': '2019-04-21T17:00:00+05:30', }],
                "time",
                "datetime",
            ]], columns=["true", "pred", "true_ent_type", "pred_ent_type"]).iloc[0]
            , 
            EntityComparisonResult(tp={}, fp={"date": 1}, fn={}, mm={"time": 1}),
        ),


        # datetime-None
        (
            pd.DataFrame([[
                [{'type': 'datetime', 'value': '2019-04-21T17:00:00+05:30', }],
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
                [{'type': 'datetime', 'value': '2019-04-21T17:00:00+05:30', }],
                [{'type': 'time', 'value': '2019-04-21T17:00:00+05:30', }],
                "datetime",
                "time",
            ]], columns=["true", "pred", "true_ent_type", "pred_ent_type"]).iloc[0]
            , 
            EntityComparisonResult(tp={"time": 1}, fp={}, fn={"date": 1}, mm={}),
        ),

        # datetime-time, time should mismatch
        (
            pd.DataFrame([[
                [{'type': 'datetime', 'value': '2019-04-21T17:00:00+05:30', }],
                [{'type': 'time', 'value': '2019-04-21T00:00:00+05:30', }],
                "datetime",
                "time",
            ]], columns=["true", "pred", "true_ent_type", "pred_ent_type"]).iloc[0]
            , 
            EntityComparisonResult(tp={}, fp={}, fn={"date": 1}, mm={"time": 1}),
        ),

        # datetime-date, date should match
        (
            pd.DataFrame([[
                [{'type': 'datetime', 'value': '2019-04-21T17:00:00+05:30', }],
                [{'type': 'date', 'value': '2019-04-21T00:00:00+05:30', }],
                "datetime",
                "date",
            ]], columns=["true", "pred", "true_ent_type", "pred_ent_type"]).iloc[0]
            , 
            EntityComparisonResult(tp={"date": 1}, fp={}, fn={"time": 1}, mm={}),
        ),

        # datetime-date, date should mismatch
        (
            pd.DataFrame([[
                [{'type': 'datetime', 'value': '2019-04-22T17:00:00+05:30', }],
                [{'type': 'date', 'value': '2019-04-21T00:00:00+05:30', }],
                "datetime",
                "date",
            ]], columns=["true", "pred", "true_ent_type", "pred_ent_type"]).iloc[0]
            , 
            EntityComparisonResult(tp={}, fp={}, fn={"time": 1}, mm={"date": 1}),
        ),

        # number-datetime
        (
            pd.DataFrame([[
                [{'type': 'number', 'value': 4, }],
                [{'type': 'datetime', 'value': '2019-04-22T17:00:00+05:30', }],
                "number",
                "datetime",
            ]], columns=["true", "pred", "true_ent_type", "pred_ent_type"]).iloc[0]
            , 
            EntityComparisonResult(tp={}, fp={"date": 1, "time": 1}, fn={"number": 1}, mm={}),
        ),

        # datetime-number
        (
            pd.DataFrame([[
                [{'type': 'datetime', 'value': '2019-04-22T17:00:00+05:30', }],
                [{'type': 'number', 'value': 4, }],
                "datetime",
                "number",
            ]], columns=["true", "pred", "true_ent_type", "pred_ent_type"]).iloc[0]
            , 
            EntityComparisonResult(tp={}, fp={"number": 1}, fn={"date": 1, "time": 1}, mm={}),
        ),

    ],
)
def test_datetime_time_pd_eq(row, ecr):
    assert compare_datetime_special_entities(row) == ecr



@pytest.mark.parametrize(
    "row, ecr",
    [
        # datetime-datetime, datetime should match
        (
            pd.DataFrame([[
                [{'type': 'datetime', 'value': '2019-04-21T17:00:00+05:30', }],
                [{'type': 'datetime', 'value': '2019-04-21T17:00:00+05:30', }],
                "datetime",
                "datetime",
            ]], columns=["true", "pred", "true_ent_type", "pred_ent_type"]).iloc[0]
            , 
            EntityComparisonResult(tp={"date": 1, "time": 1}, fp={}, fn={}, mm={}),
        ),

        # datetime-datetime, datetime should mismatch
        (
            pd.DataFrame([[
                [{'type': 'datetime', 'value': '2019-04-22T17:00:00+05:30', }],
                [{'type': 'datetime', 'value': '2019-04-21T12:00:00+05:30', }],
                "datetime",
                "datetime",
            ]], columns=["true", "pred", "true_ent_type", "pred_ent_type"]).iloc[0]
            , 
            EntityComparisonResult(tp={}, fp={}, fn={}, mm={"date": 1, "time": 1}),
        ),

        # datetime-datetime, datetime should mismatch
        (
            pd.DataFrame([[
                [{'type': 'datetime', 'value': '2019-04-22T17:00:00+05:30', }],
                [{'type': 'datetime', 'value': '2019-04-22T12:00:00+05:30', }],
                "datetime",
                "datetime",
            ]], columns=["true", "pred", "true_ent_type", "pred_ent_type"]).iloc[0]
            , 
            EntityComparisonResult(tp={"date": 1}, fp={}, fn={}, mm={"time": 1}),
        ),

        # datetime-datetime, datetime should mismatch
        (
            pd.DataFrame([[
                [{'type': 'datetime', 'value': '2019-04-22T17:00:00+05:30', }],
                [{'type': 'datetime', 'value': '2019-04-21T17:00:00+05:30', }],
                "datetime",
                "datetime",
            ]], columns=["true", "pred", "true_ent_type", "pred_ent_type"]).iloc[0]
            , 
            EntityComparisonResult(tp={"time": 1}, fp={}, fn={}, mm={"date": 1}),
        ),
    ],
)
def test_datetime_datetime_pd_eq(row, ecr):
    assert compare_row_level_entities(row) == ecr




@pytest.mark.parametrize(
    "truth, pred, same",
    [
        # checks only the time interval 18:00 to 00:00 not day/year.
        (
        {'text': '24th July night',
        'type': 'time',
        'value': {'from': {'value': '2021-07-24T18:00:00.000-07:00'},
                    'to': {'value': '2021-07-25T00:00:00.000-07:00'}}},
        {'text': '24th July night',
        'type': 'time',
        'value': {'from': {'value': '2021-07-24T18:00:00.000-07:00'},
                    'to': {'value': '2021-07-25T00:00:00.000-07:00'}}},
        True,
        ),
        # a time given and interval given for time, doesn't work right now.
        (
        {"text": "12pm", "type": "time", 'value': '2021-07-22T12:00:00.000+05:30'},
        {'text': 'twelve pm to eleven am',
        'type': 'time',
        "value": {"from": {"value": "2021-07-22T12:00:00.000+05:30"}, 
                "to": {"value": "2021-07-23T12:00:00.000+05:30"}}},
        False,
        ),
        # TODO: handle ambiguous intervals. with only `from` or `to`
        # a time given and interval given for time, doesn't work right now.
        (
        # truth
        {"text": "12pm", 
        "type": "time", 
        'value': '2021-07-22T12:00:00.000+05:30'},

        # pred
        {'text': 'twelve pm to eleven am',
        'type': 'time',
        "value": {"from": {"value": "2021-07-22T00:00:00.000+05:30"}}},
        False,
        ),
        # a time given and interval given for time, doesn't work right now.
        (
        # truth
        {"text": "12pm", 
        "type": "time", 
        'value': '2021-07-22T12:00:00.000+05:30'},

        # pred
        {'text': 'twelve pm to eleven am',
        'type': 'time',
        "value": {"to": {"value": "2021-07-22T11:00:00.000+05:30"}}},
        False,
        ),
    ],
)
def test_interval_eq(truth, pred, same):
    assert time_eq(truth, pred) == same
