import pandas as pd
import pytest

from eevee.metrics import intent_report, intent_layers_report

@pytest.mark.parametrize(
    "y_true, y_pred, macro_f1",
    [
        (
            [{"id": 1, "intent_x": "a"}],
            [{"id": 1, "intent_y": "b"}, {"id": 2, "intent_y": "a"}],
            0.0,
        ),
        (
            [{"id": 1, "intent_x": "a"}],
            [{"id": 1, "intent_y": "a"}, {"id": 2, "intent_y": "a"}],
            1.0,
        ),
    ],
)
def test_intents(y_true, y_pred, macro_f1):
    true_labels = pd.DataFrame(y_true)
    pred_labels = pd.DataFrame(y_pred)

    report = intent_report(
        true_labels,
        pred_labels,
        return_output_as_dict=True,
    )

    assert report["macro avg"]["f1-score"] == macro_f1

@pytest.mark.parametrize(
    "true_df, pred_df, og_labels, aliased_labels",
    [
        (
            [
                {"id": 1, "intent": "a"},
                {"id": 2, "intent": "b"},
                {"id": 3, "intent": "c"},
                {"id": 4, "intent": "d"},
                {"id": 5, "intent": "e"},
                {"id": 6, "intent": "a"},
                {"id": 7, "intent": "a"},
            ],
            [
                {"id": 1, "intent": "a"},
                {"id": 2, "intent": "b"},
                {"id": 3, "intent": "c"},
                {"id": 4, "intent": "d"},
                {"id": 5, "intent": "e"},
                {"id": 6, "intent": "a"},
                {"id": 7, "intent": "a"},
            ],
            ["a", "b", "c", "d", "e"],
            ["z"],
        ),
    ],
)
def test_alias_yaml(true_df, pred_df, og_labels, aliased_labels):

    # tagged intents in the list get replaced with the key
    # main use case is when tag & predictions have different values and need to be
    # coerced

    columns = ["id", "intent"]
    true_df = pd.DataFrame(true_df, columns=columns)
    pred_df = pd.DataFrame(pred_df, columns=columns)

    intent_aliases = {
        'z': [
            'b',
            'c',
            'd',
            'e',
        ], 
    }

    out = intent_report(true_labels=true_df, pred_labels=pred_df, return_output_as_dict=True, intent_aliases=intent_aliases)

    assert set(out.keys()).intersection(set(og_labels)) == set(["a"]) 
    assert set(out.keys()).intersection(set(aliased_labels)) == set(aliased_labels)



@pytest.mark.parametrize(
    "true_df, pred_df, og_labels, grouped_labels",
    [
        (
            [
                {"id": 1, "intent": "a"},
                {"id": 2, "intent": "b"},
                {"id": 3, "intent": "c"},
                {"id": 4, "intent": "d"},
                {"id": 5, "intent": "e"},
                {"id": 6, "intent": "a"},
                {"id": 7, "intent": "a"},
            ],
            [
                {"id": 1, "intent": "a"},
                {"id": 2, "intent": "b"},
                {"id": 3, "intent": "c"},
                {"id": 4, "intent": "d"},
                {"id": 5, "intent": "e"},
                {"id": 6, "intent": "a"},
                {"id": 7, "intent": "a"},
            ],
            ["a", "b", "c", "d", "e"],
            ["z", "x"],
        ),
    ],
)
def test_groups_yaml(true_df, pred_df, og_labels, grouped_labels):

    # tagged intents in the list get replaced with appropriate group/category
    # intents not part of any group fall under "in_scope"

    # usage is to group intents (confirm_hi, confirm_bye all should be mapped to confirm etc)

    columns = ["id", "intent"]
    true_df = pd.DataFrame(true_df, columns=columns)
    pred_df = pd.DataFrame(pred_df, columns=columns)

    intent_groups = {
        'z': [
            'b',
            'c',
        ], 
        'x': [
            'a',
        ]
    }

    out : pd.DataFrame = intent_report(true_labels=true_df, pred_labels=pred_df, return_output_as_dict=True, intent_groups=intent_groups)

    # checking if a, b, c, d, e got replaced with z, x, in_scope
    assert set(out.index).intersection(set(og_labels)) == set()
    assert set(out.index).union(set(grouped_labels)) == set(grouped_labels + ["in_scope"])



@pytest.mark.parametrize(
    "true_df, pred_df, intent_layers, out_index",
    [
        (
            [
                {"id": 1, "intent": "audio_channel_noise"},
                {"id": 2, "intent": "audio_speech_unclear"},
                {"id": 3, "intent": "_oos_"},
                {"id": 4, "intent": "ood"},
                {"id": 5, "intent": "audio_silent"},
                {"id": 7, "intent": "background_noise"},
                {"id": 8, "intent": "audio_silent"},
                {"id": 9, "intent": "audio_channel_noise_hold"},
                {"id": 10, "intent": "partial"},
                {"id": 11, "intent": "background_noise"},
                {"id": 12, "intent": "other_language"},
            ],
            [
                {"id": 1, "intent": "oos"},
                {"id": 2, "intent": "oos"},
                {"id": 3, "intent": "whut"},
                {"id": 4, "intent": "no"},
                {"id": 5, "intent": "hi"},
                {"id": 6, "intent": "confirm"},
                {"id": 7, "intent": "nuke_america"},
                {"id": 8, "intent": "oos"},
                {"id": 9, "intent": "oos"},
                {"id": 10, "intent": "oos"},
                {"id": 11, "intent": "oos"},
            ],
            {
                'intent_x': {
                    'acoustic_oos': ['audio_channel_noise', 'audio_channel_noise_hold', 'audio_speech_unclear', 'audio_speech_volume', 'audio_silent', 'background_noise', 'background_speech', 'other_language', '_'], 
                    'lexical_oos': ['partial', 'ood', '_oos_']
                }, 
                'intent_y': {
                    'oos': ['oos', '_']
                }
            },
            ['layer-acoustic_oos', 'layer-lexical_oos', 'layer-oos']
        ),
    ],
)
def test_layers_yaml(true_df, pred_df, intent_layers, out_index):

    # this is a very specific problem, where ground truth information 
    # has nature of call like: audio_channel_noise, audio_silent, audio_speech_unclear etc
    # but predictions might give left & right whatever they want. 
    # we want to see if they are giving oos or something weird in predictions.
    # so cutting slack for mispredictions here.

    columns = ["id", "intent"]
    true_df = pd.DataFrame(true_df, columns=columns)
    pred_df = pd.DataFrame(pred_df, columns=columns)

    out : pd.DataFrame = intent_layers_report(true_labels=true_df, pred_labels=pred_df, intent_layers=intent_layers)

    assert set(out.index) == set(out_index)
