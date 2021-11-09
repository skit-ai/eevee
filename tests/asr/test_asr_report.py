
import pandas as pd
from eevee.metrics.asr import asr_report


def test_asr_report():

    true_df = pd.read_csv("data/tagged.transcriptions.csv", usecols=["id", "transcription"])
    pred_df = pd.read_csv("data/predicted.transcriptions.csv", usecols=["id", "utterances"])

    expected = pd.DataFrame(
        {
            "Metric": ["WER", "Utterance FPR", "Utterance FNR"],
            "Value": [0.5714285714285715, 0.5, 0.25],
            "Support": [6, 2, 4],
        }
    )
    expected.set_index("Metric", inplace=True)

    given = asr_report(true_df, pred_df)

    assert given.to_dict("records") == expected.to_dict("records")
