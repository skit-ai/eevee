from typing import List, Dict
import pandas as pd


def return_columns(columns: Dict):
    return columns["id"], columns["truth"], columns["predicted"]


def return_params(params: Dict):
    return params["error"], params["cutoff"]


def map_types(label) -> str:
    if label != "SPEECH":
        return "NON-SPEECH"
    return label


def is_captured(
    pred_seg: Dict, true_segments: List[Dict], error: float = 0.1, cutoff: float = 0.2
) -> int:
    pred_start = pred_seg["time-range"][0]
    for seg in true_segments:
        if (
            (pred_start < seg["time-range"][1])
            and (0 < pred_start - seg["time-range"][0] < error + cutoff)
            or (0 < seg["time-range"][0] - pred_start < error)
        ):
            return 1
    return 0


def barge_in_report(
    true_labels: pd.DataFrame, pred_labels: pd.DataFrame, params: Dict
) -> Dict:

    # read params
    ID, TRUTH, PREDICTED = return_columns(params["data-columns"])
    ERROR, CUTOFF = return_params(params["barge-in"])

    # merhe truth and predictions
    data = pd.merge(true_labels, pred_labels, on=ID, how="outer")
    data[PREDICTED].fillna("[]", inplace=True)

    # collect SPEECH segments
    data["truth-speech"] = data[TRUTH].apply(
        lambda tag: [
            seg
            for seg in tag
            if map_types(seg["type"]) == "SPEECH"
            and seg["time-range"][1] - seg["time-range"][0] > ERROR
        ]
    )
    data["predicted-speech"] = data[PREDICTED].apply(
        lambda tag: [seg for seg in tag if map_types(seg["type"]) == "SPEECH"]
    )

    # calculate metrics
    data["speech-captured"] = data.apply(
        lambda row: [
            is_captured(pred_seg, row["truth-speech"], error=ERROR, cutoff=CUTOFF)
            for pred_seg in row["predicted-speech"]
        ],
        axis=1,
    )
    captures = (
        data["speech-captured"]
        .apply(lambda captures: len([x for x in captures if x == 1]))
        .sum()
    )
    precision = captures / data["predicted-speech"].sum()
    recall = captures / data["truth-speech"].sum()

    return {"precision": precision, "recall": recall}
