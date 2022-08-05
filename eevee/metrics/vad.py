from typing import List, Dict
import pandas as pd
import ast


def return_columns(columns: Dict):
    return columns["id"], columns["truth"], columns["predicted"]


def return_params(params: Dict):
    return params["error"], params["cutoff"]


def map_type(segment: Dict) -> str:
    if segment["type"] != "SPEECH":
        return "NON-SPEECH"
    return segment["type"]


def map_length(segment: Dict) -> float:
    return segment["time-range"][1] - segment["time-range"][0]


def is_captured(true_segment, pred_segment, error: float = 0.1, cutoff: float = 0.2):

    true_start, true_end = true_segment["time-range"]
    pred_start, pred_end = pred_segment["time-range"]

    return (true_start < pred_start) and (pred_start - (true_start - error) < cutoff)


def match_truth(
    true_seg: Dict, pred_segments: List[Dict], error: float = 0.1, cutoff: float = 0.2
) -> int:
    for pred_seg in pred_segments:
        if is_captured(true_seg, pred_seg, error=error, cutoff=cutoff):
            return 1
    return 0


def match_prediction(
    pred_seg: Dict, true_segments: List[Dict], error: float = 0.1, cutoff: float = 0.2
) -> int:
    for true_seg in true_segments:
        if is_captured(true_seg, pred_seg, error=error, cutoff=cutoff):
            return 1
    return 0


def barge_in_report(
    true_labels: pd.DataFrame, pred_labels: pd.DataFrame, params: Dict
) -> Dict:

    # read params
    ID, TRUTH, PREDICTED = return_columns(params["data-columns"])
    ERROR, CUTOFF = return_params(params["barge-in"])

    # merge truth and predictions. fill in empty predictions
    data = pd.merge(true_labels, pred_labels, on=ID, how="outer")
    data[PREDICTED].fillna("[]", inplace=True)

    # collect SPEECH segments, and keep a count
    data["truth-speech"] = data[TRUTH].apply(
        lambda tag: sorted(
            [
                seg
                for seg in ast.literal_eval(tag)
                if (map_type(seg) == "SPEECH")
                # and (map_length(seg) > ERROR)
            ],
            key=lambda seg: seg["time-range"][0],
        )
    )
    data["truth-speech-exists"] = data["truth-speech"].apply(
        lambda segments: 1 if len(segments) > 0 else 0
    )

    data["predicted-speech"] = data[PREDICTED].apply(
        lambda tag: sorted(
            [seg for seg in ast.literal_eval(tag) if (map_type(seg) == "SPEECH")],
            key=lambda seg: seg["time-range"][0],
        )
    )
    data["predicted-speech-exists"] = data["predicted-speech"].apply(
        lambda segments: 1 if len(segments) > 0 else 0
    )

    # calculate metrics
    data["truth-captured"] = data.apply(
        lambda row: [
            match_truth(seg, row["predicted-speech"], error=ERROR, cutoff=CUTOFF)
            for seg in row["truth-speech"]
        ],
        axis=1,
    )

    data["predicted-captured"] = data.apply(
        lambda row: [
            match_prediction(seg, row["truth-speech"], error=ERROR, cutoff=CUTOFF)
            for seg in row["predicted-speech"]
        ],
        axis=1,
    )

    recall = (
        data[data["truth-speech-exists"] == 1]["truth-captured"]
        .apply(lambda captures: 1 if 1 in captures else 0)
        .mean()
    )
    precision = (
        data[data["predicted-speech-exists"] == 1]["predicted-captured"]
        .apply(lambda captures: 1 if captures[0] == 1 else 0)
        .mean()
    )

    return {"precision": precision, "recall": recall}
