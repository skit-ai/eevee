import eevee.intents.ord as ord_intents
from typing import Any, List, Dict, Tuple
import pandas as pd
from collections import defaultdict

Id = Any
Truth = List[Any]
Pred = List[Any]


def collect_intent_errors(results: List[Tuple[Id, Truth, Pred]]) -> Dict:
    """
    Collect errors in intent predictions.
    """

    fp: Dict = defaultdict(list)
    fn: Dict = defaultdict(list)
    truecounts: Dict = defaultdict(int)

    for id, truth, pred in results:
        y_t = truth[0]["name"]
        y_p = pred[0]["name"]

        if y_p not in truecounts:
            truecounts[y_p] = 0
        truecounts[y_t] += 1

        # TODO: This is wrong when we consider multiple true intents.
        if not ord_intents.eq(truth[0], pred[0]):
            fp[y_p].append((id, truth, pred))
            fn[y_t].append((id, truth, pred))

    return {
        "fp": fp,
        "fn": fn,
        "truecounts": truecounts
    }


def intent_report(total: int, errors: Dict) -> pd.DataFrame:
    classes = sorted(list(errors["truecounts"].keys()))

    return pd.DataFrame({
        "class": classes,
        "fp": [len(errors["fp"][k]) for k in classes],
        "fn": [len(errors["fn"][k]) for k in classes],
        "truecounts": [
            f"{errors['truecounts'][k]} "
            f"({100 * errors['truecounts'][k] / total:.4f}%)"
            for k in classes
        ]
    })


def intent_difference_report(errors: Dict, prev_errors: Dict) -> pd.DataFrame:
    """
    TODO: Expand this when more metrics are available
    """

    return "NA"
