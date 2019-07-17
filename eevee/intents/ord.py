"""
Ordering, comparison and matching functions for intents
"""

from typing import List, Dict
from eevee.utils import match_dict


def eq(a: Dict, b: Dict, name_only=True) -> bool:
    """
    Check if intents match
    """

    if name_only:
        return a["name"] == b["name"]
    else:
        return match_dict(a, b)


def eq_lists(truth: List[Dict], pred: List[Dict], name_only=True) -> bool:
    """
    Compare truth and predictions. There are a lot of assumptions here:

    1. We assume _ood_ is the default fallback intent. This is semantically
       incorrect and might change later.
    2. In case of multiple true intents, we match the exact ordering too.
       This is not exactly correct since we don't really take any intent
       other the first one on fsm side. For now, when we do grouped matching,
       i.e. compare smalltalk, main intents etc. separately, we will be
       getting (mostly) single intent in truth list so things might be okay.
       TODO: Probably the right way to evaluate here is to match first intent
             of each type.
    """

    if not truth:
        # An error here is misfire
        return not pred
    else:
        if len(truth) == 1:
            return bool(pred) and eq(truth[0], pred[0], name_only=name_only)
        elif len(truth) <= len(pred):
            return all(eq(t, p, name_only=name_only)
                       for t, p in zip(truth, pred))
        else:
            return False
