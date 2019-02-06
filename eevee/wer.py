"""
WER evaluation functions
"""

from typing import Dict, List


def wwer(truth: List[str], pred: List[str], weights: Dict[str, float] = None) -> float:
    if weights:
        raise NotImplementedError()
    else:
        raise NotImplementedError()
