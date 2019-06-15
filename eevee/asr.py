"""
WER evaluation functions
"""

from typing import List

from eevee.levenshtein import levenshtein


def wer(ref_tokens: List[str], hyp_tokens: List[str]) -> int:
    """
    Raw Word Error Rate
    """

    if ref_tokens:
        return levenshtein(ref_tokens, hyp_tokens) / len(ref_tokens)
    else:
        raise RuntimeError("Empty reference sentence")


def per(ref_ps: List[List[str]], hyp_ps: List[List[str]]) -> int:
    """
    Phoneme Error Rate. Inputs are lists of pronunciations (represented as list
    of phonemes). Error which is minimum across the cross join between
    hypothesis and reference is reported. Note that it's easy to fool this by
    passing an exhaustive list of pronunciations in hypothesis and get low PER.
    As of yet, we don't worry about such situations.
    """

    raise NotImplementedError()
