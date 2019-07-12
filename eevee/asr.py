"""
WER evaluation functions
"""

from itertools import product
from typing import List

from eevee.levenshtein import levenshtein


def wer(ref_tokens: List[str], hyp_tokens: List[str]) -> float:
    """
    Raw Word Error Rate
    """
    error_vector, error_detail = levenshtein(ref_tokens, hyp_tokens)
    cost_idx = 0
    if ref_tokens:
        return (error_vector[0]/len(ref_tokens), *error_vector[1:]), error_detail
    else:
        raise RuntimeError("Empty reference token list")


def per(ref_ps: List[List[str]], hyp_ps: List[List[str]]) -> float:
    """
    Phoneme Error Rate. Inputs are lists of pronunciations (represented as list
    of phonemes). Error which is minimum across the cross join between
    hypothesis and reference is reported. Note that it's easy to fool this by
    passing an exhaustive list of pronunciations in hypothesis and get low PER.
    As of yet, we don't worry about such situations.
    """

    errors = []
    for ref_phones, hyp_phones in product(ref_ps, hyp_ps):
        error_vector, _ = wer(ref_phones, hyp_phones)
        errors.append(error_vector[0])

    return min(errors)
