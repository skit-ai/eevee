import pytest

from eevee.asr import per
from eevee.levenshtein import levenshtein


# NOTE: This test keeps insertion, deletion and substitution information also
#       but for now we are only using the the total cost
@pytest.mark.parametrize("reference, hypothesis, output", [
    ("hello world".split(), "hello world".split(), (0, 0, 0, 0)),
    ("hello world".split(), "hello".split(), (1, 0, 1, 0)),
    ("hello world".split(), "errr".split(), (2, 0, 1, 1)),
    ("hello world".split(), "hello that world".split(), (1, 1, 0, 0)),
    ("hello world".split(), "hola world".split(), (1, 0, 0, 1)),
    ("hello world".split(), "".split(), (2, 0, 2, 0))
])
def test_levenshtein(reference, hypothesis, output):
    assert levenshtein(reference, hypothesis) == output[0]


@pytest.mark.parametrize("reference, hypothesis, output", [
    ([["k", "aa", "t"]], [["k", "ae", "t"]], 1 / 3),
    ([["k", "aa", "t"]], [["k", "ae", "t"], ["k", "aa", "t"]], 0),
    ([["k", "aa", "t"], ["k", "ae", "t"]], [["k", "ae", "t"]], 0)
])
def test_per(reference, hypothesis, output):
    assert per(reference, hypothesis) == output
