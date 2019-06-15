import pytest

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
def test_wer(reference, hypothesis, output):
    assert levenshtein(reference, hypothesis) == output[0]
