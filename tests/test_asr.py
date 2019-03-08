import pytest
from yamraz.tokenizer import tokenize

from eevee.levenshtein import levenshtein


# NOTE: This test keeps insertion, deletion and substitution information also
#       but for now we are only using the the total cost
@pytest.mark.parametrize("reference, hypothesis, output", [
    ("hello world", "hello world", (0, 0, 0, 0)),
    ("hello world", "hello", (1, 0, 1, 0)),
    ("hello world", "errr", (2, 0, 1, 1)),
    ("hello world", "hello that world", (1, 1, 0, 0)),
    ("hello world", "hola world", (1, 0, 0, 1)),
    ("hello world", "", (2, 0, 2, 0))
])
def test_wer(reference, hypothesis, output):
    lang = "en"
    assert levenshtein(tokenize(reference, lang), tokenize(hypothesis, lang)) == output[0]
