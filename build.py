from distutils.core import Extension

extensions = [
    Extension("eevee.levenshtein", sources=["./eevee/levenshtein.c"])
]


def build(setup_kwargs):
    setup_kwargs.update({
        "ext_modules": extensions
    })
