from distutils.core import Extension

extensions = [
    Extension("eevee.levenshtein", sources=["./eevee/levenshtein.cc"])
]


def build(setup_kwargs):
    setup_kwargs.update({
        "ext_modules": extensions
    })
