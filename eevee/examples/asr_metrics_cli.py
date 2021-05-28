"""
Command line interface to get ASR metrics

Usage:
poetry run python cli.py --lang=<lang> --in=<in> --out=<out> [--stop-path=stop-path] [--lexicon=lexicon] 

Options:
--lang=<lang>               Language of transcriptions
--in=<in>                   Input file path. File should contain list of jsons
--out=<out>                 Output file path
--stop-path=stop-path       Stop word file path [default: None]
--lexicon=lexicon           Lexicon file path [default: None]

"""
import json

from docopt import docopt
from tqdm import tqdm

from eevee.asr_metrics import get_metrics


def main():
    args = docopt(__doc__)

    lang = args['--lang']
    in_path = args['--in']
    out_path = args['--out']
    stop_path = args['--stop-path']
    lexicon = args['--lexicon']


    with open(in_path) as fin:
        in_file = json.load(fin)

    if stop_path:
        with open(stop_path) as fin:
            remove_words = fin.read().split('\n')
    else:
        remove_words = None
    
    if lexicon:
        with open(lexicon) as fin:
            lex = fin.read().split('\n')
            lexicon = {x.split(' ', 1)[0]: x.split(' ', 1)[1] for x in lex}

    else:
        lexicon = None

    results = []
    for x in tqdm(in_file, desc='Records processed'):
        results.append(get_metrics(x, lang, remove_words=remove_words, lexicon=lexicon))
    
    
    with open(out_path) as fout:
        fout.write(results)



    