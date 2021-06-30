"""
Command line interface to get ASR metrics

Usage:
asr_metrics_cli.py --lang=<lang> --transcripts=<transcripts> --out=<out> [--stop-path=stop-path] [--lexicon=lexicon] [--alignments=alignments] [--phone-post=phone-post] [--lm=lm] 

Options:
--lang=<lang>               Language of transcriptions
--transcripts=<transcripts> Input file path. File should contain list of jsons
--out=<out>                 Output file path
--stop-path=<stop-path>       Stop word file path 
--lexicon=<lexicon>           Lexicon file path 
--alignments=<alignments>     NNet3/Chain alignments from Kaldi
--phone-post=<phone-post>     Phone posteriors fomr Kaldi
--lm=<lm>                     Language Model. Should be arpa format

"""

import json
import sqlite3

from docopt import docopt
from tqdm import tqdm
import arpa
import pandas as pd

from eevee.asr_metrics import get_metrics, parse_phone_posterior, parse_alignments


def main():
    tqdm.pandas()
    def get_phone_posts(uuid):
        try:
            return post[uuid]
        except KeyError:
            return None


    def get_alignment(uuid):
        try:
            return align[uuid]
        except KeyError:
            return None

    args = docopt(__doc__)
    
    transcripts = args["--transcripts"]
    
    lang = args["--lang"]
    
    out_path = args["--out"]
    stop_path = args["--stop-path"]
    lexicon = args["--lexicon"]
    alignments = args["--alignments"]
    phone_post = args["--phone-post"]
    lm = args["--lm"]

    if transcripts.endswith('.sqlite'):
        with sqlite3.connect(transcripts) as db:
            df = pd.read_sql_query("SELECT * FROM data", db)
            df.rename(columns={"tag": "transcription"}, inplace=True)
            df["uuid"] = df.apply(lambda x: json.loads(x["data"])["uuid"], axis=1)
            df["alternatives"] = df.apply(lambda x: json.loads(x["data"])["alternatives"], axis=1)
    else:
        df = pd.read_json(transcripts)
        df['uuid'] = df['uuid']
        df.rename(columns={"gasr_output_alternatives": "alternatives"}, inplace=True)

    if stop_path:
        with open(stop_path) as fin:
            remove_words = fin.read().split("\n")
    else:
        remove_words = None
    
    # parse lexicon file
    if lexicon:
        with open(lexicon) as fin:
            lex = fin.read().split("\n")
        lexicon = {}
        for word in lex:
            try:
                lexicon[word.split(" ", 1)[0]] = word.split(" ", 1)[0]
            except IndexError:
                pass

    else:
        lexicon = None

    if alignments and phone_post:
        with open(phone_post) as fin:
            post = fin.read().split("\n")

        with open(alignments) as fin:
            align = fin.read().split("\n")

        post = parse_phone_posterior(post)
        align = parse_alignments(align)

        df["phone_post"] = df.apply(lambda x: get_phone_posts(x["uuid"]), axis=1)
        df["alignment"] = df.apply(lambda x: get_alignment(x["uuid"]), axis=1)

    else:
        df["phone_post"] = None
        df["alignment"] = None

    if lm:
        lm = arpa.loadf(lm)[0]
    else:
        lm = None

    df["results"] = df.progress_apply(lambda x: get_metrics(ref=json.loads(x["transcription"])["text"], hyp=x["alternatives"], lang=lang, lexicon=lexicon, lm=lm, alignment=x["alignment"], phone_post=x["phone_post"], remove_words=remove_words), axis=1)
    
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()