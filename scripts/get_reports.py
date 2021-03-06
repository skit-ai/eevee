"""
Command line interface to generate ASR reports

Usage: get_reports.py --asr-report=<asr-report> --slu-output=<slu-sqlite> --bucket-sqlite=<bucket-sqlite> --lang=<lang> --client-config=<client-config> --lang-list=<lang-list> --dest-dir=<dest-dir> [--prefix=<prefix>]

Options:
--asr-report=<asr-report>           CSV file generated using eevee ASR metrics
--slu-output=<slu-sqlite>           JSON file generated by SLU
--bucket-sqlite=<bucket-sqlite>     Region tagging sqlite containing audio bucket tags
--lang=<lang>                       Language of files
--client-config=<client-config>     YAML containing intents and smalltalk tags
--lang-list=<lang-list>             YAML with list of languages
--dest-dir=<dest-dir>               Destination for reports
--prefix=prefix                     Prefix for report files [default: '']

"""

import os
import re
import json
import sqlite3

import numpy as np
import yaml
from docopt import docopt
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support


def _get_languages(lang_path):
    """
    Return supported language list
    """
    with open(lang_path) as fin:
        languages = yaml.load(fin, Loader=yaml.FullLoader)
    return languages["lang"]


def _parse_client_config(path):
    """
    Return intents associated with the client
    """
    with open(path) as fin:
        contents = yaml.load(fin, Loader=yaml.FullLoader)
    return contents["intents"], contents["smalltalk"]


def _get_buckets(rec):

    buckets = []
    for x in json.loads(rec):
        buckets.append(x["type"])
    return buckets


def _parse_audio_metrics(record):
    record = json.loads(record)
    res = {}
    try:
        res["lemmatized_wer"] = record["alternatives"][0]["lemmatized"]["wer"]
        res["lemmatized_cer"] = record["alternatives"][0]["lemmatized"]["cer"]
        res["lemmatized_mer"] = record["alternatives"][0]["lemmatized"]["mer"]
        res["lemmatized_hper"] = record["alternatives"][0]["lemmatized"]["hper"]
        res["lemmatized_rper"] = record["alternatives"][0]["lemmatized"]["rper"]
        res["lemmatized_wil"] = record["alternatives"][0]["lemmatized"]["wil"]
        res["lemmatized_wip"] = record["alternatives"][0]["lemmatized"]["wip"]

        res["ref_ppl"] = record.get("ref_ppl", 0)
        res["wer"] = record["alternatives"][0]["base"]["wer"]
        res["cer"] = record["alternatives"][0]["base"]["cer"]
        res["mer"] = record["alternatives"][0]["base"]["mer"]
        res["hper"] = record["alternatives"][0]["base"]["hper"]
        res["rper"] = record["alternatives"][0]["base"]["rper"]
        res["wil"] = record["alternatives"][0]["base"]["wil"]
        res["wip"] = record["alternatives"][0]["base"]["wip"]
        res["ppl"] = record["alternatives"][0]["base"]["ppl"]
        res["per"] = record["alternatives"][0]["base"]["phone_error"]
        res["oov_rate"] = record["alternatives"][0]["base"]["oov_rate"]
        res["unk_rate"] = record["alternatives"][0]["base"]["unk_rate"]
        res["hits"] = record["alternatives"][0]["base"]["hits"]
        res["substitutions"] = record["alternatives"][0]["base"]["substitutions"]
        res["insertions"] = record["alternatives"][0]["base"]["insertions"]
        res["deletions"] = record["alternatives"][0]["base"]["deletions"]

        res["avg_wer"] = record["avg"]["base"]["wer"]
        res["avg_cer"] = record["avg"]["base"]["cer"]
        res["avg_mer"] = record["avg"]["base"]["mer"]
        res["avg_hper"] = record["avg"]["base"]["hper"]
        res["avg_rper"] = record["avg"]["base"]["rper"]
        res["avg_wil"] = record["avg"]["base"]["wil"]
        res["avg_wip"] = record["avg"]["base"]["wip"]
        res["avg_ppl"] = record["avg"]["base"]["ppl"]
        res["avg_per"] = record["avg"]["base"]["phone_error"]
        res["avg_unk_rate"] = record["avg"]["base"]["unk_rate"]
        res["avg_hits"] = record["avg"]["base"]["hits"]
        res["avg_substitutions"] = record["avg"]["base"]["substitutions"]
        res["avg_insertions"] = record["avg"]["base"]["insertions"]
        res["avg_deletions"] = record["avg"]["base"]["deletions"]

        res["top_3_wer"] = record["top_3"]["base"]["wer"]
        res["top_3_cer"] = record["top_3"]["base"]["cer"]
        res["top_3_mer"] = record["top_3"]["base"]["mer"]
        res["top_3_hper"] = record["top_3"]["base"]["hper"]
        res["top_3_rper"] = record["top_3"]["base"]["rper"]
        res["top_3_wil"] = record["top_3"]["base"]["wil"]
        res["top_3_wip"] = record["top_3"]["base"]["wip"]
        res["top_3_ppl"] = record["top_3"]["base"]["ppl"]
        res["top_3_per"] = record["top_3"]["base"]["phone_error"]
        res["top_3_unk_rate"] = record["top_3"]["base"]["unk_rate"]

        res["top_5_wer"] = record["top_5"]["base"]["wer"]
        res["top_5_cer"] = record["top_5"]["base"]["cer"]
        res["top_5_mer"] = record["top_5"]["base"]["mer"]
        res["top_5_hper"] = record["top_5"]["base"]["hper"]
        res["top_5_rper"] = record["top_5"]["base"]["rper"]
        res["top_5_wil"] = record["top_5"]["base"]["wil"]
        res["top_5_wip"] = record["top_5"]["base"]["wip"]
        res["top_5_ppl"] = record["top_5"]["base"]["ppl"]
        res["top_5_per"] = record["top_5"]["base"]["phone_error"]
        res["top_5_unk_rate"] = record["top_5"]["base"]["unk_rate"]

        res["top_7_wer"] = record["top_7"]["base"]["wer"]
        res["top_7_cer"] = record["top_7"]["base"]["cer"]
        res["top_7_mer"] = record["top_7"]["base"]["mer"]
        res["top_7_hper"] = record["top_7"]["base"]["hper"]
        res["top_7_rper"] = record["top_7"]["base"]["rper"]
        res["top_7_wil"] = record["top_7"]["base"]["wil"]
        res["top_7_wip"] = record["top_7"]["base"]["wip"]
        res["top_7_ppl"] = record["top_7"]["base"]["ppl"]
        res["top_7_per"] = record["top_7"]["base"]["phone_error"]
        res["top_7_unk_rate"] = record["top_7"]["base"]["unk_rate"]

        res["1v5_wer"] = record["1v5"]["base"]["wer"]
        res["1v5_cer"] = record["1v5"]["base"]["cer"]
        res["1v5_mer"] = record["1v5"]["base"]["mer"]
        res["1v5_hper"] = record["1v5"]["base"]["hper"]
        res["1v5_rper"] = record["1v5"]["base"]["rper"]
        res["1v5_wil"] = record["1v5"]["base"]["wil"]
        res["1v5_wip"] = record["1v5"]["base"]["wip"]
        res["1v5_ppl"] = record["1v5"]["base"]["ppl"]
        res["1v5_per"] = record["1v5"]["base"]["phone_error"]

        res["1v10_wer"] = record["1v10"]["base"]["wer"]
        res["1v10_cer"] = record["1v10"]["base"]["cer"]
        res["1v10_mer"] = record["1v10"]["base"]["mer"]
        res["1v10_hper"] = record["1v10"]["base"]["hper"]
        res["1v10_rper"] = record["1v10"]["base"]["rper"]
        res["1v10_wil"] = record["1v10"]["base"]["wil"]
        res["1v10_wip"] = record["1v10"]["base"]["wip"]
        res["1v10_ppl"] = record["1v10"]["base"]["ppl"]
        res["1v10_per"] = record["1v10"]["base"]["phone_error"]

        return res
    except (KeyError, IndexError):
        return


def _get_bucket_report(view, bucket, smalltalk, intents):

    if bucket != "all":
        view = view[view[bucket] == True]

    avg = {}
    avg["IRR-support"] = len(view)

    if avg["IRR-support"] == 0:
        return {"discard": True}

    avg["IRR-precision"], avg["IRR-recall"], _, _ = precision_recall_fscore_support(
        view["true-tag"],
        view["pred-tag"],
        labels=intents + smalltalk + ["_oos_"],
        average="weighted",
        zero_division=1,
    )
    (
        avg["IRR-inscope-precision"],
        avg["IRR-inscope-recall"],
        _,
        _,
    ) = precision_recall_fscore_support(
        view["true-tag"],
        view["pred-tag"],
        labels=intents,
        average="weighted",
        zero_division=1,
    )
    avg["IRR-inscope-support"] = len(view[view["true-tag"].isin(intents)])
    (
        avg["IRR-smalltalk-precision"],
        avg["IRR-smalltalk-recall"],
        _,
        _,
    ) = precision_recall_fscore_support(
        view["true-tag"],
        view["pred-tag"],
        labels=smalltalk,
        average="weighted",
        zero_division=1,
    )
    avg["IRR-smalltalk-support"] = len(view[view["true-tag"].isin(smalltalk)])
    oos_p, oos_r, _, oos_s = precision_recall_fscore_support(
        view["true-tag"], view["pred-tag"], labels=["_oos_"], zero_division=1
    )
    avg["IRR-oos-precision"], avg["IRR-oos-recall"], avg["IRR-oos-support"] = (
        oos_p[0],
        oos_r[0],
        oos_s[0],
    )

    avg.update(
        pd.DataFrame(
            view.apply(lambda x: _parse_audio_metrics(x["results"]), axis=1)
            .dropna()
            .to_list()
        ).mean()
    )

    return avg


def _get_utterance_report(view):
    view["metrics"] = view.apply(lambda x: _parse_audio_metrics(x["results"]), axis=1)

    view = view.join(pd.DataFrame(view.dropna().pop("metrics").values.tolist()))

    view["results"] = view["results"].apply(lambda x: json.loads(x))
    view["transcription"] = view["transcription"].apply(lambda x: json.loads(x))

    return view


def _get_potential_improv(view, metric, support, target, index):
    values = view[metric].tolist()
    support = view[support].tolist()
    if support[index] > 0:
        values[index] = target

    values = np.average(values, weights=support)

    return values


def main():
    args = docopt(__doc__)

    lang = args["--lang"]
    dest_dir = args["--dest-dir"]
    prefix = args["--prefix"]
    client_config = args["--client-config"]
    lang_path = args["--lang-list"]

    languages = _get_languages(lang_path)

    # replacing lanuage codes with words. eg en -> english
    if len(lang) == 2:
        lang = languages[lang]

    # consider everyhting other than lang as different language
    diff_lang_set = list(languages.values())
    diff_lang_set.remove(lang)

    # convert list to string for regex
    diff_lang_set = "|".join([f"<{x}_" for x in diff_lang_set])

    intents, smalltalk = _parse_client_config(client_config)

    if prefix != "" and not prefix.endswith("-"):
        prefix = prefix + "-"

    with sqlite3.connect(args["--bucket-sqlite"]) as db:
        df_it = pd.read_sql_query("SELECT * FROM data", db)

    df_res = pd.read_csv(args["--asr-report"])
    df_slu_pred = pd.read_csv(args["--slu-output"])

    df_it["uuid"] = df_it.apply(lambda x: json.loads(x["data"])["uuid"], axis=1)
    df_it.rename(columns={"tag": "bucket"}, inplace=True)

    df = pd.merge(
        pd.merge(
            df_it[["uuid", "bucket"]],
            df_res[["uuid", "results", "transcription"]],
            on="uuid",
        ),
        df_slu_pred[["uuid", "true-tag", "pred-tag", "audio_url", "alternatives"]],
        on="uuid",
    )

    if len(df) == 0:
        raise ValueError(
            "UUIDs in bucket-sqlite, asr-report and slu-output do not match. Kindly make sure they match. Exiting"
        )

    # convert dict of bucket keys to list of buckets.
    df["bucket-list"] = df.apply(lambda x: _get_buckets(x["bucket"]), axis=1)

    # create new bool columns indicating bucket association
    df = pd.merge(
        df,
        (df["bucket-list"].str.join(",").str.get_dummies(sep=",").astype(bool)),
        left_index=True,
        how="outer",
        right_index=True,
    )

    df["ref-len"] = df.apply(
        lambda x: len(json.loads(x["results"])["ref"].split()), axis=1
    )

    df["short_sentence"] = df["ref-len"].between(0, 3, inclusive=False)

    df["long_sentence"] = df["ref-len"] > 2

    df["no_sentence"] = df["ref-len"] == 0

    if lang == "english":
        df["code_mix"] = df.apply(
            lambda x: True
            if re.search(diff_lang_set, json.loads(x["transcription"])["text"].lower())
            or x["true-tag"] == f"non_{lang}"
            else False,
            axis=1,
        )

    else:
        df["code_mix"] = df.apply(
            lambda x: True
            if re.search(diff_lang_set, json.loads(x["transcription"])["text"].lower())
            or re.search("[a-z]", json.loads(x["transcription"])["text"].lower())
            or x["true-tag"] == f"non_{lang}"
            else False,
            axis=1,
        )

    df["model_lang"] = ~df["code_mix"]

    buckets = list(
        set(",".join([",".join(x) for x in df["bucket-list"].to_list()]).split(","))
    ) + ["model_lang", "code_mix", "short_sentence", "long_sentence", "no_sentence"]

    df.loc[~df["true-tag"].isin(intents + smalltalk), "true-tag"] = "_oos_"
    df.loc[~df["pred-tag"].isin(intents + smalltalk), "pred-tag"] = "_oos_"

    overall_report = []

    bucket_report = []
    for bucket in buckets + ["all"]:
        overall_report.append(
            {
                "bucket": bucket,
                **_get_bucket_report(
                    view=df, bucket=bucket, smalltalk=smalltalk, intents=intents
                ),
            }
        )

    speech_tags = [x for x in buckets if "speech" in x and "background" not in x]
    background_tags = [x for x in buckets if "background" in x]
    noise_tags = [x for x in buckets if "noise" in x]
    len_tags = [x for x in buckets if "_sentence" in x]
    other_tags = [
        x
        for x in buckets
        if x not in speech_tags + background_tags + noise_tags + len_tags
    ]

    for speech in speech_tags:
        for background in background_tags:
            for noise in noise_tags:
                for length in len_tags:
                    for ot in other_tags:
                        res = {
                            "speech_tag": speech,
                            "background_tag": background,
                            "noise_tag": noise,
                            "sentence_length": length,
                            "bucket": ot,
                            **_get_bucket_report(
                                view=df[
                                    (df[speech] == True)
                                    & (df[background] == True)
                                    & (df[noise] == True)
                                    & (df[length] == True)
                                ],
                                bucket=ot,
                                smalltalk=smalltalk,
                                intents=intents,
                            ),
                        }
                        if "discard" not in res:
                            bucket_report.append(res)

    bucket_report = pd.DataFrame(bucket_report).fillna(0)

    overall_report = pd.DataFrame(overall_report)

    intent_report = pd.DataFrame(
        classification_report(
            df["true-tag"].tolist(), df["pred-tag"].tolist(), output_dict=True
        )
    ).transpose()

    utterance_report = _get_utterance_report(df).drop(
        columns=buckets + ["bucket"], axis=1
    )

    for metric in ["IRR", "IRR-inscope", "IRR-smalltalk", "IRR-oos"]:
        bucket_report.insert(
            bucket_report.columns.get_loc(f"{metric}-support") + 1,
            f"{metric}-precision-potential",
            bucket_report.apply(
                lambda x: _get_potential_improv(
                    bucket_report,
                    f"{metric}-precision",
                    f"{metric}-support",
                    1.0,
                    x.name,
                ),
                axis=1,
            ),
        )
        bucket_report.insert(
            bucket_report.columns.get_loc(f"{metric}-support") + 2,
            f"{metric}-recall-potential",
            bucket_report.apply(
                lambda x: _get_potential_improv(
                    bucket_report, f"{metric}-recall", f"{metric}-support", 1.0, x.name
                ),
                axis=1,
            ),
        )

    os.makedirs(dest_dir, exist_ok=True)

    bucket_report.to_csv(
        os.path.join(dest_dir, f"{prefix}bucket-report.csv"), index=False
    )

    overall_report.to_csv(
        os.path.join(dest_dir, f"{prefix}overall-report.csv"), index=False
    )

    intent_report.to_csv(os.path.join(dest_dir, f"{prefix}intent-report.csv"))

    utterance_report.to_csv(
        os.path.join(dest_dir, f"{prefix}utterance-report.csv"), index=False
    )

    print(f"Intents : {len(df[df['true-tag'].isin(intents)])}")
    print(f"Smalltalk : {len(df[df['true-tag'].isin(smalltalk)])}")
    print(f"OOS : {len(df[df['true-tag']== '_oos_'])}")

    print("Done")

    return


if __name__ == "__main__":
    main()
