import itertools
import json
from collections import Counter
from functools import reduce
from operator import mul
from typing import Any, Dict, List, Mapping, Tuple, Union

from pandas import Series, DataFrame
from pandas.core.arrays import ExtensionArray
from pandas.core.generic import NDFrame

import eevee.transforms as tr
import Levenshtein
import numpy as np
import pandas as pd
from eevee.metrics.utils import fpr_fnr

_default_transform = tr.Compose(
    [
        tr.RemoveMultipleSpaces(),
        tr.Strip(),
        tr.ToLowerCase(),
        tr.SentencesToListOfWords(),
        tr.RemoveEmptyStrings(),
    ]
)

_standardize_transform = tr.Compose(
    [
        tr.ToLowerCase(),
        tr.ExpandCommonEnglishContractions(),
        tr.RemoveKaldiNonWords(),
        tr.RemoveWhiteSpace(replace_by_space=True),
    ]
)

AlternativeMetric = Dict[str, Any]


def aggregate_metrics(
    alternative_metrics: List[AlternativeMetric], aggregation_fn=np.mean
) -> AlternativeMetric:
    """
    Aggregate metric dictionaries from multiple alternatives using
    `aggregation_fn`.

    An alternative metric looks like the following:
    {
      "base": {"metric-name": <metric-value>},
      "lemmatized": {...},
      "stopword": {...},
      "hypothesis": <str>
    }
    """
    # Assuming first alternative has all the keys that are involved.
    variants = alternative_metrics[0].keys()
    # print(variants)
    # Skipping these items. They don't make sense from aggregation standpoint.
    variant_blacklist = {"hyp"}

    output = {}
    for variant in variants:
        if variant in variant_blacklist:
            continue

        metric_dicts = [am[variant] for am in alternative_metrics]
        output[variant] = {
            name: aggregation_fn([m[name] for m in metric_dicts])
            for name in metric_dicts[0].keys()
        }

    return output


def wer(
    truth: str,
    hypothesis: str,
    truth_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    **kwargs,
) -> float:
    """
    Calculate word error rate (WER) between a ground-truth sentence and
    a hypothesis sentence.
    See `compute_asr_measures` for details on the arguments.
    :return: WER as a floating point number
    """
    measures = compute_asr_measures(
        truth, hypothesis, truth_transform, hypothesis_transform, **kwargs
    )
    return measures["wer"]


def mer(
    truth: str,
    hypothesis: str,
    truth_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    **kwargs,
) -> float:
    """
    Calculate match error rate (MER) between a ground-truth sentence and
    a hypothesis sentence.
    See `compute_asr_measures` for details on the arguments.
    :return: MER as a floating point number
    """
    measures = compute_asr_measures(
        truth, hypothesis, truth_transform, hypothesis_transform, **kwargs
    )
    return measures["mer"]


def wip(
    truth: str,
    hypothesis: str,
    truth_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    **kwargs,
) -> float:
    """
    Calculate Word Information Preserved (WIP) between a ground-truth
    sentence and a hypothesis sentence.
    See `compute_asr_measures` for details on the arguments.
    :return: WIP as a floating point number
    """
    measures = compute_asr_measures(
        truth, hypothesis, truth_transform, hypothesis_transform, **kwargs
    )
    return measures["wip"]


def wil(
    truth: str,
    hypothesis: str,
    truth_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    **kwargs,
) -> float:
    """
    Calculate Word Information Lost (WIL) between a ground-truth sentence and a hypothesis sentence.
    See `compute_asr_measures` for details on the arguments.
    :return: WIL as a floating point number
    """
    measures = compute_asr_measures(
        truth, hypothesis, truth_transform, hypothesis_transform, **kwargs
    )
    return measures["wil"]


def compute_asr_measures(
    truth: str,
    hypothesis: str,
    truth_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    **kwargs,
) -> Mapping[str, float]:
    """
    Calculate error measures between a ground-truth sentence and a
    hypothesis sentence.
    The set of sentences can be given as a string. A string input is assumed to be a single sentence.
    Each word in a sentence is separated by one or more spaces.
    A sentence is not expected to end with a specific token (such as a `.`). If
    the ASR does delimit sentences it is expected that these tokens are filtered out.
    The optional `transforms` arguments can be used to apply pre-processing to
    respectively the ground truth and hypotheses input. Note that the transform
    should ALWAYS include `SentencesToListOfWords`, as that is the expected input.
    :param truth: the ground-truth sentence as a string
    :param hypothesis: the hypothesis sentence as a string
    :param truth_transform: the transformation to apply on the truths input
    :param hypothesis_transform: the transformation to apply on the hypothesis input
    :return: a dict with WER, MER, WIP and WIL measures as floating point numbers
    """

    # deal with old API
    if "standardize" in kwargs:
        truth = _standardize_transform(truth)
        hypothesis = _standardize_transform(hypothesis)

    if "words_to_filter" in kwargs:
        t = tr.Compose(
            [tr.ToLowerCase(), tr.RemoveSpecificWords(kwargs["words_to_filter"])]
        )
        truth = t(truth)
        hypothesis = t(hypothesis)

    if "lemmatize" in kwargs:
        t = tr.Compose([tr.ToLowerCase(), tr.Lemmatize(lang=kwargs["lang"])])
        truth = t(truth)
        hypothesis = t(hypothesis)

    # Preprocess truth and hypothesis
    truth, hypothesis, truth_raw, hypothesis_raw = _preprocess(
        truth, hypothesis, truth_transform, hypothesis_transform
    )

    # Get the operation counts (#hits, #substitutions, #deletions, #insertions)
    H, S, D, I, _ = _get_operation_counts(truth, hypothesis)

    # Compute Word Error Rate
    wer = float(S + D + I) / max(1, float(H + S + D))

    # Compute Match Error Rate
    mer = float(S + D + I) / max(1, float(H + S + D + I))

    # Compute Word Information Preserved
    wip = (
        (float(H) / max(1, len(truth))) * (float(H) / max(1, len(hypothesis)))
        if hypothesis
        else 0
    )

    # Compute Word Information Lost
    wil = 1 - wip

    # Get hPER and rPER
    hper, rper = _get_per(truth_raw, hypothesis_raw)

    # Get CER
    cer = _get_cer(truth_raw, hypothesis_raw)

    oov = 0
    if "lexicon" in kwargs and kwargs["lexicon"] is not None:
        phn_error = _get_phn_error(truth_raw, hypothesis_raw, kwargs["lexicon"])

        for word in truth_raw:
            try:
                kwargs["lexicon"][word]
            except KeyError:
                oov += 1

    else:
        phn_error = 0

    if "lm" in kwargs and kwargs["lm"] is not None:
        ppl = _get_ppl(hypothesis_raw, kwargs["lm"])
    else:
        ppl = 0

    oov_rate = oov / len(truth_raw)

    unk_rate = hypothesis_raw.count("<unk>") / len(truth_raw)

    return {
        "wer": wer,
        "cer": cer,
        "mer": mer,
        "wil": wil,
        "wip": wip,
        "hper": hper,
        "rper": rper,
        "phone_error": phn_error,
        "ppl": ppl,
        "oov_rate": oov_rate,
        "unk_rate": unk_rate,
        "hits": H,
        "substitutions": S,
        "deletions": D,
        "insertions": I,
    }


def _preprocess(
    truth: str,
    hypothesis: str,
    truth_transform: Union[tr.Compose, tr.AbstractTransform],
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform],
) -> Tuple[str, str]:
    """
    Pre-process the truth and hypothesis into a form that Levenshtein can handle.
    :param truth: the ground-truth sentence as a string
    :param hypothesis: the hypothesis sentence as a string
    :param truth_transform: the transformation to apply on the truths input
    :param hypothesis_transform: the transformation to apply on the hypothesis input
    :return: the preprocessed truth and hypothesis
    """

    # Apply transforms. By default, it collapses input to a list of words
    if truth.strip() not in [" ", ""]:
        truth = truth_transform(truth)
    else:
        truth = [""]

    if hypothesis.strip() not in [" ", ""]:
        hypothesis = hypothesis_transform(hypothesis)
    else:
        hypothesis = [""]

    # raise an error if the ground truth is empty
    # doesn't raise an error anymore due to the check in line 271. This is because we want to know the errors in silent segments
    if len(truth) == 0:
        raise ValueError("the ground truth cannot be an empty")

    # tokenize each word into an integer
    vocabulary = set(truth + hypothesis)
    word2char = dict(zip(vocabulary, range(len(vocabulary))))

    truth_chars = [chr(word2char[w]) for w in truth if w not in ["", " "]]
    hypothesis_chars = [chr(word2char[w]) for w in hypothesis if w not in ["", " "]]

    truth_str = "".join(truth_chars)
    hypothesis_str = "".join(hypothesis_chars)

    return truth_str, hypothesis_str, truth, hypothesis


def _get_operation_counts(
    source_string: str, destination_string: str
) -> Tuple[int, int, int, int]:
    """
    Check how many edit operations (delete, insert, replace) are required to
    transform the source string into the destination string. The number of hits
    can be given by subtracting the number of deletes and substitutions from the
    total length of the source string.
    :param source_string: the source string to transform into the destination string
    :param destination_string: the destination to transform the source string into
    :return: a tuple of #hits, #substitutions, #deletions, #insertions
    """

    editops = Levenshtein.editops(source_string, destination_string)

    substitutions = sum(1 if op[0] == "replace" else 0 for op in editops)
    deletions = sum(1 if op[0] == "delete" else 0 for op in editops)
    insertions = sum(1 if op[0] == "insert" else 0 for op in editops)
    hits = len(source_string) - (substitutions + deletions)

    return hits, substitutions, deletions, insertions, editops


def _get_per(truth: str, hypothesis: str) -> Tuple[float, float]:
    """
    Calculates hPer and rPer
    :param truth: the ground truth
    :param hypothesis: the ASR hypothesis string
    :return: a tuple of #hper and #rper
    """
    r_count = Counter(truth)
    h_count = Counter(hypothesis)

    try:
        h_per = sum((h_count - r_count).values()) / sum(r_count.values())
        r_per = sum((r_count - h_count).values()) / sum(r_count.values())

    except ZeroDivisionError:
        h_per = sum((h_count - r_count).values())
        r_per = sum((r_count - h_count).values())

    return h_per, r_per


def _get_cer(truth: str, hypothesis: str) -> float:
    """
    Calculates Character Error Rate.
    :param truth: the ground truth
    :param hypothesis: ASR hypothesis
    :return: CER (float)
    """
    truth = " ".join(truth)
    hypothesis = " ".join(hypothesis)

    editops = Levenshtein.editops(truth, hypothesis)

    S = sum(1 if op[0] == "replace" else 0 for op in editops)
    D = sum(1 if op[0] == "delete" else 0 for op in editops)
    I = sum(1 if op[0] == "insert" else 0 for op in editops)
    H = len(truth) - (S + D)

    cer = float(S + D + I) / max(1, float(H + S + D))

    return cer


def _get_phn_error(truth: str, hypothesis: str, lexicon: Dict) -> float:
    """
    Calculates Phone Error Rate between ground truth and ASR hypothesis. Ths is not the AM phone error rate
    :param truth: the ground truth
    :param hypothesis: ASR hypothesis
    :param lexicon: The ASR lexicon dictionary
    :return: Phone Error Rate (float)
    """
    truth = " ".join([lexicon[x] for x in truth if x in lexicon])
    hypothesis = " ".join([lexicon[x] for x in hypothesis if x in lexicon])

    vocabulary = set(truth + hypothesis)
    word2char = dict(zip(vocabulary, range(len(vocabulary))))

    truth = "".join([chr(word2char[w]) for w in truth])
    hypothesis = "".join([chr(word2char[w]) for w in hypothesis])

    editops = Levenshtein.editops(truth, hypothesis)

    S = sum(1 if op[0] == "replace" else 0 for op in editops)
    D = sum(1 if op[0] == "delete" else 0 for op in editops)
    I = sum(1 if op[0] == "insert" else 0 for op in editops)
    H = len(truth) - (S + D)

    phn_er = float(S + D + I) / max(float(H + S + D), 1)

    return phn_er


def _get_am_errors(align: List, phone_post: List) -> float:
    """
    Calculates frame error rate between alignments and AM predictions
    :param align: AM alignment on ground truth
    :param phone_post: AM phone posteriors
    :return: AM Frame Error Rate
    """

    align_phones = "".join([chr(int(p)) for p in align])
    post_phones = "".join([chr(int(p)) for p in phone_post])

    H, S, D, I = _get_operation_counts(align_phones, post_phones)

    # Compute frame error rate
    fer = float(S + D + I) / max(float(H + S + D), 1)
    return fer


def _get_ppl(sent: str, lm) -> float:
    """
    Calculates perplexity of a sentence based on n-gram lm
    :param sent: Sentence for which perplexity needs to be calculated
    :param lm: N-Gram LM
    :return: Perplexity of sentence
    """
    sent = sent.split()

    sent = [x for x in sent if x in lm.vocabulary()]
    sentence = " ".join(sent)
    # Perplexity = 1 / (P(sent)**(1/len(sent)))
    if len(sent) > 1:
        return (1 / lm.s(sentence)) ** (1 / len(sent))
    else:
        try:
            return (1 / lm.p(sentence)) ** (1 / len(sent))
        except KeyError:
            try:
                return (1 / lm.p("<UNK>")) ** (1 / len(sent))
            except KeyError:
                return lm.counts()[0][1]


def get_ops(truths: List[str], preds: List[str]) -> pd.DataFrame:
    ops = []
    ops_list = []
    for truth, pred in zip(truths, preds):
        truth_rep, pred_rep, truth, pred = _preprocess(
            truth, pred, _default_transform, _default_transform
        )
        _, _, _, _, editops = _get_operation_counts(truth_rep, pred_rep)

        for op in editops:
            if op[0] == "insert":
                ops_list.append(("insertion", "***", pred[op[2]]))
            elif op[0] == "delete":
                ops_list.append(("deletion", truth[op[1]], "***"))
            else:
                ops_list.append(("substitution", truth[op[1]], pred[op[2]]))
    op_counts = Counter(ops_list)
    for op in op_counts:
        ops.append(
            {"operation": op[0], "truth": op[1], "pred": op[2], "count": op_counts[op]}
        )

    return ops


def get_alt_metric(truth: str, predictions: List[str], metric) -> List[float]:
    """
    Get a metric over a list of prediction alternatives
    """
    results = []
    for pred in predictions:
        results.append(metric(truth, pred))
    return results


def merge_utterances(utterances):
    """
    At times when the user is speaking with gaps, we get more than one results
    from Google ASR, each with its own list of alternatives.

    To make sure that the rest of the systems work fine, we merge those
    results using a cross join and re-rank them using the following two attributes:
    1. Sum of index for each utterance
    2. Product of confidence values

    NOTE: We are still returning items as if the ASR returned a single utterance.
          Also the confidence values here will be very low so care must be taken
          in interpretation.
    """

    if len(utterances) < 2:
        return utterances

    merged = []

    def _join_transcripts(transcripts):
        return " ".join(text.strip() for text in transcripts if text != None)

    indexed_results = [enumerate(utt) for utt in utterances]

    for tup in itertools.product(*indexed_results):
        if tup:
            index_sum = sum(idx for idx, _ in tup)
            # NOTE: We lose fields other than transcript and confidence here.
            alternative = {
                "transcript": _join_transcripts([alt["transcript"] for _, alt in tup]),
                "confidence": reduce(mul, [alt["confidence"] or 0 for _, alt in tup]),
            }
            merged.append((index_sum, alternative))

    merged = sorted(merged, key=lambda it: (it[0], -it[1]["confidence"] or 0))[:10]
    return [[alt for _, alt in merged]]


def get_n_transcripts(utterances, n=3) -> List[str]:
    """
    Return a list of first n transcripts.
    """
    transcripts = []
    if utterances == []:
        return [""]
    for x in range(min(n, len(utterances[0]))):
        try:
            if utterances[0][x]["transcript"]:
                transcripts.append(utterances[0][x]["transcript"])
        except (KeyError, IndexError):
            pass
    if transcripts == []:
        transcripts.append("")
    return transcripts


def get_first_transcript(utterances) -> str:
    """
    Return first transcript from the first utterance. Return '' if utterances
    are empty.
    """

    try:
        return utterances[0][0]["transcript"]
    except (KeyError, IndexError):
        return ""


def asr_report(
    true_labels: pd.DataFrame, pred_labels: pd.DataFrame, dump: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate ASR report based on true and predicted labels.

    `true_labels` is a CSV following TranscriptionLabel protobuf definition
    from dataframes. While `pred_labels` follows RichTranscriptionLabel
    protobuf definition.

    The report covers the following metrics:
    - WER: Mean WER over all the utterances.
    - Utterance FPR: Ratio of utterances where truth is empty but prediction is non-empty.
    - Utterance FNR: Ratio of utterances where predictions are empty, while the truth is non-empty.
    """

    # TODO: Add min-k variant
    df = pd.merge(true_labels, pred_labels, on="id", how="inner")

    # Since empty items in true transcription is read as NaN, we have to
    # replace them
    df["transcription"] = df["transcription"].fillna("")

    # TODO: Do validation on type of input
    df["utterances"] = df["utterances"].apply(
        lambda it: merge_utterances(json.loads(it))
    )

    df["all_pred_transcriptions"] = df["utterances"].apply(
        get_n_transcripts, args=(10,)
    )

    df["pred_transcription"] = df["all_pred_transcriptions"].map(lambda x: x[0])

    df["all_wer"] = df.apply(
        lambda row: get_alt_metric(
            row["transcription"], row["all_pred_transcriptions"], wer
        ),
        axis=1,
    )

    df["wer"] = df["all_wer"].map(lambda x: x[0])

    for n in [3, 10]:
        df[f"min_{n}_wer"] = df.apply(
            lambda row: min(row["all_wer"][:n]),
            axis=1,
        )

    (utterance_fpr, total_empty), (utterance_fnr, total_non_empty) = fpr_fnr(
        df["transcription"] == "", df["pred_transcription"] == "", labels=[False, True]
    )

    # sentence error rate = number of sentences with error / number of sentences
    ser = len(list(filter(lambda x: x > 0, df["wer"].tolist()))) / len(df["wer"])

    # short utterance WER. Short deifined as less than or equal to 2 words
    short_utterance_len = 3
    df["length"] = df["transcription"].apply(lambda x: len(x.strip().split()))
    short_utterance_df = df[(df["length"] > 0) & (df["length"] < short_utterance_len)]

    # long utterance WER
    long_utterance_df = df[df["length"] >= short_utterance_len]

    # TODO: Find WER over the corpus (like this → https://kaldi-asr.org/doc/compute-wer_8cc.html)
    report = pd.DataFrame(
        {
            "Metric": [
                "WER",
                "Utterance FPR",
                "Utterance FNR",
                "SER",
                "Min 3 WER",
                "Min WER",
                "Short Utterance WER",
                "Long Utterance WER",
            ],
            "Value": [
                df["wer"].mean(),
                utterance_fpr,
                utterance_fnr,
                ser,
                df["min_3_wer"].mean(),
                df["min_10_wer"].mean(),
                short_utterance_df["wer"].mean(),
                long_utterance_df["wer"].mean(),
            ],
            "Support": [
                len(df),
                total_empty,
                total_non_empty,
                len(df),
                len(df),
                len(df),
                len(short_utterance_df),
                len(long_utterance_df),
            ],
        }
    )
    report.set_index("Metric", inplace=True)
    if dump:
        ops = pd.DataFrame(
            get_ops(df["transcription"], df["pred_transcription"])
        ).sort_values(by=["operation", "count"], ascending=[True, False])

        return report, df, ops

    else:
        return report


def extract_info_tags(transcription: str) -> str:
    pass


def clean_info_tags(transcription: str) -> str:
    pass


## change this if you want to change the definition of noisy
def define_noisy(tag):
    if tag is None:
        return -1
    elif "silent" in tag:
        return "silent"
    else:
        return "noisy"


def process_noise_info(
        true_labels: pd.DataFrame, pred_labels: pd.DataFrame
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:

    df = pd.merge(true_labels, pred_labels, on="id", how="inner")
    
    ## separate out info tags in transcripts and clean the original transcriptions
    df["info-tag"] = df["transcription"].apply(lambda trancription: extract_info_tags(trancription))
    df["cleaned-transcription"] = df["transcription"].apply(lambda transcription: clean_info_tags(trancription))
    
    ## separate noisy and not-noisy subsets
    df["noise-label"] = df["info-tag"].apply(lambda tag: define_noisy(tag))
    noisy_df = df[df["noise-label"]=="noisy"]
    not_noisy_df = df[df["noise-label"]!="noisy"]
    
    data_subsets: List = []
    for df in [noisy_df, not_noisy_df]:
        data_subsets.append({"true": df[["id", "transcription"]], "pred": df[["id", "utterances"]]})
        
    return tuple(data_subsets)
