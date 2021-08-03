from collections import Counter
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import Levenshtein
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

import eevee.transforms as tr
from eevee.types import SlotLabel

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


def aggregate_metrics(alternative_metrics: List[AlternativeMetric], aggregation_fn=np.mean) -> AlternativeMetric:
    """
    Aggregate metric dictionaries from multiple alternatives using
    `aggregation_fn`.

    An alternative metric books like the following:
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
        output[variant] =  {
            name: aggregation_fn([m[name] for m in metric_dicts])
            for name in metric_dicts[0].keys()
        }

    return output


def slot_capture_rate(slots_predicted: List[SlotLabel], expected_slot_value: str) -> float:
    """
    """

    if isinstance(slots_predicted, list) and slots_predicted:
        captured_count = slots_predicted.count(expected_slot_value)
        return captured_count / len(slots_predicted)
    else:
        raise ValueError(f"Expected {slots_predicted = } to be a list of size greater than 0")


def slot_retry_rate(slot_turn_counts: List[Optional[int]], agg_fn: Callable = np.mean) -> float:
    """
    Slot retry rate refers to aggregating on the number of turns a slot requires.

    Let `n` be the number of calls,
    `slot_turn_counts` be number of turns required to handle the particular slot in each call.
    `agg_fn` runs aggregation on `slot_turn_counts` and returns a float.
    """

    if isinstance(slot_turn_counts, list) and slot_turn_counts:
        filtered_turn_counts = list(filter(None, slot_turn_counts))
        return agg_fn(filtered_turn_counts)
    else:
        raise ValueError(f"Expected {slot_turn_counts =} to be a list of size greater than 0")


def slot_mismatch_rate(y_true: List[SlotLabel], y_pred: List[SlotLabel]) -> float:
    """
    As per the definition, slot mismatch rate captures
    ratio of match on types-but-not-values with
    match on types-but-not-values + match on types-and-values
    """

    type_and_value_matched = 0
    type_matched_but_value_didnt = 0

    for y_true_i, y_pred_i in zip(y_true, y_pred):

        if type(y_true_i) == type(y_pred_i) and y_true_i is not None:
            if y_true_i == y_pred_i:
                type_and_value_matched += 1
            else:
                type_matched_but_value_didnt += 1

    if type_matched_but_value_didnt + type_and_value_matched == 0:
        return 0.0
    return (type_matched_but_value_didnt) / (type_matched_but_value_didnt + type_and_value_matched)


def top_k_slot_mismatch_rate(y_true: List[SlotLabel], y_pred: List[List[SlotLabel]], k=1) -> float:
    ...


def slot_fnr(y_true: List[SlotLabel], y_pred: List[SlotLabel]) -> float:
    """
    False negative rate for slot prediction.

    Slot type is handled outside this, so you will have to segregate the slot
    labels based on types beforehand.
    """

    _y_true = [0 if y is None else 1 for y in y_true]
    _y_pred = [0 if y is None else 1 for y in y_pred]

    mat = confusion_matrix(_y_true, _y_pred, labels=[0, 1])

    fn = mat[1, 0]
    tp = mat[1, 1]

    if (fn + tp) == 0:
        return 0
    else:
        return fn / (fn + tp)


def wer(
    truth: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    truth_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    **kwargs
) -> float:
    """
    Calculate word error rate (WER) between a set of ground-truth sentences and
    a set of hypothesis sentences.
    See `compute_asr_measures` for details on the arguments.
    :return: WER as a floating point number
    """
    measures = compute_asr_measures(
        truth, hypothesis, truth_transform, hypothesis_transform, **kwargs
    )
    return measures["wer"]


def mer(
    truth: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    truth_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    **kwargs
) -> float:
    """
    Calculate match error rate (MER) between a set of ground-truth sentences and
    a set of hypothesis sentences.
    See `compute_asr_measures` for details on the arguments.
    :return: MER as a floating point number
    """
    measures = compute_asr_measures(
        truth, hypothesis, truth_transform, hypothesis_transform, **kwargs
    )
    return measures["mer"]


def wip(
    truth: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    truth_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    **kwargs
) -> float:
    """
    Calculate Word Information Preserved (WIP) between a set of ground-truth
    sentences and a set of hypothesis sentences.
    See `compute_asr_measures` for details on the arguments.
    :return: WIP as a floating point number
    """
    measures = compute_asr_measures(
        truth, hypothesis, truth_transform, hypothesis_transform, **kwargs
    )
    return measures["wip"]


def wil(
    truth: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    truth_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    **kwargs
) -> float:
    """
    Calculate Word Information Lost (WIL) between a set of ground-truth sentences
    and a set of hypothesis sentences.
    See `compute_asr_measures` for details on the arguments.
    :return: WIL as a floating point number
    """
    measures = compute_asr_measures(
        truth, hypothesis, truth_transform, hypothesis_transform, **kwargs
    )
    return measures["wil"]

def compute_asr_measures(
    truth: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    truth_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform] = _default_transform,
    **kwargs
) -> Mapping[str, float]:
    """
    Calculate error measures between a set of ground-truth sentences and a set of
    hypothesis sentences.
    The set of sentences can be given as a string or a list of strings. A string
    input is assumed to be a single sentence. A list of strings is assumed to be
    multiple sentences. Each word in a sentence is separated by one or more spaces.
    A sentence is not expected to end with a specific token (such as a `.`). If
    the ASR does delimit sentences it is expected that these tokens are filtered out.
    The optional `transforms` arguments can be used to apply pre-processing to
    respectively the ground truth and hypotheses input. Note that the transform
    should ALWAYS include `SentencesToListOfWords`, as that is the expected input.
    :param truth: the ground-truth sentence(s) as a string or list of strings
    :param hypothesis: the hypothesis sentence(s) as a string or list of strings
    :param truth_transform: the transformation to apply on the truths input
    :param hypothesis_transform: the transformation to apply on the hypothesis input
    :return: a dict with WER, MER, WIP and WIL measures as floating point numbers
    """

    # deal with old API
    if "standardize" in kwargs:
        truth = _standardize_transform(truth)
        hypothesis = _standardize_transform(hypothesis)

    if "words_to_filter" in kwargs:
        t = tr.Compose([tr.ToLowerCase(), tr.RemoveSpecificWords(kwargs["words_to_filter"])])
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
    H, S, D, I = _get_operation_counts(truth, hypothesis)

    # Compute Word Error Rate
    wer = float(S + D + I) / max(1, float(H + S + D))

    # Compute Match Error Rate
    mer = float(S + D + I) / max(1, float(H + S + D + I))

    # Compute Word Information Preserved
    wip = (float(H) / len(truth)) * (float(H) / len(hypothesis)) if hypothesis else 0

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
    
    oov_rate = oov/len(truth_raw)

    unk_rate = hypothesis_raw.count("<unk>")/len(truth_raw)

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
    truth: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    truth_transform: Union[tr.Compose, tr.AbstractTransform],
    hypothesis_transform: Union[tr.Compose, tr.AbstractTransform],
) -> Tuple[str, str]:
    """
    Pre-process the truth and hypothesis into a form that Levenshtein can handle.
    :param truth: the ground-truth sentence(s) as a string or list of strings
    :param hypothesis: the hypothesis sentence(s) as a string or list of strings
    :param truth_transform: the transformation to apply on the truths input
    :param hypothesis_transform: the transformation to apply on the hypothesis input
    :return: the preprocessed truth and hypothesis
    """

    # Apply transforms. By default, it collapses input to a list of words
    if truth.strip() not in  [" ", ""]:
        truth = truth_transform(truth)
    else:
        truth = [""]
    hypothesis = hypothesis_transform(hypothesis)

    # raise an error if the ground truth is empty
    # doesn't raise an error anymore due to the check in line 271. This is because we want to know the errors in silent segments
    if len(truth) == 0:
        raise ValueError("the ground truth cannot be an empty")

    # tokenize each word into an integer
    vocabulary = set(truth + hypothesis)
    word2char = dict(zip(vocabulary, range(len(vocabulary))))

    truth_chars = [chr(word2char[w]) for w in truth]
    hypothesis_chars = [chr(word2char[w]) for w in hypothesis]

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

    return hits, substitutions, deletions, insertions


def _get_per(
    truth: str, hypothesis: str
) -> Tuple[float, float]:
    """
    Calculates hPer and rPer
    :param truth: the ground truth
    :param hypothesis: the ASR hypothesis string
    :return: a tuple of #hper and #rper
    """
    r_count = Counter(truth)
    h_count = Counter(hypothesis)

    try:
        h_per = sum((h_count - r_count).values())/sum(r_count.values())
        r_per = sum((r_count - h_count).values())/sum(r_count.values())
    
    except ZeroDivisionError:
        h_per = sum((h_count - r_count).values())
        r_per = sum((r_count - h_count).values())


    return h_per, r_per


def _get_cer(
    truth: str, hypothesis: str
) -> float:
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


def _get_phn_error(
    truth: str, hypothesis: str, lexicon: Dict
) -> float:
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


def _get_am_errors(
    align: List, phone_post: List
) -> float:
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


def _get_ppl(
    sent: Union[str, List], lm
) -> float:
    """
    Calculates perplexity of sentences based on n-gram lm
    :param sent: Sentence for which perplexity needs to be calculated
    :param lm: N-Gram LM
    :return: Perplexity of sentence
    """
    if type(sent) == str:
        sent = sent.split()

    sent = [x for x in sent if x in lm.vocabulary()]
    sentence = " ".join(sent)
    # Perplexity = 1 / (P(sent)**(1/len(sent)))
    if len(sent) > 1:
        return (1/lm.s(sentence))**(1/len(sent))
    else:
        try:
            return (1/lm.p(sentence))**(1/len(sent))
        except KeyError:
            try:
                return (1/lm.p('<UNK>'))**(1/len(sent))
            except KeyError:
                return lm.counts()[0][1]


def slot_fpr(y_true: List[SlotLabel], y_pred: List[SlotLabel]) -> float:
    """
    False positive rate for slot prediction.

    Slot type is handled outside this, so you will have to segregate the slot
    labels based on types beforehand.

    0 -> 1 is false positive, assuming 0 is negative label. 1 as positive label.
    None -> {} is false positive under our situation.
    """

    _y_true = [0 if y is None else 1 for y in y_true]
    _y_pred = [0 if y is None else 1 for y in y_pred]

    mat = confusion_matrix(_y_true, _y_pred, labels=[0, 1])

    tn, fp, *_ = mat.ravel()

    if (fp + tn) == 0:
        return 0
    else:
        return fp / (fp + tn)


def intent_report(true_labels: pd.DataFrame, pred_labels: pd.DataFrame, output_dict=False):
    """
    Make an intent report from given label dataframes. We only support single
    intent dataframes as of now.

    TODO:
    - Check type of labels (we are not supporting rich labels right now)
    - Handle 'null' labels.
    """
    df = pd.merge(true_labels, pred_labels, on="id", how="inner")

    return classification_report(df["intent_x"], df["intent_y"], output_dict=output_dict)

