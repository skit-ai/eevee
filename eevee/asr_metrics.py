import json
import re
from typing import Union, List, Dict
from collections import Counter

from eevee import metrics


def get_metrics(ref: str, hyp:Union[str, List], lang: str, remove_words: Union[str, List] = None, lexicon: Union[str, Dict] = None, lm=None, alignment=None, phone_post=None) -> Dict:
    """
    Takes ground truth and predictions (can be string or list) and returns related ASR metrics. Optional arguments provide more metrics
    :param ref: ground truth
    :param hyp: the predicted text (ASR hypothesis)
    :param lang: language code (eg en, hi, ta)
    :param remove_words: Text file path or a list of strings to be removed from the ground truth and hypothesis. Can be used to discount stop words etc
    :param lexicon: Kaldi lexicon file  path or a {word:lexicon} dict. Will give prediction phone error rate (not the AM per)
    :param lm: Language model in ARPA format. Check examples for loading example
    :param alignment: Kaldi forced alignment vector
    :param phone_post: Kaldi NNET3/Chain phone posteriors. 
    :return: JSON string containing all ASR metrics. 

    """
    ref = re.sub("\<.*?\>"," ",ref) 
    
    if remove_words and type(remove_words) == str:
        with open(remove_words) as fin:
            remove_words = fin.read().split("\n")

    if lexicon and type(lexicon) == str:
        with open(lexicon) as fin:
            lex = fin.read().split("\n")
        
    
        lexicon = {}
        for word in lex:
            try:
                lexicon[word.split(" ", 1)[0]] = word.split(" ", 1)[0]
            except IndexError:
                pass

    if type(hyp) == str:
        results = _parse_string(ref, hyp, lang, remove_words, lexicon, lm)

    elif type(hyp) == list:
        results = _parse_alters(ref, hyp, lang, remove_words, lexicon, lm)

    try:
        if alignment and phone_post:
            results["am_fer"] = _get_am_errors(phone_post, alignment)
    except KeyError:
        results["am_fer"] = "NA"
        

    return results


def _parse_string(ref:str, hyp:Union[str, List], lang: str = "en", remove_words: List = None, lexicon: Dict = None, lm=None) -> Dict:
    """
    Parses a reference and hpothesis string and return ASR metrics
    :param ref: ground truth
    :param hyp: the predicted text (ASR hypothesis)
    :param lang: language code (eg en, hi, ta)
    :param remove_words: Text file path or a list of strings to be removed from the ground truth and hypothesis. Can be used to discount stop words etc
    :param lexicon: Kaldi lexicon file  path or a {word:lexicon} dict. Will give prediction phone error rate (not the AM per)
    :param lm: Language model in ARPA format. Check examples for loading example
    :return: Dictionary containing all ASR metrics. 

    """
    

    results = metrics.compute_asr_measures(ref, hyp, lexicon=lexicon, lm=lm)
    
    if lm:
        results["ref_ppl"] = metrics._get_ppl(ref, lm)

    if remove_words:
        results_stopwords = metrics.compute_asr_measures(ref, hyp, words_to_filter=remove_words, lang=lang, lexicon=lexicon)
    
        results_lemma = metrics.compute_asr_measures(ref, hyp.replace("<UNK>", " "), words_to_filter=remove_words, lemmatize=True, lang=lang, lexicon=lexicon)

        return {"base": results, "stopwords": results_stopwords, "lemmatized": results_lemma}
    
    else:
        results_lemma = metrics.compute_asr_measures(ref, hyp.replace("<UNK>", " "), lemmatize=True, lang=lang, lexicon=lexicon)
        
        return {"base": results, "lemmatized": results_lemma}
        


def _parse_alters(ref:str, hyp: Union[str, List], lang: str, remove_words: List = None, lexicon: Dict = None, lm=None) -> Dict:
    """
    Give ASR metrics for a reference string and a list of hypotheses (plural of hypothesis). Can parse kaldi-serve/gasr alternatives

    :param ref: ground truth
    :param hyp: List of the predicted text (ASR hypothesis). kaldi-serve alternatives are valid
    :param lang: language code (eg en, hi, ta)
    :param remove_words: Text file path or a list of strings to be removed from the ground truth and hypothesis. Can be used to discount stop words etc
    :param lexicon: Kaldi lexicon file  path or a {word:lexicon} dict. Will give prediction phone error rate (not the AM per)
    :param lm: Language model in ARPA format. Check examples for loading example
    :return: JSON string containing all ASR metrics. 

    """
    results = {}
    alternatives = []

    if ref not in [' ', ''] and len(hyp)==0:
        hyp.extend(['']*10)

    if len(hyp) > 0 and type(hyp[0]) == str:
        for alter in hyp:
            alternatives.append({"hyp": alter, **_parse_string(ref=ref, hyp=alter, lang=lang, remove_words=remove_words, lexicon=lexicon, lm=lm)})
    
    elif len(hyp) > 0 and type(hyp[0]) == list:
        for alter in hyp[0]:
            alternatives.append({"hyp": alter["transcript"], **_parse_string(ref=ref, hyp=alter["transcript"], lang=lang, remove_words=remove_words, lexicon=lexicon, lm=lm)})
    results["ref"] = ref

    if lm:
        results["ref_ppl"] = metrics._get_ppl(ref, lm)
    results["alternatives"] = alternatives

    if len(hyp) > 0:
        results.update(_get_top_n(alternatives))

        results.update(_get_delta(results["alternatives"], lang))

    return results
    


def _get_top_n(alternatives: Union[Dict, List]) -> Dict:
    """
    Get an average of the first n alternatives
    :param alternatives: List of ASR metric results
    :return: Dictionary with first n averages
    """
    
    top = {n: metrics.aggregate_metrics(alternatives[:n]) for n in [3, 5, 7, 10]}
       
    return {"top_3": top[3], "top_5": top[5], "top_7": top[7], "avg": top[10]}


def _get_delta(alternatives: Union[Dict, List], lang: str) -> Dict:
    """
    Get difference between first, fifth and last alternatives
    :param alternatives: List of parsed alternatives
    :param lang: Language of prediction
    :return: Dictionary containing 1v5, 1v10 and 5v10 prediction metrics
    """

    return {"1v5": _parse_string(ref=alternatives[0]["hyp"], hyp=alternatives[len(alternatives)//2]["hyp"], lang=lang), "1v10": _parse_string(ref=alternatives[0]["hyp"], hyp=alternatives[-1]["hyp"], lang=lang), "5v10": _parse_string(ref=alternatives[len(alternatives)//2]["hyp"], hyp=alternatives[-1]["hyp"], lang=lang)}


def _get_max_vote(truth: str, alternatives: Union[Dict, List], lang: str) -> Dict:
    ...



def _get_am_errors(post: List, align: List):
    """
    Uses forced alignments and phone posteriors to get AM frame error rate. 
    :param post: parsed phone posteriors
    :param align: parsed force aligned phones
    :return: AM frame error rate of type float
    """
    return metrics._get_am_errors(align, post)




def parse_phone_posterior(phone_post: List) -> Dict:
    """
    Parse a list of strings containing phone posteriors.
    :param phone_post; List of phone_post strings
    :return: Dictionary with {uuid:phone_post} mapping. 
    """
    result = {}
    for utt in phone_post:
        
        if utt not in ["", " "]:
            uuid = utt.split(" ", 1)[0]
            temp_post = [x.strip() for x in utt.split(" ", 1)[1].replace("[", "-").replace("]", "-").split("-") if x not in ["", " "]]
            post = [list(zip([int(x) for x in frame.split()[0::2]], [float(x) for x in frame.split()[1::2]])) for frame in temp_post]
            post = [max(x, key=lambda x: x[1])[0] for x in post]
            result[uuid] = post

    return result


def parse_alignments(alignments: List) -> Dict:
    """
    Parse a list of strings containing forcefully aligned phone posteriors.
    :param phone_post; List of aligned phone_post strings
    :return: Dictionary with {uuid:align_phone_post} mapping. 
    """
    result = {}
    for utt in alignments:
        if utt not in ["", " "]:
            uuid = utt.split(" ", 1)[0]
            phones = utt.split(" ", 1)[1].strip().split()
            result[uuid] = phones
    return result