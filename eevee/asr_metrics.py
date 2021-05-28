import json
from typing import Union, List, Dict
from collections import Counter

from eevee import metrics


def get_metrics(x: Dict, lang: str, remove_words: Union[str, List] = None, lexicon: Union[str, Dict] = None) -> Dict:
    if remove_words and type(remove_words) == str:
        with open(remove_words) as fin:
            remove_words = fin.read().split('\n')

    if lexicon and type(lexicon) == str:
        with open(lexicon) as fin:
            lex = fin.read().split('\n')
            lexicon = {x.split(' ', 1)[0]: x.split(' ', 1)[1] for x in lex}

    if type(x['hyp']) == str:
        results = parse_string(x, lang, remove_words, lexicon)

    elif type(x['hyp']) == list:
        results = parse_list(x, lang, remove_words, lexicon)

    elif type(x['hyp']) == dict:
        results = parse_alters(x, lang, remove_words, lexicon)

    return results


def parse_string(x: Dict, lang: str, remove_words: List = None, lexicon: Dict = None) -> Dict:

    results = metrics.compute_measures(x['ref'], x['hyp'], lexicon=lexicon)
    
    if remove_words:
        results_stopwords = metrics.compute_measures(x['ref'], x['hyp'], words_to_filter=remove_words, lang=lang, lexicon=lexicon)
    
        results_lemma = metrics.compute_measures(x['ref'], x['hyp'], words_to_filter=remove_words, lemmatize=True, lang=lang, lexicon=lexicon)

        return {'base': results, 'stopwords': results_stopwords, 'lemmatized': results_lemma}
    
    else:
        results_lemma = metrics.compute_measures(x['ref'], x['hyp'], lemmatize=True, lang=lang, lexicon=lexicon)
        
        return {'base': results, 'lemmatized': results_lemma}
        


def parse_list(x: Dict, lang: str, remove_words: List = None, lexicon: Dict = None) -> Dict:
    results = []
    alternatives = []
    for hyp in x['hyp']:
        alternatives.append({'hyp': hyp, **parse_string(x['ref'], hyp, remove_words=remove_words, lexicon=lexicon)})
    
    results['ref'] = x['ref']
    results['alternatives'] = alternatives

    results.update(get_top_n(alternatives))

    results.update(get_delta(x['hyp'], lang))

    
    return results


def parse_alters(x: Dict, lang: str) -> Dict:
    ...


def get_top_n(alternatives: Union[Dict, List]) -> Dict:
    
    top = {3: Counter(), 5: Counter(), 7: Counter(), 10: Counter()}
    
    for i, x in alternatives:    
        if i<3:
            top[3].update(x)
        
        if i<5:
            top[5].update(x)
        
        if i<7:
            top[7].update(x)
        
        top[10].update(x)
        
    for x in top.keys():
        H, S, I, D = top[x]['H'], top[x]['S'], top[x]['I'], top[x]['D']
        top[x] /= x
        top[x]['H'], top[x]['S'], top[x]['I'], top[x]['D'] = H, S, I, D
    
    return {'top_3': top[3], 'top_5': top[5], 'top_7': top[7], 'avg': top[10]}


def get_delta(alternatives: Union[Dict, List], lang: str) -> Dict:

    return {'1v5': parse_string(alternatives[0], alternatives[4]), '1v10': parse_string(alternatives[0], alternatives[9]), '5v10': parse_string(alternatives[4], alternatives[9])}


def get_max_vote(truth: str, alternatives: Union[Dict, List], lang: str) -> Dict:
    ...

