from typing import Any, Dict, List, Tuple

import pandas as pd
import pydash as py_
import eevee.entities.ord.datetime as ord_datetime
import eevee.entities.ord.location as ord_location
import eevee.entities.ord.people as ord_people

Id = Any
Truth = List[Any]
Pred = List[Any]

EQ_TYPES = {
    "time": ["datetime", "time"],
    "date": ["datetime", "date"],
    "datetime": ["datetime", "time", "date"],
    "people": ["people", "number"],
    "location": ["location"]
}

EQ_FNS = {
    "date": ord_datetime.date_eq_lists,
    "time": ord_datetime.time_eq_lists,
    "datetime": ord_datetime.datetime_eq_lists,
    "people": ord_people.eq_lists,
    "location": ord_location.eq_lists
}

SUPERSET_FNS = {
    "location": ord_location.superset_list,
    "time": ord_datetime.time_superset_list
}


def collect_entity_errors(items: List[Tuple[Id, Truth, Pred]],
                          entity_type: str) -> Dict:
    if entity_type in ["location", "time"]:
        return collect_dtmf_entity_errors(items, entity_type)
    elif entity_type in ["people", "date", "datetime"]:
        return collect_non_dtmf_errors(items, entity_type)
    else:
        raise NotImplementedError


def collect_dtmf_entity_errors(items: List[Tuple[Id, Truth, Pred]], entity_type) -> Dict:
    """
    In case of entities whose resolution can be solved using DTMF.
    This is 4 types of basic errors:

    1. misfires
    2. nofires
    3. mismatches_solved (mismatches solved by dtmf)
    4. mismatches_unsolved
    """
    eq_fn = EQ_FNS[entity_type]
    superset_fn = SUPERSET_FNS[entity_type]

    misfires = []
    nofires = []
    mismatches_solved = []
    mismatches_unsolved = []
    exceed_dtmf_nofires = []
    truecounts = 0

    for id, truth, pred in items:
        pred = [ent for ent in pred if ent["type"] in EQ_TYPES[entity_type]]

        if truth:
            truecounts += 1
            if pred:

                if len(py_.uniq([ent["type"] for ent in pred])) > 1 or any(ent["values"][0].get("type") == "interval" for ent in pred):
                    # HACK: Since we are only worried about location and time
                    #  for now. In case of location there is only one type
                    #  also it don't have ent type as interval, therefore
                    #  location evaluation wont come here.
                    if not eq_fn([truth[0]], [pred[0]]):
                        mismatches_unsolved.append((id, truth, pred))

                elif not eq_fn(truth, pred):
                    # Entities would be of same type and non-interval
                    # HACK: Hard match for just time
                    if len(pred) >= 5 and entity_type == "time":
                        exceed_dtmf_nofires.append((id, truth, pred))
                    elif len(pred) < 5 and superset_fn(pred, truth):
                        mismatches_solved.append((id, truth, pred))
                    else:
                        mismatches_unsolved.append((id, truth, pred))

            else:
                nofires.append((id, truth, pred))
        else:
            if not eq_fn(truth, pred):
                misfires.append((id, truth, pred))

    return {
        "misfires": misfires,
        "nofires": nofires,
        "mismatches_solved": mismatches_solved,
        "mismatches_unsolved": mismatches_unsolved,
        "exceed_dtmf_nofires": exceed_dtmf_nofires,
        "truecounts": truecounts
    }


def collect_non_dtmf_errors(items: List[Tuple[Id, Truth, Pred]],
                            entity_type: str) -> Dict:
    """
    Entities which are not dtmf in case of confusion.
    Find three basic entity errors:

    1. misfires
    2. nofires
    3. mismatches
    """
    test_fn = EQ_FNS[entity_type]

    misfires = []
    nofires = []
    mismatches = []
    truecounts = 0

    for id, truth, pred in items:
        pred = [ent for ent in pred if ent["type"] in EQ_TYPES[entity_type]]

        if truth:
            truecounts += 1
            if pred:
                if not test_fn(truth, pred):
                    mismatches.append((id, truth, pred))
            else:
                nofires.append((id, truth, pred))
        else:
            if not test_fn(truth, pred):
                misfires.append((id, truth, pred))

    return {
        "misfires": misfires,
        "nofires": nofires,
        "mismatches": mismatches,
        "truecounts": truecounts
    }


def entity_report(total: int, errors: Dict, entity_type: str) -> pd.DataFrame:
    if entity_type in ["location", "time"]:
        return dtmf_report(total, errors)
    elif entity_type in ["date", "datetime", "people"]:
        return non_dtmf_report(total, errors)


def non_dtmf_report(total: int, errors: Dict) -> pd.DataFrame:
    """
    Return metrics for entity identification. `errors` has the following items.

    - `total` is the number of cases
    - `truecounts` tell how many cases had the desired entity
    - `misfires` are the cases where we found an entity without
      there actually being one.
    - `nofires` are cases where we missed identifying entity
    - `mismatches` are where our entity results didn't match the truth

    TODO: As of now, the misfires will be underestimated since we are taking
          more than the scope for an entity while tagging. This should be fixed
          by narrowing down the states a little.
    """

    truecounts = errors["truecounts"]
    n_misfires = len(errors["misfires"])
    n_nofires = len(errors["nofires"])
    n_mismatches = len(errors["mismatches"])

    nullcounts = total - truecounts

    return pd.DataFrame({
        "": ["total", "truecounts", "misfires", "nofires",
             "mismatches", "nofires + mismatches"],
        "counts": [
            total,
            f"{truecounts}/{total}",
            f"{n_misfires}/{nullcounts}",
            f"{n_nofires}/{truecounts}",
            f"{n_mismatches}/{truecounts}",
            f"{n_nofires + n_mismatches}/{truecounts}"
        ],
        "percent": [
            100 * i for i in [
                1, truecounts / total,
                (n_misfires / nullcounts) if nullcounts else 0,
                n_nofires / truecounts,
                n_mismatches / truecounts,
                (n_nofires + n_mismatches) / truecounts
            ]
        ]
    })


def dtmf_report(total: int, errors: Dict) -> pd.DataFrame:
    """
    Return metrics for entity identification. `errors` has the following items.

    - `total` is the number of cases
    - `truecounts` tell how many cases had the desired entity
    - `misfires` are the cases where we found an entity without
      there actually being one.
    - `nofires` are cases where we missed identifying entity
    - `mismatches_solved` are where our entity results is a superset of truth
        results and length of result is less than dtmf
    - `mismatches_unsolved` are reverse of the above
    - `exceed_dtmf_nofires` are cases when we do not convey anything to user
        because of #predictions exceed dtmf limit (i.e. 4)
    """

    truecounts = errors["truecounts"]
    n_misfires = len(errors["misfires"])
    n_nofires = len(errors["nofires"])
    n_mismatches_solved = len(errors["mismatches_solved"])
    n_mismatches_unsolved = len(errors["mismatches_unsolved"])
    n_exceed_dtmf_nofires = len(errors["exceed_dtmf_nofires"])

    nullcounts = total - truecounts

    return pd.DataFrame({
        "": ["total", "truecounts", "misfires", "nofires", "mismatches_solved",
             "mismatches_unsolved", "exceed_dtmf_nofires", "nofires + mismatches"],
        "counts": [
            total,
            f"{truecounts}/{total}",
            f"{n_misfires}/{nullcounts}",
            f"{n_nofires}/{truecounts}",
            f"{n_mismatches_solved}/{truecounts}",
            f"{n_mismatches_unsolved}/{truecounts}",
            f"{n_exceed_dtmf_nofires}/{truecounts}",            
            f"{n_nofires + n_mismatches_solved + n_mismatches_unsolved + n_exceed_dtmf_nofires}/{truecounts}"
        ],
        "percent": [
            100 * i for i in [
                1, truecounts / total,
                (n_misfires / nullcounts) if nullcounts else 0,
                n_nofires / truecounts,
                n_mismatches_solved / truecounts,
                n_mismatches_unsolved / truecounts,
                n_exceed_dtmf_nofires / truecounts,
                (n_nofires + n_mismatches_solved +
                 n_mismatches_unsolved + n_exceed_dtmf_nofires) / truecounts
            ]
        ]
    })


def entity_difference_report(errors: Dict, prev_errors: Dict,
                             error_types: List = ["misfires", "nofires", "mismatches"]) -> pd.DataFrame:
    counts = []
    for error_type in error_types:
        errors_added = py_.difference_by(errors[error_type], prev_errors[error_type], lambda k: k[0])
        errors_removed = py_.difference_by(prev_errors[error_type], errors[error_type], lambda k: k[0])

        counts.append(f"+{len(errors_added)}/-{len(errors_removed)}")

    return pd.DataFrame({
        "": error_types,
        "counts": counts
    })
