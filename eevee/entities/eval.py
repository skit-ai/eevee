from typing import Any, Dict, List, Tuple

import pandas as pd

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
    "people": ["people", "number"]
}

TEST_FNS = {
    "date": ord_datetime.date_eq_lists,
    "time": ord_datetime.time_eq_lists,
    "datetime": ord_datetime.datetime_eq_lists,
    "people": ord_people.eq_lists
}


def collect_entity_errors(items: List[Tuple[Id, Truth, Pred]],
                          entity_type: str) -> Dict:
    if entity_type == "location":
        return collect_atlas_errors(items)
    elif entity_type in ["time", "people", "date", "datetime"]:
        return collect_duckling_errors(items, entity_type)
    else:
        raise NotImplementedError


def collect_atlas_errors(items: List[Tuple[Id, Truth, Pred]]) -> Dict:
    """
    This is 4 types of basic location errors:

    1. misfires
    2. nofires
    3. mismatches_solved (mismatches solved by dtmf)
    4. mismatches_unsolved
    """

    misfires = []
    nofires = []
    mismatches_solved = []
    mismatches_unsolved = []
    truecounts = 0

    for id, truth, pred in items:
        pred = [ent for ent in pred if ent["type"] == "location"]

        """
        HACK: For using negative location. location-negative has `type`
         "LOCATION-PRESENT", and `value` as True/False
         Since we do not know if the predicted value is correct or not.
         We only add counts to nofires/misfires in our analysis.
        """
        if truth and all(e["type"] == "LOCATION-PRESENT" for e in truth):
            if all(e["values"][0]["value"] for e in truth) and len(pred) == 0:
                truecounts += 1
                nofires.append((id, truth, pred))
            elif any(not e["values"][0]["value"] for e in truth) and len(pred) > 0:
                misfires.append((id, truth, pred))
            else:
                pass

        elif all(e["type"] == "location" for e in truth):
            if truth:
                truecounts += 1
                if pred:
                    if not ord_location.eq_lists(truth, pred):
                        if len(pred) < 5 and ord_location.superset_list(pred, truth):
                            mismatches_solved.append((id, truth, pred))
                        else:
                            mismatches_unsolved.append((id, truth, pred))

                else:
                    nofires.append((id, truth, pred))
            else:
                if not ord_location.eq_lists(truth, pred):
                    misfires.append((id, truth, pred))
        else:
            print("Not Handling cases where both `location` and \
                  `LOCATION-PRESENT` are present.")

    return {
        "misfires": misfires,
        "nofires": nofires,
        "mismatches_solved": mismatches_solved,
        "mismatches_unsolved": mismatches_unsolved,
        "truecounts": truecounts
    }


def collect_duckling_errors(items: List[Tuple[Id, Truth, Pred]],
                            entity_type: str) -> Dict:
    """
    Find three basic entity errors:

    1. misfires
    2. nofires
    3. mismatches
    """
    test_fn = TEST_FNS[entity_type]

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
    if entity_type == "location":
        return atlas_report(total, errors)
    elif entity_type in ["time", "date", "datetime", "people"]:
        return duckling_report(total, errors)


def duckling_report(total: int, errors: Dict) -> pd.DataFrame:
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


def atlas_report(total: int, errors: Dict) -> pd.DataFrame:
    """
    Return metrics for entity identification. `errors` has the following items.

    - `total` is the number of cases
    - `truecounts` tell how many cases had the desired entity
    - `misfires` are the cases where we found an entity without
      there actually being one.
    - `nofires` are cases where we missed identifying entity
    - `mismatches_solved` are where our entity results is a superset of truth results and length of result is less than dtmf
    - `mismatches_unsolved` are reverse of the above
    """

    truecounts = errors["truecounts"]
    n_misfires = len(errors["misfires"])
    n_nofires = len(errors["nofires"])
    n_mismatches_solved = len(errors["mismatches_solved"])
    n_mismatches_unsolved = len(errors["mismatches_unsolved"])

    nullcounts = total - truecounts

    return pd.DataFrame({
        "": ["total", "truecounts", "misfires", "nofires", "mismatches_solved", "mismatches_unsolved", "nofires + mismatches"],
        "counts": [
            total,
            f"{truecounts}/{total}",
            f"{n_misfires}/{nullcounts}",
            f"{n_nofires}/{truecounts}",
            f"{n_mismatches_solved}/{truecounts}",
            f"{n_mismatches_unsolved}/{truecounts}",
            f"{n_nofires + n_mismatches_solved + n_mismatches_unsolved}/{truecounts}"
        ],
        "percent": [
            100 * i for i in [
                1, truecounts / total,
                (n_misfires / nullcounts) if nullcounts else 0,
                n_nofires / truecounts,
                n_mismatches_solved / truecounts,
                n_mismatches_unsolved / truecounts,
                (n_nofires + n_mismatches_solved +
                 n_mismatches_unsolved) / truecounts
            ]
        ]
    })


def entity_difference_report(errors: Dict, prev_errors: Dict,
                             error_types: List = ["misfires", "nofires", "mismatches"]) -> pd.DataFrame:
    counts = []
    for error_type in error_types:
        errors_added = py_.difference_by(errors[error_type], prev_errors[error_type], lambda k: k[1].id)
        errors_removed = py_.difference_by(prev_errors[error_type], errors[error_type], lambda k: k[1].id)

        counts.append(f"+{len(errors_added)}/-{len(errors_removed)}")

    return pd.DataFrame({
        "": error_types,
        "counts": counts
    })
