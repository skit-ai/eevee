def are_entities_superset(ss_entities, sb_entities):
    """
    Check if a group of entities are a proper superset of other group. It uses
    zeroth_value, type and parser to perform this check.
    """
    tmp_ss_entities = [{"value": e["values"], "type": e["type"], "parser": e["parser"]}
                       for e in ss_entities]

    tmp_sb_entities = [{"value": e["values"], "type": e["type"], "parser": e["parser"]}
                       for e in sb_entities]

    if len(tmp_sb_entities) != len(tmp_ss_entities):
        if all(e in tmp_ss_entities for e in tmp_sb_entities):
            return True

    return False
