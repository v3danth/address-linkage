"""Final pair scores: the trained model's score, raised where an address rule is certain, and a relation.

Deterministic evidence beats a learned probability: an identical specific address is 1.0, the same house
number on the same street is at least 0.95, one address containing the other at least 0.9. Identical text
shared by many people (campus, hostel) is a generic address, not a household.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from . import rules

FLOOR = {'exact address': 1.0, 'same house': 0.95, 'same address (more detail)': 0.9}
NAME_MATCH = 90
FATHER_MATCH = 85


def apply_rules(model, addresses):
    """model: id_a, id_b, pincode, model_score. Adds rule pairs the model never proposed."""
    f = rules.with_keys(addresses)
    found = [rules.exact_pairs(f).with_columns(pl.lit('exact address').alias('rule')),
             rules.same_house_pairs(f).with_columns(pl.lit('same house').alias('rule')),
             rules.contained_pairs(f).with_columns(pl.lit('same address (more detail)').alias('rule'))]
    order = {name: i for i, name in enumerate(FLOOR)}
    best = (pl.concat(found).with_columns(pl.col('rule').replace_strict(order, return_dtype=pl.Int8).alias('o'))
            .sort('o').group_by('id_a', 'id_b', 'pincode', maintain_order=True).first().drop('o'))
    generic = rules.generic_pairs(f).with_columns(pl.lit(True).alias('generic'))
    out = (model.join(best, on=['id_a', 'id_b', 'pincode'], how='full', coalesce=True)
           .join(generic, on=['id_a', 'id_b', 'pincode'], how='left').with_columns(pl.col('generic').fill_null(False)))
    floor = pl.col('rule').replace_strict(FLOOR, default=0.0, return_dtype=pl.Float64)
    return out.with_columns(pl.max_horizontal(pl.col('model_score').fill_null(0.0), floor).alias('score'))


def same_person(a, b, facts):
    """A matching name alone is weak ("Pratik Kumar" is common), and a matching father proves family, not
    one person: siblings share a father and often have similar names. The date of birth decides; without
    one on both sides the pair stays uncertain."""
    (dob_a, father_a), (dob_b, father_b) = facts.get(a, ('', '')), facts.get(b, ('', ''))

    def known(dob):
        return bool(dob) and not dob.startswith('0000')

    if known(dob_a) and known(dob_b):
        return 'same person' if dob_a == dob_b else 'same name, different person'
    if father_a and father_b and fuzz.token_sort_ratio(father_a, father_b) < FATHER_MATCH:
        return 'same name, different person'
    return 'same person?'


def relations(pairs, people):
    """One person with two records (e.g. two admissions) looks like two people at one address, so a matching
    name is checked first. Different people who share a name (siblings often do) keep their address relation."""
    names = dict(zip(people['id'].to_list(), people['name'].str.to_lowercase().to_list()))
    facts = dict(zip(people['id'].to_list(), zip(people['dob'].to_list(), people['father'].str.to_lowercase().to_list())))
    a, b = pairs['id_a'].to_list(), pairs['id_b'].to_list()
    sim = cpdist([names.get(x, '') for x in a], [names.get(x, '') for x in b], scorer=fuzz.token_sort_ratio, workers=-1)
    by_address = np.where(pairs['generic'].to_numpy(), 'generic address', pairs['rule'].fill_null('neighbour').to_numpy())
    # two blank names compare as identical, so both sides must have a name
    match = (sim >= NAME_MATCH) & np.array([bool(names.get(x)) and bool(names.get(y)) for x, y in zip(a, b)], dtype=bool)
    named = np.array([same_person(x, y, facts) if m else '' for x, y, m in zip(a, b, match)], dtype=object)
    duplicate = match & (named != 'same name, different person')
    return pl.Series('relation', np.where(duplicate, named, by_address).tolist(), dtype=pl.String)


def final_pairs(model, addresses, people):
    """id_a, id_b, pincode, score, model_score, relation, generic for every scored or rule-linked pair."""
    scored = apply_rules(model, addresses)
    return scored.with_columns(relations(scored, people)).select('id_a', 'id_b', 'pincode', 'score', 'model_score', 'relation', 'generic')
