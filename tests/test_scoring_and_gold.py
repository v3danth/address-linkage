import numpy as np
import polars as pl

from address_linkage import gold
from address_linkage import scoring


def test_date_of_birth_decides_same_person():
    facts = {'A': ('2001-05-04', 'ram kumar'), 'B': ('2001-05-04', 'r kumar'), 'C': ('1999-01-01', 'mohan lal'),
             'D': ('', 'ram kumar'), 'E': ('', 'mohan lal')}
    assert scoring.same_person('A', 'B', facts) == 'same person'
    assert scoring.same_person('A', 'C', facts) == 'same name, different person'
    assert scoring.same_person('A', 'D', facts) == 'same person?'
    assert scoring.same_person('A', 'E', facts) == 'same name, different person'


def test_siblings_with_similar_names_are_not_one_person():
    facts = {'A': ('2001-05-04', 'ram kumar'), 'B': ('2004-02-11', 'ram kumar')}
    assert scoring.same_person('A', 'B', facts) == 'same name, different person'


def test_similar_named_siblings_keep_their_address_relation():
    people = pl.DataFrame({'id': ['A', 'B'], 'name': ['Amit Kumar', 'Amit Kumar'], 'dob': ['2001-01-01', '2004-06-02'],
                           'father': ['Ram Kumar', 'Ram Kumar']})
    pairs = pl.DataFrame({'id_a': ['A'], 'id_b': ['B'], 'rule': ['same house'], 'generic': [False]})
    assert scoring.relations(pairs, people).to_list() == ['same house']


def test_blank_names_are_never_the_same_person():
    people = pl.DataFrame({'id': ['A', 'B'], 'name': ['', ''], 'dob': ['2001-01-01', '2001-01-01'], 'father': ['', '']})
    pairs = pl.DataFrame({'id_a': ['A'], 'id_b': ['B'], 'rule': [None], 'generic': [False]}, schema_overrides={'rule': pl.String})
    assert scoring.relations(pairs, people).to_list() == ['neighbour']


def test_rules_raise_scores_but_never_lower_them():
    model = pl.DataFrame({'id_a': ['A', 'C'], 'id_b': ['B', 'D'], 'pincode': ['1', '1'], 'model_score': [0.2, 0.99]})
    best = pl.DataFrame({'rule': ['same house', 'same address (more detail)']})
    floor = pl.col('rule').replace_strict(scoring.FLOOR, default=0.0, return_dtype=pl.Float64)
    scored = pl.concat([model, best], how='horizontal').with_columns(pl.max_horizontal(pl.col('model_score'), floor).alias('score'))
    assert scored['score'].to_list() == [0.95, 0.99]


def test_person_f05_follows_the_challenge_singleton_rule():
    assert gold.person_f05(set(), set()) == 1.0
    assert gold.person_f05({'x'}, set()) == 0.0
    assert gold.person_f05({'x'}, {'x'}) == 1.0
    assert abs(gold.person_f05({'x', 'y'}, {'x'}) - 1.25 / 2.25) < 1e-9


def test_isotonic_is_monotone_and_averages_violations():
    f = gold.isotonic([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1])
    out = f(np.array([0.1, 0.2, 0.3, 0.4]))
    assert list(out) == sorted(out)
    assert out[1] == out[2] == 0.5


def test_weighted_macro_uses_stratum_weights():
    rows = [{'stratum': 'a', 'f': 1.0}, {'stratum': 'b', 'f': 0.0}]
    assert gold.weighted_macro(rows, {'a': 3.0, 'b': 1.0}) == 0.75
