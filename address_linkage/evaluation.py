"""Evaluate an output against an independent answer key built from family data, never used by the model.

Siblings (both parents' names match, different date of birth, same permanent pincode) should share an
address, so they measure recall (scored >= 0.5) and ranking (is the sibling a person's #1 neighbour).
The family share of output pairs is a lower bound on precision: neighbours need not be family.
Every rate carries a 95% interval from a bootstrap over pincodes, because pairs in one pincode are not
independent. Results are aggregates only.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from . import rules
from . import text

PIN = r'^[1-9]\d{5}$'
PLACEHOLDERS = ['na', 'n a', 'none', 'nil', 'not available']
PARENT_MATCH = 90
# A parent-name pair shared by more families than this is not evidence of siblinghood.
PARENT_PAIR_MAX = 6
BOOTSTRAP = 1000
NOT_LINKED = 10_000


def with_parents(people):
    """People with both parents' names recorded, cleaned."""
    p = people.with_columns(text.clean_person_name(pl.col('father')).alias('father'),
                            text.clean_person_name(pl.col('mother')).alias('mother'),
                            pl.col('last').str.to_lowercase().str.strip_chars().alias('last'))
    known = lambda c: (pl.col(c).str.len_chars() >= 4) & ~pl.col(c).is_in(PLACEHOLDERS)  # noqa: E731
    return p.filter(known('father') & known('mother'))


def sibling_pairs(people, addresses):
    """Sibling pairs living in the same permanent pincode: id_a, id_b, pincode."""
    p = with_parents(people).join(addresses.select('id', 'pincode'), on='id')
    p = p.with_columns(pl.col('father').str.slice(0, 3).alias('bf'), pl.col('mother').str.slice(0, 3).alias('bm'))
    c = p.join(p, on=['pincode', 'bf', 'bm'], suffix='_b').filter(pl.col('id') < pl.col('id_b'))
    father = cpdist(c['father'].to_list(), c['father_b'].to_list(), scorer=fuzz.token_sort_ratio, workers=-1)
    mother = cpdist(c['mother'].to_list(), c['mother_b'].to_list(), scorer=fuzz.token_sort_ratio, workers=-1)
    c = c.filter(pl.Series((father >= PARENT_MATCH) & (mother >= PARENT_MATCH)) & (pl.col('dob') != pl.col('dob_b')))
    c = c.filter(pl.len().over('father', 'mother') <= PARENT_PAIR_MAX)
    return c.select(pl.col('id').alias('id_a'), 'id_b', 'pincode')


def bootstrap(frame, statistic, rng):
    """Point estimate and 95% interval, resampling pincodes with replacement."""
    if frame.is_empty():
        return float('nan'), float('nan'), float('nan')
    groups = frame.partition_by('pincode')
    draws = [statistic(pl.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])) for _ in range(BOOTSTRAP)]
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(statistic(frame)), float(lo), float(hi)


def best_sibling_rank(pairs, siblings):
    """For each person with a sibling: the rank of their best-ranked sibling among all their scored pairs."""
    ranked = (rules.both_directions(pairs, 'score')
              .with_columns(pl.col('score').rank('ordinal', descending=True).over('id_a').alias('rank')).drop('score'))
    return (rules.both_directions(siblings, 'pincode', 'held').join(ranked, on=['id_a', 'id_b'], how='left')
            .group_by('id_a', 'pincode', 'held').agg(pl.col('rank').min().fill_null(NOT_LINKED).alias('rank')))


def evaluate(pairs, addresses, people, held):
    """Headline metrics, each as (estimate, low, high)."""
    rng = np.random.default_rng(0)
    neighbours = pairs.filter(~pl.col('relation').str.starts_with('same person') & (pl.col('relation') != 'generic address'))
    sib = sibling_pairs(people, addresses).with_columns(pl.col('pincode').is_in(sorted(held)).alias('held'))
    found = (sib.join(neighbours.select('id_a', 'id_b', 'score'), on=['id_a', 'id_b'], how='left')
             .with_columns(pl.col('score').fill_null(0.0)))
    ranks = best_sibling_rank(neighbours, sib)
    m = {'sibling_pairs': sib.height, 'sibling_pairs_heldout': int(sib['held'].sum()),
         'recall': bootstrap(found, lambda d: (d['score'] >= 0.5).mean(), rng),
         'recall_heldout': bootstrap(found.filter(pl.col('held')), lambda d: (d['score'] >= 0.5).mean(), rng),
         'hit1': bootstrap(ranks, lambda d: (d['rank'] <= 1).mean(), rng),
         'hit1_heldout': bootstrap(ranks.filter(pl.col('held')), lambda d: (d['rank'] <= 1).mean(), rng),
         'hit3': bootstrap(ranks, lambda d: (d['rank'] <= 3).mean(), rng),
         'mrr': float((1 / ranks['rank']).mean())}
    known = set(with_parents(people)['id'].to_list())
    fam = sib.select('id_a', 'id_b', pl.lit(True).alias('family'))
    checkable = (neighbours.filter(pl.col('id_a').is_in(list(known)) & pl.col('id_b').is_in(list(known)))
                 .join(fam, on=['id_a', 'id_b'], how='left').with_columns(pl.col('family').fill_null(False)))
    for name, cond in (('family_share_rules', pl.col('relation').is_in(['exact address', 'same house', 'same address (more detail)'])),
                       ('family_share_0.8+', (pl.col('relation') == 'neighbour') & (pl.col('score') >= 0.8)),
                       ('family_share_0.5-0.8', (pl.col('relation') == 'neighbour') & pl.col('score').is_between(0.5, 0.8, closed='left'))):
        m[name] = bootstrap(checkable.filter(cond), lambda d: d['family'].mean(), rng)
    m['pairs_0.5+'] = neighbours.filter(pl.col('score') >= 0.5).height
    return m


def recall_at_k(named, addresses, people, held, ks=(500, 1000, 2000, 4000)):
    """Fair comparison between outputs calibrated differently: of the sibling pairs, the share inside each
    output's top-K pairs (all pincodes, held-out pincodes)."""
    sib = sibling_pairs(people, addresses)
    sib_held = sib.filter(pl.col('pincode').is_in(sorted(held)))
    table = {}
    for name, pairs in named.items():
        ranked = (pairs.filter(~pl.col('relation').str.starts_with('same person') & (pl.col('relation') != 'generic address'))
                  .sort('score', descending=True).select('id_a', 'id_b'))
        table[name] = {k: (ranked.head(k).join(sib, on=['id_a', 'id_b']).height / max(1, sib.height),
                           ranked.head(k).join(sib_held, on=['id_a', 'id_b']).height / max(1, sib_held.height)) for k in ks}
    return table
