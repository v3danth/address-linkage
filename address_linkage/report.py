"""The local HTML report: overview, people with ranked neighbours and evidence, pairs to review,
households and duplicate records. It contains names and home addresses, so it is never published."""
import json
from importlib import resources

import polars as pl
from rapidfuzz import fuzz

from . import config
from . import evaluation
from . import rules

STRONG = ('exact address', 'same house', 'same address (more detail)')
HOUSEHOLD_MAX = 12


def tier(relation, score):
    if relation in STRONG or score >= 0.8:
        return 'high'
    return 'medium' if score >= 0.5 else 'low'


def households(links):
    """Connected groups over strong links; one larger than HOUSEHOLD_MAX is a building, not a household."""
    parent = {}

    def root(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in links:
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[ra] = rb
    groups = {}
    for x in parent:
        groups.setdefault(root(x), []).append(x)
    return sorted((sorted(g) for g in groups.values() if len(g) >= 2), key=len, reverse=True)


def evidence(pair, rank_a, rank_b, landmark, house, surname_equal):
    parts = {'exact address': ['identical address'], 'same house': [f'house no {house} on the same street'],
             'same address (more detail)': ['one address contains the other']}.get(pair['relation'], [])
    if landmark:
        parts.append(f'both near {landmark}')
    if rank_a == 1 and rank_b == 1:
        parts.append("each other's best match")
    if surname_equal:
        parts.append('same surname')
    if pair['model_score'] is not None:
        parts.append(f'model {pair["model_score"]:.2f}')
    return ' · '.join(parts)


def build(pairs, addresses, people, metrics, label):
    """The report as one self-contained HTML string."""
    pairs = pairs.filter(pl.col('relation') != 'generic address')
    both = rules.both_directions(pairs, 'score').with_columns(pl.col('score').rank('ordinal', descending=True).over('id_a').alias('rank'))
    rank = dict(zip(zip(both['id_a'].to_list(), both['id_b'].to_list()), both['rank'].to_list()))
    lm = rules.landmark_pairs(addresses)
    landmark = dict(zip(zip(lm['id_a'].to_list(), lm['id_b'].to_list()), lm['landmark'].to_list()))
    house = dict(zip(addresses['id'].to_list(), addresses['hn'].to_list()))
    surname = dict(zip(people['id'].to_list(), people['last'].str.to_lowercase().str.strip_chars().to_list()))
    parents = evaluation.with_parents(people)
    known = {i: (f, m) for i, f, m in parents.select('id', 'father', 'mother').iter_rows()}

    def family(a, b):
        if a not in known or b not in known:
            return 'unknown'
        (fa, ma), (fb, mb) = known[a], known[b]
        match = min(fuzz.token_sort_ratio(fa, fb), fuzz.token_sort_ratio(ma, mb)) >= evaluation.PARENT_MATCH
        return 'yes' if match else 'no'

    keep = pairs.filter((pl.col('score') >= config.REPORT_MIN_SCORE) | pl.col('relation').str.starts_with('same person'))
    rows = []
    for r in keep.iter_rows(named=True):
        a, b = r['id_a'], r['id_b']
        ra, rb = rank.get((a, b)), rank.get((b, a))
        rows.append({'a': a, 'b': b, 'pin': r['pincode'], 's': round(float(r['score']), 3), 'rel': r['relation'],
                     'tier': tier(r['relation'], r['score']), 'ra': ra, 'rb': rb, 'fam': family(a, b),
                     'ev': evidence(r, ra, rb, landmark.get((a, b)), house.get(a) or '?',
                                    bool(surname.get(a)) and surname.get(a) == surname.get(b))})
    ids = sorted({x['a'] for x in rows} | {x['b'] for x in rows})
    names = dict(zip(people['id'].to_list(), people['name'].to_list()))
    where = dict(zip(addresses['id'].to_list(), zip(addresses['address'].to_list(), addresses['pincode'].to_list())))
    data = {'label': label, 'metrics': metrics, 'household_max': HOUSEHOLD_MAX, 'pairs': rows,
            'people': {i: {'n': names.get(i, ''), 'ad': where.get(i, ('', ''))[0], 'pin': where.get(i, ('', ''))[1]} for i in ids},
            'households': households([(x['a'], x['b']) for x in rows if x['rel'] in STRONG])}
    template = resources.files('address_linkage').joinpath('templates', 'report.html').read_text(encoding='utf-8')
    return template.replace('__DATA__', json.dumps(data, ensure_ascii=False).replace('</', '<\\/'))
