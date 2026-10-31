"""The gold set: ground truth labelled by a person, scored like the challenge leaderboard.

sample  draws people in strata by the score of their best candidate (obvious, hard, and "nobody nearby"),
        with every plausible candidate pooled from the outputs given, so the truth does not depend on one model
page    a blind labelling page: no scores, candidates shuffled
score   F0.5 per person (a person with no true match scores 1 only if nothing is predicted, as in the
        challenge), averaged with stratum weights so the sample stands for everyone; the decoding policy
        (threshold, or decode.expected_f_policy on scores calibrated on dev) is chosen on dev, reported on test
"""
import hashlib
import json
import random
from importlib import resources

import numpy as np
import polars as pl

from . import config
from . import pipeline
from . import rules

STRATA = [('rule match', 0.95, 2.0), ('score 0.8-0.95', 0.8, 0.95), ('score 0.5-0.8', 0.5, 0.8),
          ('score 0.3-0.5', 0.3, 0.5), ('score 0.1-0.3', 0.1, 0.3), ('no candidate >= 0.1', -1.0, 0.1)]
PER_STRATUM = 25
MAX_CANDIDATES = 8
MIN_CANDIDATES = 4
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
EXPECTED_F = [(m, floor) for m in (0.0, 0.1, 0.3) for floor in (0.05, 0.1, 0.2)]
TASKS = {'household': {'house'}, 'neighbour': {'house', 'near'}}


def split_of(pincode):
    """dev or test by a stable hash, so one pincode never sits on both sides."""
    return 'dev' if int(hashlib.md5(pincode.encode()).hexdigest(), 16) % 2 == 0 else 'test'


def sample(outputs, addresses):
    """outputs: final pair frames to pool candidates from. Returns {'strata': […], 'sample': […]}."""
    pool = (pl.concat([o.select('id_a', 'id_b', 'score', 'relation') for o in outputs])
            .filter(~pl.col('relation').str.starts_with('same person'))
            .group_by('id_a', 'id_b').agg(pl.col('score').max()))
    both = rules.both_directions(pool, 'score').rename({'id_a': 'p', 'id_b': 'c'}).sort('score', descending=True)
    best = (addresses.select(pl.col('id').alias('p'), 'pincode')
            .join(both.group_by('p').agg(pl.col('score').max().alias('best')), on='p', how='left')
            .with_columns(pl.col('best').fill_null(-0.5)))
    rng = np.random.default_rng(config.SEED)
    strata, people = [], []
    for name, lo, hi in STRATA:
        members = best.filter((pl.col('best') >= lo) & (pl.col('best') < hi))
        taken = members.sample(min(PER_STRATUM, members.height), seed=int(rng.integers(1 << 31)))
        strata.append({'name': name, 'population': members.height, 'sampled': taken.height})
        people += [{'person': p, 'pincode': pin, 'stratum': name, 'split': split_of(pin)}
                   for p, pin in zip(taken['p'].to_list(), taken['pincode'].to_list())]
    top = {}
    for p, c in zip(both['p'].to_list(), both['c'].to_list()):
        if len(top.setdefault(p, [])) < MAX_CANDIDATES:
            top[p].append(c)
    grouped = addresses.group_by('pincode').agg(pl.col('id'))
    members = dict(zip(grouped['pincode'].to_list(), grouped['id'].to_list()))
    for s in people:
        cands = top.get(s['person'], [])
        others = [m for m in members.get(s['pincode'], []) if m != s['person'] and m not in cands]
        rng.shuffle(others)
        # people with few candidates get pincode neighbours added, so "nobody near" is a real judgement
        s['candidates'] = cands + others[:max(0, MIN_CANDIDATES - len(cands))]
    return {'strata': strata, 'sample': people}


def page(gold, people, addresses):
    """The labelling page as one self-contained HTML string (personal data: local use only)."""
    names = dict(zip(people['id'].to_list(), people['name'].to_list()))
    where = dict(zip(addresses['id'].to_list(), addresses['address'].to_list()))
    rnd = random.Random(7)
    order = gold['sample'][:]
    rnd.shuffle(order)  # strata interleaved, so labelling fatigue is spread evenly
    items = []
    for s in order:
        cands = s['candidates'][:]
        rnd.shuffle(cands)
        items.append({'p': s['person'], 'n': names.get(s['person'], ''), 'a': where.get(s['person'], ''), 'pin': s['pincode'],
                      'c': [{'id': c, 'n': names.get(c, ''), 'a': where.get(c, '')} for c in cands]})
    template = resources.files('address_linkage').joinpath('templates', 'gold.html').read_text(encoding='utf-8')
    return template.replace('__DATA__', json.dumps(items, ensure_ascii=False).replace('</', '<\\/'))


def person_f05(predicted, truth):
    if not truth:
        return 1.0 if not predicted else 0.0
    return 1.25 * len(predicted & truth) / (0.25 * len(truth) + len(predicted))


def isotonic(x, y):
    """Pool-adjacent-violators: a monotone map from score to probability, fitted on dev labels."""
    order = np.argsort(x)
    blocks = []
    for xi, yi in zip(np.asarray(x)[order], np.asarray(y, float)[order]):
        blocks.append([yi, 1.0, xi])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, hi = blocks.pop()
            v1, w1, _ = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, hi])
    upper, value = np.array([b[2] for b in blocks]), np.array([b[0] for b in blocks])
    return lambda s: value[np.clip(np.searchsorted(upper, s, side='left'), 0, len(value) - 1)]


def weighted_macro(rows, weights):
    w = np.array([weights[r['stratum']] for r in rows])
    return float((w * np.array([r['f'] for r in rows])).sum() / w.sum()) if rows else float('nan')


def score(gold, labels, outputs):
    """labels: person, candidate, label (house / near / far / unknown). outputs: {name: final pairs}.
    Returns {name: {task: {'test': (estimate, low, high), 'dev': float, 'policy': …}}}."""
    expected_f_policy = pipeline.challenge_module('decode').expected_f_policy
    lab = {(p.upper(), c.upper()): v for p, c, v in labels.select('person', 'candidate', 'label').iter_rows()}
    complete = [{**s, 'labels': {c: lab[(s['person'], c)] for c in s['candidates']}}
                for s in gold['sample'] if all((s['person'], c) in lab for c in s['candidates'])]
    done = {}
    for s in complete:
        done[s['stratum']] = done.get(s['stratum'], 0) + 1
    weights = {st['name']: st['population'] / done[st['name']] for st in gold['strata'] if done.get(st['name'])}
    rng = np.random.default_rng(1)
    results = {}
    for name, pairs in outputs.items():
        s_ = {}
        for a, b, v in pairs.select('id_a', 'id_b', 'score').iter_rows():
            s_[(a, b)] = s_[(b, a)] = max(v, s_.get((a, b), 0.0))
        results[name] = {}
        for task, positive in TASKS.items():
            dev = [(s_.get((p['person'], c), 0.0), label in positive) for p in complete if p['split'] == 'dev'
                   for c, label in p['labels'].items() if label != 'unknown']
            calibrate = isotonic([d[0] for d in dev], [d[1] for d in dev]) if dev else (lambda v: v)

            def rows(policy, split):
                out = []
                for p in complete:
                    if p['split'] != split:
                        continue
                    known = {c: label for c, label in p['labels'].items() if label != 'unknown'}
                    truth = {c for c, label in known.items() if label in positive}
                    cand = [(c, s_.get((p['person'], c), 0.0)) for c in known]
                    if policy[0] == 'threshold':
                        predicted = {c for c, v in cand if v >= policy[1]}
                    elif cand:
                        frame = pl.DataFrame({'qi': [c for c, _ in cand], 'ri': [p['person']] * len(cand),
                                              'p': [float(calibrate(np.array([v]))[0]) for _, v in cand]})
                        predicted = set(expected_f_policy(frame, m=policy[1], floor=policy[2])['qi'].to_list())
                    else:
                        predicted = set()
                    out.append({'stratum': p['stratum'], 'f': person_f05(predicted, truth)})
                return out

            policies = [('threshold', t) for t in THRESHOLDS] + [('expected_f', m, f) for m, f in EXPECTED_F]
            best = max(policies, key=lambda pol: weighted_macro(rows(pol, 'dev'), weights))
            test = rows(best, 'test')
            by = {}
            for r in test:
                by.setdefault(r['stratum'], []).append(r)
            draws = [weighted_macro([g[i] for g in by.values() for i in rng.integers(0, len(g), len(g))], weights) for _ in range(1000)]
            lo, hi = np.percentile(draws, [2.5, 97.5]) if draws else (float('nan'), float('nan'))
            results[name][task] = {'test': (weighted_macro(test, weights), float(lo), float(hi)),
                                   'dev': weighted_macro(rows(best, 'dev'), weights), 'policy': list(best)}
    return {'complete': len(complete), 'of': len(gold['sample']), 'results': results}
