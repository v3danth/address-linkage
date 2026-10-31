"""Command line: python -m address_linkage <step>.

  export       addresses -> work/data (challenge format) with address-rule training labels
  train        run the ml-challenge pipeline on work/data
  score        final pair scores -> work/pairs.parquet (and work/model_pairs.parquet, model alone)
  evaluate     family-based evaluation -> work/metrics.json
  report       work/report.html
  all          export, train, score, evaluate, report
  gold-sample  draw the gold set -> work/gold/gold_sample.json
  gold-page    blind labelling page -> work/gold/gold_labelling.html
  gold-score   score outputs on labelled gold data: gold-score LABELS.csv
"""
import argparse
import json
import logging

import polars as pl

from . import config
from . import evaluation
from . import export
from . import gold
from . import pipeline
from . import report
from . import scoring
from . import source

log = logging.getLogger('address_linkage')


def addresses():
    return pl.read_parquet(config.path('addresses.parquet'))


def heldout():
    return set(pl.read_csv(config.path('heldout_pincodes.csv'), infer_schema=False)['pincode'].to_list())


def fmt(t):
    return f'{t[0]:6.1%} [{t[1]:5.1%}, {t[2]:5.1%}]'


def score():
    model = pipeline.pair_scores()
    a, people = addresses(), source.people()
    plain = model.with_columns(pl.lit(None, pl.String).alias('rule'), pl.lit(False).alias('generic'),
                               pl.col('model_score').alias('score'))
    plain.with_columns(scoring.relations(plain, people)).write_parquet(config.path('model_pairs.parquet'))
    final = scoring.final_pairs(model, a, people)
    final.write_parquet(config.path('pairs.parquet'))
    log.info('pairs scored: pairs=%d score>=0.5=%d relations=%s', final.height, final.filter(pl.col('score') >= 0.5).height,
             dict(final.filter(pl.col('score') >= config.REPORT_MIN_SCORE).group_by('relation').len().iter_rows()))


def evaluate():
    a, people, held = addresses(), source.people(), heldout()
    pairs = pl.read_parquet(config.path('pairs.parquet'))
    m = evaluation.evaluate(pairs, a, people, held)
    config.path('metrics.json').write_text(json.dumps(m, indent=1))
    print(f'sibling pairs in one pincode: {m["sibling_pairs"]} (held-out pincodes {m["sibling_pairs_heldout"]})')
    print(f'  found (score >= 0.5)   {fmt(m["recall"])}   held-out {fmt(m["recall_heldout"])}')
    print(f'  ranked #1 neighbour    {fmt(m["hit1"])}   held-out {fmt(m["hit1_heldout"])}')
    print(f'  in top 3               {fmt(m["hit3"])}   MRR {m["mrr"]:.3f}')
    print('family share of output pairs (lower bound on precision):')
    for k in ('family_share_rules', 'family_share_0.8+', 'family_share_0.5-0.8'):
        print(f'  {k[13:]:12} {fmt(m[k])}')
    outputs = {'model alone': pl.read_parquet(config.path('model_pairs.parquet')), 'model + rules': pairs}
    table = evaluation.recall_at_k(outputs, a, people, held)
    print('siblings inside the top-K pairs (all pincodes / held-out):')
    for name, row in table.items():
        print(f'  {name:14}' + ''.join(f'  K={k:<5} {v[0]:5.1%} / {v[1]:5.1%}' for k, v in row.items()))


def make_report():
    metrics = json.loads(config.path('metrics.json').read_text()) if config.path('metrics.json').exists() else {}
    html = report.build(pl.read_parquet(config.path('pairs.parquet')), addresses(), source.people(), metrics, 'address-linkage')
    config.path('report.html').write_text(html, encoding='utf-8')
    log.info('report written: %s', config.path('report.html'))


def gold_sample():
    g = gold.sample([pl.read_parquet(config.path('pairs.parquet')), pl.read_parquet(config.path('model_pairs.parquet'))], addresses())
    config.path('gold').mkdir(parents=True, exist_ok=True)
    config.path('gold', 'gold_sample.json').write_text(json.dumps(g, indent=1))
    log.info('gold sample: people=%d candidates=%d', len(g['sample']), sum(len(s['candidates']) for s in g['sample']))


def gold_page():
    g = json.loads(config.path('gold', 'gold_sample.json').read_text())
    config.path('gold', 'gold_labelling.html').write_text(gold.page(g, source.people(), addresses()), encoding='utf-8')
    log.info('labelling page written: %s', config.path('gold', 'gold_labelling.html'))


def gold_score(labels):
    g = json.loads(config.path('gold', 'gold_sample.json').read_text())
    outputs = {'model + rules': pl.read_parquet(config.path('pairs.parquet')),
               'model alone': pl.read_parquet(config.path('model_pairs.parquet'))}
    r = gold.score(g, pl.read_csv(labels, infer_schema=False), outputs)
    print(f'complete gold people: {r["complete"]} of {r["of"]}')
    for name, tasks in r['results'].items():
        for task, v in tasks.items():
            print(f'  {name:14} {task:9} test macro F0.5 {v["test"][0]:.4f} [{v["test"][1]:.4f}, {v["test"][2]:.4f}]  '
                  f'dev {v["dev"]:.4f}  policy {v["policy"]}')


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
    ap = argparse.ArgumentParser(prog='address_linkage', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=['export', 'train', 'score', 'evaluate', 'report', 'all', 'gold-sample', 'gold-page', 'gold-score'])
    ap.add_argument('labels', nargs='?', help='gold labels CSV (gold-score only)')
    args = ap.parse_args()
    config.WORK.mkdir(parents=True, exist_ok=True)
    steps = {'export': export.build, 'train': pipeline.train, 'score': score, 'evaluate': evaluate, 'report': make_report,
             'gold-sample': gold_sample, 'gold-page': gold_page}
    if args.step == 'gold-score':
        if not args.labels:
            ap.error('gold-score needs the labels CSV')
        gold_score(args.labels)
    elif args.step == 'all':
        for name in ('export', 'train', 'score', 'evaluate', 'report'):
            steps[name]()
    else:
        steps[args.step]()


if __name__ == '__main__':
    main()
