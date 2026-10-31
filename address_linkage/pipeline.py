"""The boundary with the ml-challenge pipeline: train it with run.py, and read its pair scores back."""
import importlib
import json
import logging
import os
import subprocess
import sys

import polars as pl

from . import config

log = logging.getLogger(__name__)


def challenge_module(name):
    """A module of the ml-challenge pipeline (block, run, …), imported from ML_CHALLENGE_SRC."""
    src = str(config.ML_CHALLENGE_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    return importlib.import_module(name)


def train():
    """Run the full challenge pipeline (encoder, blocking, Stage A/B/C) on work/data."""
    cmd = [sys.executable, '-u', str(config.ML_CHALLENGE_SRC / 'run.py'), '--prepared', str(config.path('pipeline', 'none')),
           '--model', config.ENCODER, '--work', str(config.path('pipeline')), '--out', str(config.path('pipeline_out')),
           '--data', str(config.path('data')), '--config', str(config.PIPELINE_CONFIG)]
    log.info('training: encoder=%s work=%s', config.ENCODER, config.path('pipeline'))
    subprocess.run(cmd, check=True, env={**os.environ, 'TOKENIZERS_PARALLELISM': 'false'})


def pair_scores():
    """Stage C probability for every candidate pair of different people: id_a, id_b, pincode, model_score.
    A pair is scored from both sides (a as query, b as owner and the reverse); the larger score is kept."""
    run = challenge_module('run')  # imports torch's OpenMP runtime before LightGBM's, as run.py itself does
    work = config.path('pipeline')
    cfg = json.loads(config.PIPELINE_CONFIG.read_text())
    cfg['workers'] = cfg['threads'] = cfg.get('threads') or os.cpu_count()
    chunks = sorted((work / 'pruned' / 'test').glob('*.parquet'))
    if not chunks:
        raise FileNotFoundError(f'no scored candidates in {work}; run the train step first')
    scores, _, _ = run.stages_bc(cfg, work, 'test', chunks, run.load_models(work / 'models' / 'B'), run.load_models(work / 'models' / 'C'))
    queries = pl.read_parquet(work / 'prepared' / 'test_q.parquet', columns=['entity_id', 'country']).with_row_index('qi')
    owners = pl.read_parquet(work / 'prepared' / 'test_s1.parquet', columns=['entity_id']).with_row_index('ri')
    pairs = (scores.join(queries.select('qi', pl.col('entity_id').str.slice(3).alias('a'), pl.col('country').alias('pincode')), on='qi')
             .join(owners.select('ri', pl.col('entity_id').str.slice(3).alias('b')), on='ri'))
    return (pairs.select(pl.min_horizontal('a', 'b').alias('id_a'), pl.max_horizontal('a', 'b').alias('id_b'), 'pincode', 'pC')
            .group_by('id_a', 'id_b', 'pincode').agg(pl.col('pC').max().alias('model_score')))
