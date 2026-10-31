"""Turn home addresses into the ml-challenge input format, with address-rule training labels.

Every person is an S1 record (S1-<id>) and a query (S2-<id>); the pipeline's exclude_same_id option keeps
a person from being their own candidate. Names stay blank: the model reads addresses only. The pincode
fills the pipeline's "country" scope, so all comparisons stay within one pincode.
"""
import logging

import numpy as np
import polars as pl

from . import config
from . import pipeline
from . import rules
from . import source
from . import text

log = logging.getLogger(__name__)


def prepare(raw):
    """id, pincode, address (cleaned, no pincode), lines (street level only), hn, words."""
    a = raw.select('id', pl.col('pincode').str.strip_chars().alias('pincode'),
                   text.clean_address(pl.concat_str(['line1', 'line2', 'city', 'state'], separator=', ')).alias('address'),
                   text.clean_address(pl.concat_str(['line1', 'line2'], separator=', ')).alias('lines'))
    a = a.filter(pl.col('pincode').str.contains(r'^[1-9]\d{5}$') & (pl.col('address').str.len_chars() >= 8))
    aliases = text.spelling_aliases(a, pipeline.challenge_module('block').phonetic)
    a = text.apply_aliases(a, aliases, ['address', 'lines'])
    parts = a.select(pl.col('address').str.split(', ').alias('part')).explode('part').group_by('part').len()
    common = parts.filter(pl.col('len') >= config.LOCALITY_MIN)['part'].to_list()
    log.info('addresses prepared: people=%d pincodes=%d spelling_aliases=%d locality_parts=%d',
             a.height, a['pincode'].n_unique(), aliases.height, len(common))
    # rules compare only the specific part of an address; a shared town or district says nothing about the home
    return a.with_columns(text.house_number(pl.col('lines')).alias('hn'),
                          text.place_words(text.without_parts(pl.col('lines'), common)).alias('words')).sort('id')


def holdout_pincodes(pincodes):
    """A fixed random share of pincodes, never used for training."""
    unique = np.array(sorted(set(pincodes)))
    rng = np.random.default_rng(config.SEED)
    return set(rng.choice(unique, size=int(len(unique) * config.HOLDOUT_SHARE), replace=False).tolist())


def single_owners(pairs):
    """The pipeline gives each query exactly one owner: pick the strongest rule link, then the smaller id."""
    both = rules.both_directions(pairs, 'rank').rename({'id_a': 'query', 'id_b': 'owner'})
    return both.sort(['query', 'rank', 'owner']).group_by('query', maintain_order=True).first()


def records(frame, prefix):
    return frame.select((pl.lit(prefix) + pl.col('id')).alias('entity_id'), pl.lit('').alias('business_name'),
                        pl.col('address').alias('business_address'), pl.col('pincode').alias('country'))


def write_tsv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(path, separator='\t', quote_style='never')


def build():
    """Write work/data (challenge-format TSVs), work/addresses.parquet and work/heldout_pincodes.csv."""
    addresses = prepare(source.addresses())
    held = holdout_pincodes(addresses['pincode'].to_list())
    links = rules.rule_pairs(addresses).filter(~pl.col('pincode').is_in(sorted(held)))
    train_people = addresses.filter(~pl.col('pincode').is_in(sorted(held)))
    owners = single_owners(links).filter(pl.col('query').is_in(train_people['id'].implode()))
    truth = (owners.group_by('owner').agg((pl.lit('S2-') + pl.col('query')).sort().str.join(',').alias('matched_entity_ids'))
             .select((pl.lit('S1-') + pl.col('owner')).alias('source1_entity_id'), 'matched_entity_ids'))
    truth = (records(addresses, 'S1-').select(pl.col('entity_id').alias('source1_entity_id'))
             .join(truth, on='source1_entity_id', how='left').fill_null(''))

    data = config.path('data')
    write_tsv(records(addresses, 'S1-'), data / 'train' / 'train_source1.tsv')
    write_tsv(records(train_people, 'S2-'), data / 'train' / 'train_source2.tsv')
    write_tsv(records(train_people.head(0), 'S2-'), data / 'train' / 'train_source3.tsv')
    write_tsv(truth, data / 'train' / 'train_ground_truth.tsv')
    write_tsv(records(addresses, 'S1-'), data / 'test' / 'test_source1.tsv')
    write_tsv(records(addresses, 'S2-'), data / 'test' / 'test_source2.tsv')
    write_tsv(records(addresses.head(0), 'S2-'), data / 'test' / 'test_source3.tsv')
    addresses.write_parquet(config.path('addresses.parquet'))
    pl.DataFrame({'pincode': sorted(held)}).write_csv(config.path('heldout_pincodes.csv'))
    log.info('dataset written: train_queries=%d labelled_queries=%d heldout_pincodes=%d label_levels=%s',
             train_people.height, owners.height, len(held), dict(links.group_by('level').len().iter_rows()))
