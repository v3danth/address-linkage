"""Address-only rules that link two people. One implementation serves both the training labels and the
final scoring, so the model is trained and judged against the same definitions.

Input frame (from export.prepare): id, pincode, address, lines, hn, words.
Every function returns id_a < id_b, pincode.
"""
import polars as pl

from . import config
from . import text

# Strongest first; the order decides a query's single training owner.
LEVELS = ['exact address', 'same house', 'same address (more detail)', 'same place name', 'same landmark']
# A house number shared by more homes than this in one pincode is a block or ward number.
HOUSE_GROUP_MAX = 50


def both_directions(pairs, *columns):
    """Each unordered pair twice, as (id_a, id_b) and (id_b, id_a), so per-person views see all their pairs."""
    reverse = pairs.select(pl.col('id_b').alias('id_a'), pl.col('id_a').alias('id_b'), *columns)
    return pl.concat([pairs.select('id_a', 'id_b', *columns), reverse])


def ordered(frame, a='id', b='id_b'):
    return frame.select(pl.min_horizontal(a, b).alias('id_a'), pl.max_horizontal(a, b).alias('id_b'), 'pincode').unique()


def with_keys(frame):
    """Exact-address key, informative place words and how many people share each exact address."""
    return frame.with_columns(
        pl.col('address').str.extract_all(r'[a-z0-9]+').list.unique().list.sort().list.join(' ').alias('key'),
        pl.col('words').list.unique().alias('wset')).with_columns(
        pl.col('wset').list.len().alias('nw'), pl.len().over('pincode', 'key').alias('sharing'))


def exact_pairs(f):
    """Same pincode and the same words: but only a specific address (>= 3 place words, shared by at most
    GENERIC_MAX people) says "same home"."""
    spec = f.filter((pl.col('nw') >= 3) & (pl.col('sharing') <= config.GENERIC_MAX))
    return ordered(spec.join(spec, on=['pincode', 'key'], suffix='_b').filter(pl.col('id') < pl.col('id_b')))


def generic_pairs(f):
    """Identical address text shared by more than GENERIC_MAX people: a campus, hostel or institution."""
    g = f.filter(pl.col('sharing') > config.GENERIC_MAX)
    return ordered(g.join(g, on=['pincode', 'key'], suffix='_b').filter(pl.col('id') < pl.col('id_b')))


def same_house_pairs(f):
    """Same pincode, same house number, and >= 90% of the shorter address's place words shared: the same
    home even when city or state text is wrong or repeated."""
    h = f.filter((pl.col('hn') != '') & (pl.col('nw') >= 3)).with_columns(pl.len().over('pincode', 'hn').alias('group'))
    h = h.filter(pl.col('group') <= HOUSE_GROUP_MAX)
    h = h.join(h, on=['pincode', 'hn'], suffix='_b').filter(pl.col('id') < pl.col('id_b'))
    shared = pl.col('wset').list.set_intersection(pl.col('wset_b')).list.len()
    return ordered(h.filter(shared >= 0.9 * pl.min_horizontal('nw', 'nw_b')))


def contained_pairs(f):
    """Every place word of the shorter address appears in the longer one: one person wrote more detail.
    The shorter address must be specific (contained in at most GENERIC_MAX addresses of its pincode)
    and the house numbers must not conflict."""
    c = f.filter(pl.col('nw') >= 3)
    c = c.join(c, on='pincode', suffix='_b').filter((pl.col('id') != pl.col('id_b')) & (pl.col('nw') <= pl.col('nw_b')))
    c = c.filter(pl.col('wset').list.set_intersection(pl.col('wset_b')).list.len() == pl.col('nw'))
    c = c.with_columns(pl.len().over('id').alias('containers'))
    conflict = (pl.col('hn') != '') & (pl.col('hn_b') != '') & (pl.col('hn') != pl.col('hn_b'))
    return ordered(c.filter((pl.col('containers') <= config.GENERIC_MAX) & ~conflict))


def place_name_pairs(f):
    """A two-word place name (e.g. a housing complex) that occurs in one pincode only, for 2 to GENERIC_MAX people."""
    bigrams = (f.select('id', 'pincode', pl.col('words').list.eval(pl.element() + ' ' + pl.element().shift(-1)).alias('bg'))
               .explode('bg').drop_nulls('bg').filter(pl.col('bg').str.contains(r'^[a-z]{3,} [a-z]{3,}$')).unique())
    spread = bigrams.group_by('bg').agg(pl.col('pincode').n_unique().alias('pins'), pl.col('id').n_unique().alias('people'))
    rare = bigrams.join(spread.filter((pl.col('pins') == 1) & pl.col('people').is_between(2, config.GENERIC_MAX)), on='bg')
    return ordered(rare.join(rare, on=['pincode', 'bg'], suffix='_b').filter(pl.col('id') < pl.col('id_b')))


def landmark_pairs(f, max_people=30):
    """Two homes placed by the same landmark in one pincode ("near mlb medical college"); a landmark named
    by more than max_people says little."""
    lm = text.landmarks(f).filter(pl.len().over('pincode', 'landmark') <= max_people)
    pairs = lm.join(lm, on=['pincode', 'landmark'], suffix='_b').filter(pl.col('id') < pl.col('id_b'))
    return pairs.group_by('id', 'id_b', 'pincode').agg(pl.col('landmark').first()).rename({'id': 'id_a'})


def rule_pairs(frame):
    """Every rule link with its strongest level: id_a, id_b, pincode, level, rank (0 = strongest)."""
    f = with_keys(frame)
    found = [exact_pairs(f), same_house_pairs(f), contained_pairs(f), place_name_pairs(f), landmark_pairs(f).drop('landmark')]
    tagged = pl.concat([p.with_columns(pl.lit(i, pl.Int8).alias('rank')) for i, p in enumerate(found)])
    best = tagged.group_by('id_a', 'id_b', 'pincode').agg(pl.col('rank').min())
    return best.with_columns(pl.col('rank').replace_strict(dict(enumerate(LEVELS)), return_dtype=pl.String).alias('level'))
