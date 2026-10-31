"""Address and person-name normalisation: pure Polars expressions and frame functions."""
import polars as pl

PINCODE = r'\b[1-9]\d{5}\b'

# Renamed cities and Hindi/English synonyms: one spelling per place, or the same place looks like two.
ALIASES = {
    'allahabad': 'prayagraj', 'gurgaon': 'gurugram', 'bangalore': 'bengaluru', 'calcutta': 'kolkata',
    'bombay': 'mumbai', 'madras': 'chennai', 'poona': 'pune', 'baroda': 'vadodara', 'mysore': 'mysuru',
    'trivandrum': 'thiruvananthapuram', 'banaras': 'varanasi', 'benares': 'varanasi', 'orissa': 'odisha',
    'mandir': 'temple', 'masjid': 'mosque', 'marg': 'road', 'gali': 'lane', 'chauk': 'chowk',
    'aawas': 'awas', 'avas': 'awas', 'muhalla': 'mohalla', 'mohala': 'mohalla',
}

# "s/o ramesh kumar", "d/o …", "w/o …", "c/o …": a family or guardian name inside an address. It must
# never reach the model, and it makes siblings look alike for the wrong reason.
RELATION = r'(?:\b(?:s|d|w|c)\s*[/\\.]\s*o\b\.?|\b(?:son|daughter|wife|husband) of\b|\bcare of\b)[^,]*'

# Words that locate an address without saying which place it is.
FILLER = ['near', 'nr', 'po', 'ps', 'post', 'vill', 'village', 'gram', 'dist', 'distt', 'district', 'at', 'via', 'opp',
          'behind', 'tehsil', 'teh', 'thana', 'the', 'and', 'of', 'in', 'no', 'h', 'house', 'flat', 'plot', 'qr', 'qtr',
          'quarter', 'door', 'hno', 'p', 'o', 's', 'c', 'w', 'd']

# A house number starts the address ("216, …", "no. 253, …", "f-329 …") or follows a house cue.
# Ward, sector, block and phase numbers are shared by hundreds of homes and never count.
UNIT = r'([a-z]{0,3}\s*[-/]?\s*\d+[a-z]?(?:\s*[-/]\s*[a-z0-9]+)?)'
HOUSE_CUE = (r'\b(?:h\s*\.?\s*no|house\s*(?:no|number)?|flat\s*(?:no)?|plot\s*(?:no)?|qr|qtr|quarter\s*(?:no)?|door\s*no|'
             r'd\s*\.?\s*no|bungalow\s*(?:no)?)\b\s*[\.:#\-]*\s*')
NOT_HOUSE = r'^(?:ward|sector|sec|block|phase|gali|lane|road|street)\b'

# Landmarks: the words after a cue, up to the next comma ("near mlb medical college").
LANDMARK_CUE = r'\b(?:near|nr|behind|opp|opposite|in front of|beside|besides|adjacent to|next to)\b\.?\s*'
GENERIC_LANDMARKS = ['temple', 'mosque', 'school', 'post office', 'bus stand', 'railway station', 'hospital', 'bank',
                     'market', 'park', 'water tank', 'police station', 'petrol pump', 'church', 'chowk', 'station',
                     'road', 'college']

HONORIFICS = r'\b(?:mr|mrs|ms|miss|shri|sri|shree|smt|late|lt|dr|prof|er|sh|sm)\b'


def canonical_places(expr):
    for old, new in ALIASES.items():
        expr = expr.str.replace_all(r'\b' + old + r'\b', new)
    return expr


def strip_relations(expr):
    return expr.str.replace_all(RELATION, ' ')


def dedupe_parts(expr):
    """Addresses often repeat themselves ("a, b, c, a, b, c"); keep each comma part once, in order."""
    return (expr.str.split(',').list.eval(pl.element().str.strip_chars()).list.eval(pl.element().filter(pl.element() != ''))
            .list.unique(maintain_order=True).list.join(', '))


def clean_address(expr):
    """Lower case, no relation phrases or pincode, one spelling per place, no repeated parts."""
    text = canonical_places(strip_relations(expr.str.to_lowercase()))
    text = text.str.replace_all(PINCODE, ' ').str.replace_all(r'[\t\r\n]+', ' ').str.replace_all(r'\s*,\s*', ', ')
    return dedupe_parts(text).str.replace_all(r'\s+', ' ').str.strip_chars(' ,')


def without_parts(expr, common):
    """Drop the comma parts listed in `common` (towns, districts, states)."""
    return expr.str.split(', ').list.eval(pl.element().filter(~pl.element().is_in(common))).list.join(', ')


def house_number(expr):
    """The house number of an address, or '' when it has none."""
    text = expr.str.to_lowercase().str.strip_chars(' ,.#')
    lead = text.str.extract(r'^(?:no\s*\.?\s*[:#\-]*\s*|#\s*)?' + UNIT + r'\b', 1)
    cued = text.str.extract(HOUSE_CUE + UNIT + r'\b', 1)
    hn = pl.when(cued.is_not_null()).then(cued).otherwise(pl.when(~text.str.contains(NOT_HOUSE)).then(lead))
    return hn.str.replace_all(r'\s+', '').fill_null('')


def place_words(expr):
    """Words that say which place an address is (street, colony, complex, village), in order:
    no filler words and no tokens containing digits (house numbers)."""
    return expr.str.extract_all(r'[a-z0-9]+').list.eval(pl.element().filter(
        ~pl.element().is_in(FILLER) & (pl.element().str.len_chars() >= 2) & ~pl.element().str.contains(r'\d')))


def landmarks(frame):
    """id, pincode, landmark for every landmark named in frame['address']."""
    return (frame.select('id', 'pincode', pl.col('address').str.extract_all(LANDMARK_CUE + r'([^,;]{4,40})').alias('landmark'))
            .explode('landmark').drop_nulls('landmark')
            .with_columns(pl.col('landmark').str.replace(LANDMARK_CUE, '').str.replace_all(r'[^a-z0-9 ]', ' ')
                          .str.replace_all(r'\s+', ' ').str.strip_chars())
            .filter((pl.col('landmark').str.len_chars() >= 6) & ~pl.col('landmark').is_in(GENERIC_LANDMARKS)
                    & (pl.col('landmark').str.split(' ').list.len() >= 2))
            .unique())


def spelling_aliases(frame, phonetic):
    """Within one pincode, words that sound alike and are spelt alike are one place written two ways
    ("firozpur" / "ferozpur"): map each to the most frequent spelling in that pincode.
    Short sound skeletons collide ("nehru" and "near" are both "nr"), so both tests are required."""
    from rapidfuzz import fuzz
    from rapidfuzz.process import cpdist

    words = (frame.select('id', 'pincode', pl.col('address').str.extract_all(r'[a-z]{4,}').alias('word'))
             .explode('word').drop_nulls('word').with_columns(phonetic(pl.col('word')).alias('sound')))
    counts = words.group_by('pincode', 'sound', 'word').agg(pl.col('id').n_unique().alias('n'))
    best = (counts.sort(['n', 'word'], descending=[True, False]).group_by('pincode', 'sound', maintain_order=True).first()
            .select('pincode', 'sound', pl.col('word').alias('to')))
    alias = counts.join(best, on=['pincode', 'sound']).filter(pl.col('word') != pl.col('to'))
    similar = cpdist(alias['word'].to_list(), alias['to'].to_list(), scorer=fuzz.ratio, workers=-1) >= 80
    return alias.filter(pl.Series(similar) & (pl.col('sound').str.len_chars() >= 4)).select('pincode', 'word', 'to')


def apply_aliases(frame, aliases, columns):
    """Rewrite aliased words in `columns`, pincode by pincode."""
    if aliases.is_empty():
        return frame
    affected = aliases['pincode'].unique()
    parts = [frame.filter(~pl.col('pincode').is_in(affected.implode()))]
    for (pincode,), group in frame.filter(pl.col('pincode').is_in(affected.implode())).group_by(['pincode']):
        rows = aliases.filter(pl.col('pincode') == pincode)
        exprs = []
        for col in columns:
            expr = pl.col(col)
            for word, to in zip(rows['word'].to_list(), rows['to'].to_list()):
                expr = expr.str.replace_all(r'\b' + word + r'\b', to)
            exprs.append(expr.alias(col))
        parts.append(group.with_columns(exprs))
    return pl.concat(parts)


def clean_person_name(expr):
    """Lower case letters only, no honorifics ("Late Shri Ram Kumar" -> "ram kumar")."""
    return (expr.str.to_lowercase().str.replace_all(r'[^a-z]+', ' ').str.replace_all(HONORIFICS, ' ')
            .str.replace_all(r'\s+', ' ').str.strip_chars())
