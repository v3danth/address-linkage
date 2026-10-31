import polars as pl

from address_linkage import text


def apply(expr, values):
    return pl.DataFrame({'v': values}).select(expr(pl.col('v')).alias('v'))['v'].to_list()


def test_clean_address_drops_family_names_pincode_and_repeats():
    got = apply(text.clean_address, ['S/O Ramesh Kumar, 12 Gandhi Road, Allahabad, UP 211003, 12 Gandhi Road, Allahabad'])
    assert got == ['12 gandhi road, prayagraj, up']


def test_clean_address_maps_synonyms():
    assert apply(text.clean_address, ['Near Shiv Mandir, Gali 4']) == ['near shiv temple, lane 4']


def test_without_parts_drops_localities():
    assert apply(lambda e: text.without_parts(e, ['dhanbad', 'jharkhand']), ['kusum vihar, dhanbad, jharkhand']) == ['kusum vihar']


def test_house_number_counts_leading_or_cued_numbers_only():
    cases = {'216 DC Colony, Bhiwani': '216', 'No. 253, Pearl City': '253', 'F - 329, Sarita Vihar': 'f-329',
             'H.No 12, Gali 4': '12', 'Ward no - 13, Saja Road': '', 'Sector 21, Gurugram': '', 'Near Sapna Nursing Home': ''}
    assert apply(text.house_number, list(cases)) == list(cases.values())


def test_place_words_drop_filler_and_numbers():
    assert apply(text.place_words, ['vill rampur po sonpur dist gaya 12b']) == [['rampur', 'sonpur', 'gaya']]


def test_spelling_aliases_merge_variants_within_a_pincode_only():
    def vowels_removed(expr):
        return expr.str.replace_all('[aeiou]', '')

    frame = pl.DataFrame({'id': ['A', 'B', 'C', 'D'], 'pincode': ['111111', '111111', '111111', '222222'],
                          'address': ['ferozepur jhirka', 'ferozepur jhirka', 'firozpur jhirka', 'firozpur jhirka']})
    aliases = text.spelling_aliases(frame, vowels_removed)
    assert aliases.rows() == [('111111', 'firozpur', 'ferozepur')]
    fixed = text.apply_aliases(frame, aliases, ['address']).sort('id')
    assert fixed['address'].to_list() == ['ferozepur jhirka'] * 3 + ['firozpur jhirka']


def test_clean_person_name_removes_honorifics():
    assert apply(text.clean_person_name, ['Late Shri Ram Kumar', 'Smt. Sunita Devi']) == ['ram kumar', 'sunita devi']
