import polars as pl

from address_linkage import rules
from address_linkage import text


def frame(rows):
    """rows: (id, pincode, street-level text, city)."""
    f = pl.DataFrame(rows, schema=['id', 'pincode', 'lines', 'city'], orient='row')
    return f.with_columns(text.clean_address(pl.concat_str(['lines', 'city'], separator=', ')).alias('address'),
                          text.clean_address(pl.col('lines')).alias('lines')).with_columns(
        text.house_number(pl.col('lines')).alias('hn'), text.place_words(pl.col('lines')).alias('words'))


def pairs(found):
    return sorted(zip(found['id_a'].to_list(), found['id_b'].to_list()))


def test_exact_address_needs_a_specific_address():
    f = rules.with_keys(frame([('A', '1', 'kusum vihar sector four', 'dhanbad'), ('B', '1', 'kusum vihar sector four', 'dhanbad'),
                               ('C', '1', 'dhanbad', 'dhanbad'), ('D', '1', 'dhanbad', 'dhanbad')]))
    assert pairs(rules.exact_pairs(f)) == [('A', 'B')]


def test_identical_address_of_many_people_is_generic_not_exact():
    rows = [(f'P{i}', '1', 'iit campus hostel road', 'dhanbad') for i in range(7)]
    f = rules.with_keys(frame(rows))
    assert rules.exact_pairs(f).is_empty()
    assert rules.generic_pairs(f).height == 21


def test_same_house_survives_wrong_city_text():
    f = rules.with_keys(frame([('A', '1', '216 dc colony bhiwani', 'bhiwani'), ('B', '1', '216 dc colony bhiwani', 'delhi'),
                               ('C', '1', '217 dc colony bhiwani', 'bhiwani')]))
    assert pairs(rules.same_house_pairs(f)) == [('A', 'B')]


def test_contained_address_when_one_side_wrote_more():
    f = rules.with_keys(frame([('A', '1', 'vill rampur po sonpur dist gaya', 'gaya'),
                               ('B', '1', 'vill rampur po sonpur ps belaganj dist gaya near shiv temple', 'gaya'),
                               ('C', '1', 'vill kurtha po sonpur dist gaya', 'gaya')]))
    assert pairs(rules.contained_pairs(f)) == [('A', 'B')]


def test_contained_address_rejects_conflicting_house_numbers():
    f = rules.with_keys(frame([('A', '1', '12 nehru nagar main road', 'patna'), ('B', '1', '14 nehru nagar main road ward six', 'patna')]))
    assert rules.contained_pairs(f).is_empty()


def test_place_name_must_be_rare_and_local():
    rows = [('A', '1', '43 pearl city chandapura road', 'bengaluru'), ('B', '1', '253 pearl city sarjapura road', 'bengaluru'),
            ('C', '2', 'gomti nagar extension', 'lucknow')]
    assert pairs(rules.place_name_pairs(rules.with_keys(frame(rows)))) == [('A', 'B')]


def test_rule_pairs_keep_the_strongest_level():
    f = frame([('A', '1', 'kusum vihar sector four', 'dhanbad'), ('B', '1', 'kusum vihar sector four', 'dhanbad')])
    found = rules.rule_pairs(f)
    assert found.select('id_a', 'id_b', 'level').rows() == [('A', 'B', 'exact address')]
