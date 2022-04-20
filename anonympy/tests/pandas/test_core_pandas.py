import urllib
import pytest
import pandas as pd
from pandas import testing as pdt

from anonympy import __version__
from anonympy.pandas import dfAnonymizer
from anonympy.pandas.utils_pandas import load_dataset


@pytest.fixture(scope="module")
def anonym_small():
    df = load_dataset('small')
    anonym = dfAnonymizer(df)
    return anonym


@pytest.fixture(scope="module")
def anonym_big():
    try:
        df = load_dataset('big')
        anonym = dfAnonymizer(df)
    except urllib.error.HTTPError:
        anonym = None
    return anonym


def test_anonym_obj(anonym_small, anonym_big):
    assert isinstance(anonym_small, dfAnonymizer), "should have\
     returned `dfAnonymizer` object"
    if anonym_big is None:
        assert False, "Failed to fetch the DataFrame"
    assert isinstance(anonym_big, dfAnonymizer), "should have returned\
     `dfAnonymizer` object"


def test_numeric_noise(anonym_small):
    output = anonym_small.numeric_noise('age', seed=42, inplace=False)
    expected = pd.Series([38, 47], dtype='int64')
    pdt.assert_series_equal(expected, output, check_names=False)

    output = anonym_small.numeric_noise(['age', 'salary'],
                                        seed=42,
                                        inplace=False)
    expected = pd.DataFrame({'age': [38, 47],
                             'salary': [59239.79912097112, 49323.30756879504]})
    pdt.assert_frame_equal(expected, output)


def test_numeric_binning(anonym_small):
    output = anonym_small.numeric_binning('salary', bins=2, inplace=False)
    dtype = pd.CategoricalDtype([
        pd.Interval(49315.0, 54279.0, closed='right'),
        pd.Interval(54279.0, 59234.0, closed='right')],
        ordered=True)
    expected = pd.Series([
                    pd.Interval(54279.0, 59234.0, closed='right'),
                    pd.Interval(49315.0, 54279.0, closed='right')],
                    dtype=dtype)
    pdt.assert_series_equal(expected, output, check_names=False)

    output = anonym_small.numeric_binning(['age', 'salary'],
                                          bins=2,
                                          inplace=False)
    dtype2 = pd.CategoricalDtype([
                pd.Interval(33.0, 40.0, closed='right'),
                pd.Interval(40.0, 48.0, closed='right')],
                ordered=True)
    ser2 = pd.Series([
                pd.Interval(33.0, 40.0, closed='right'),
                pd.Interval(40.0, 48.0, closed='right')],
                dtype=dtype2)
    expected = pd.DataFrame({'age': ser2, 'salary': expected})
    pdt.assert_frame_equal(expected, output)


def test_numeric_masking(anonym_small):
    output = anonym_small.numeric_masking('age', inplace=False)
    expected = pd.Series([7.5, -7.5], dtype='float64')
    pdt.assert_series_equal(expected, output, check_names=False)

    output = anonym_small.numeric_masking(['age', 'salary'], inplace=False)
    expected = pd.DataFrame({'age': [-4954.900676201789, 4954.900676201798],
                             'salary': [5.840670901327418e-15,
                                        5.840670901327409e-15]})

    pdt.assert_frame_equal(expected, output)


def test_numeric_rounding(anonym_small):
    output = anonym_small.numeric_rounding('salary', inplace=False)
    expected = pd.Series([60000.0, 50000.0], dtype='float64')
    pdt.assert_series_equal(expected, output, check_names=False)

    output = anonym_small.numeric_rounding(['age', 'salary'], inplace=False)
    expected = pd.DataFrame({'age': {0: 30, 1: 50}, 'salary': {0: 60000.0,
                                                               1: 50000.0}})

    pdt.assert_frame_equal(expected, output)


@pytest.mark.skipif(__version__ == '0.2.4',
                    reason="Requires anonympy >= 0.2.5")
def test_categorical_fake(anonym_small):
    output = anonym_small.categorical_fake('name',
                                           locale=['en_US'],
                                           seed=42,
                                           inplace=False)
    expected = pd.Series(['Allison Hill', 'Noah Rhodes'])

    pdt.assert_series_equal(expected, output, check_names=False)

    output = anonym_small.categorical_fake(['name', 'email'],
                                           locale=['en_GB'],
                                           seed=42,
                                           inplace=False)
    expected = pd.DataFrame({'name': {0: 'Mr James Cox', 1: 'Nathan Davidson'},
                             'email': {0: 'jenningshenry@example.org',
                                       1: 'bradleynewman@example.net'}})
    pdt.assert_frame_equal(expected, output)

    output = anonym_small.categorical_fake({'name': 'name_female'},
                                           seed=42,
                                           inplace=False)
    expected = pd.Series(['Allison Hill', 'Nancy Rhodes'])
    pdt.assert_series_equal(expected, output, check_names=False)

    output = anonym_small.categorical_fake({'ssn': 'ssn', 'web': 'url'},
                                           seed=42,
                                           inplace=False)
    expected = pd.DataFrame({'ssn': {0: '655-15-0410', 1: '760-36-4013'},
                             'web': {0: 'http://www.hill.net/',
                                     1: 'http://johnson.com/'}})
    pdt.assert_frame_equal(expected, output)


def test_categorical_fake_auto(anonym_small):
    output = anonym_small.categorical_fake_auto(seed=42, inplace=False)
    expected = pd.DataFrame({'name': {0: 'Allison Hill', 1: 'Noah Rhodes'},
                             'email': {0: 'johnsonjoshua@example.org',
                                       1: 'jillrhodes@example.net'},
                             'ssn': {0: '655-15-0410', 1: '760-36-4013'}})
    pdt.assert_frame_equal(expected, output)


@pytest.mark.skipif(__version__ == '0.2.4',
                    reason="Requires anonympy >= 0.2.5")
def test_categorical_resampling(anonym_small):
    output = anonym_small.categorical_resampling('name',
                                                 inplace=False,
                                                 seed=42)
    expected = pd.Series(['Bruce', 'Tony'])
    pdt.assert_series_equal(expected, output, check_names=False)

    output = anonym_small.categorical_resampling(['web', 'ssn'],
                                                 seed=2,
                                                 inplace=False)
    expected = pd.DataFrame({'web':
                            {0: 'http://www.alandrosenburgcpapc.co.uk',
                             1: 'http://www.alandrosenburgcpapc.co.uk'},
                             'ssn': {0: '656564664', 1: '343554334'}})
    pdt.assert_frame_equal(expected, output)


@pytest.mark.skipif(__version__ == '0.2.4',
                    reason="Requires anonympy >= 0.2.5")
def test_categorical_tokenization(anonym_small):
    output = anonym_small.categorical_tokenization('name',
                                                   key='test',
                                                   inplace=False)
    expected = pd.Series(['45fe1a783c', 'bda8a41313'])
    pdt.assert_series_equal(expected, output, check_names=False)

    output = anonym_small.categorical_tokenization(['web', 'ssn'],
                                                   key='test',
                                                   inplace=False)
    expected = pd.DataFrame({'web': {0: 'e667d84f37', 1: '986a819ea2'},
                             'ssn': {0: '0f7c17cc6f', 1: 'f42ad34907'}})
    pdt.assert_frame_equal(expected, output)


def test_categorical_email_masking(anonym_small):
    output = anonym_small.categorical_email_masking('email', inplace=False)
    expected = pd.Series(['j*****r@owen.com', 'e*****n@lewis.com'])
    pdt.assert_series_equal(expected, output, check_names=False)

    output = anonym_small.categorical_email_masking(['email', 'email'],
                                                    inplace=False)
    expected = pd.DataFrame(
                    {'email': {0: 'j*****r@owen.com', 1: 'e*****n@lewis.com'}})
    pdt.assert_frame_equal(expected, output)


def test_datetime_noise(anonym_small):
    output = anonym_small.datetime_noise('birthdate', seed=42, inplace=False)
    expected = pd.Series([pd.Timestamp('1914-07-22 00:00:00'),
                          pd.Timestamp('1970-10-25 00:00:00')])
    pdt.assert_series_equal(expected, output, check_names=False)

    output = anonym_small.datetime_noise(['birthdate', 'birthdate'],
                                         seed=42,
                                         inplace=False)
    expected = pd.DataFrame(
        {'birthdate': {0: pd.Timestamp('1914-07-22 00:00:00'),
                       1: pd.Timestamp('1970-10-25 00:00:00')}})
    pdt.assert_frame_equal(expected, output)


@pytest.mark.skipif(__version__ == '0.2.4',
                    reason="Requires anonympy >= 0.2.5")
def test_datetime_fake(anonym_small):
    output = anonym_small.datetime_fake('birthdate', seed=42, inplace=False)
    expected = pd.Series([pd.Timestamp('2013-07-07 00:00:00'),
                         pd.Timestamp('1977-07-30 00:00:00')])
    pdt.assert_series_equal(expected, output, check_names=False)

    output = anonym_small.datetime_fake(['birthdate', 'birthdate'],
                                        seed=42,
                                        inplace=False)
    expected = pd.DataFrame(
        {'birthdate': {0: pd.Timestamp('1971-09-14 00:00:00'),
                       1: pd.Timestamp('2020-06-18 00:00:00')}})
    pdt.assert_frame_equal(expected, output)


def test_column_suppression(anonym_small):
    output = anonym_small.column_suppression('name', inplace=False)
    expected = pd.DataFrame(
                    {'age': {0: 33, 1: 48},
                     'birthdate': {0: pd.Timestamp('1915-04-17 00:00:00'),
                                   1: pd.Timestamp('1970-05-29 00:00:00')},
                     'salary': {0: 59234.32, 1: 49324.53},
                     'web': {0: 'http://www.alandrosenburgcpapc.co.uk',
                             1: 'http://www.capgeminiamerica.co.uk'},
                     'email': {0: 'josefrazier@owen.com',
                               1: 'eryan@lewis.com'},
                     'ssn': {0: '343554334', 1: '656564664'}})
    pdt.assert_frame_equal(expected, output)

    output = anonym_small.column_suppression(['name', 'ssn', 'birthdate'],
                                             inplace=False)

    expected = pd.DataFrame({'age': {0: 33, 1: 48},
                             'salary': {0: 59234.32, 1: 49324.53},
                             'web': {0: 'http://www.alandrosenburgcpapc.co.uk',
                                     1: 'http://www.capgeminiamerica.co.uk'},
                             'email': {0: 'josefrazier@owen.com',
                                       1: 'eryan@lewis.com'}})
    pdt.assert_frame_equal(expected, output)


@pytest.mark.skipif(__version__ == '0.2.4',
                    reason="Requires anonympy >= 0.2.5")
def test_anonymize(anonym_small):
    output = anonym_small.anonymize(inplace=False, seed=42)
    expected = pd.DataFrame(
                {'name': {0: 'Allison Hill', 1: 'Noah Rhodes'},
                 'email': {0: 'johnsonjoshua@example.org',
                           1: 'jillrhodes@example.net'},
                 'ssn': {0: '655-15-0410', 1: '760-36-4013'},
                 'age': {0: 30, 1: 50},
                 'birthdate': {0: pd.Timestamp('1914-07-22 00:00:00'),
                               1: pd.Timestamp('1970-10-25 00:00:00')},
                 'salary': {0: 60000.0, 1: 50000.0},
                 'web': {0: '0e6e42d7d0', 1: '2aa40c1d15'}})
    pdt.assert_frame_equal(expected, output)

    output = anonym_small.anonymize(
                 {'name': 'categorical_fake',
                  'age': 'numeric_noise',
                  'birthdate': 'datetime_noise',
                  'salary': 'numeric_rounding',
                  'web': 'categorical_tokenization',
                  'email': 'categorical_email_masking',
                  'ssn': 'column_suppression'}, inplace=False, seed=42)
    expected = pd.DataFrame(
               {'name': {0: 'Allison Hill', 1: 'Noah Rhodes'},
                'age': {0: 38, 1: 47},
                'birthdate': {0: pd.Timestamp('1914-07-22 00:00:00'),
                              1: pd.Timestamp('1970-10-25 00:00:00')},
                'salary': {0: 60000.0, 1: 50000.0},
                'web': {0: '0e6e42d7d0', 1: '2aa40c1d15'},
                'email': {0: 'j*****r@owen.com', 1: 'e*****n@lewis.com'}})
    pdt.assert_frame_equal(expected, output)
