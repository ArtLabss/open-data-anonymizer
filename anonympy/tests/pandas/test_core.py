import urllib
import pytest
import pandas as pd

from anonympy.pandas import dfAnonymizer
from anonympy.pandas.utils import load_dataset


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
        return anonym
    except urllib.error.HTTPError as error:
        anonym = None
        return anonym 


def test_anonym_obj(anonym_small, anonym_big):
    assert isinstance(anonym_small, dfAnonymizer), "should have returned `dfAnonymizer` object"
    if anonym_big is None:
        assert False, "Failed to fetch the DataFrame"
    assert isinstance(anonym_big, dfAnonymizer), "should have returned `dfAnonymizer` object"
    

def test_numeric_noise(anonym_small):
    output = anonym_small.numeric_noise('age', seed = 42, inplace = False)
    expected = pd.Series([38, 47], dtype='int64')
    if isinstance(output, pd.Series):
        assert output.equals(expected), "`dfAnonymizer.numeric_noise` returned unexpected values"
    else:
        assert False, "Applying `dfAnonymizer.numeric_noise` on 1 column should have returned a pd.Series."

    output = anonym_small.numeric_noise(['age', 'salary'], seed = 42, inplace = False)
    expected = pd.DataFrame({'age': [38, 47], 'salary': [59239.79912097112, 49323.30756879504]})
    if isinstance(output, pd.DataFrame):
        assert output.equals(expected), "`dfAnonymizer.numeric_noise` returned unexpected values"
    else:
        assert False, "Applying `dfAnonymizer.numeric_noise` on more than 1 column should have returned a pd.DataFrame."

     
def test_numeric_binning(anonym_small):
    output = anonym_small.numeric_binning('salary', bins=2, inplace = False)
    dtype = pd.CategoricalDtype([pd.Interval(49315.0, 54279.0, closed='right'),pd.Interval(54279.0, 59234.0, closed='right')], ordered=True)
    expected = pd.Series([pd.Interval(54279.0, 59234.0, closed='right'), pd.Interval(49315.0, 54279.0, closed='right')], dtype=dtype)
    
    if isinstance(output, pd.Series):
        assert output.equals(expected), "`dfAnonymizer.numeric_binning` returned unexpected values"
    else:
        assert False, "Applying `dfAnonymizer.numeric_binning` on 1 column should have returned a pd.Series."

    output = anonym_small.numeric_binning(['age', 'salary'], bins=2, inplace = False)
    dtype2 = pd.CategoricalDtype([pd.Interval(33.0, 40.0, closed='right'), pd.Interval(40.0, 48.0, closed='right')], ordered=True)
    ser2 = pd.Series([pd.Interval(33.0, 40.0, closed='right'), pd.Interval(40.0, 48.0, closed='right')], dtype=dtype2)
    expected = pd.DataFrame({'age': ser2,'salary': expected})

    if isinstance(output, pd.DataFrame):
        assert output.equals(expected), "`dfAnonymizer.numeric_binning` returned unexpected values"
    else:
        assert False, "Applying `dfAnonymizer.numeric_binning` on more than 1 column should have returned a pd.DataFrame."


def test_numeric_masking(anonym_small):
    output = anonym_small.numeric_masking(['age', 'salary'], inplace = False)
    expected = pd.DataFrame({'age': [-4954.900676201789, 4954.900676201798],
                             'salary': [5.840670901327418e-15, 5.840670901327409e-15]})
    assert output.equals(expected), "`dfAnonymizer.numeric_masking` returned unexpected values"


def test_numeric_rounding(anonym_small):
    anonym_small.numeric_masking(['age', 'salary'], inplace = False)
    
    
