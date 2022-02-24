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
    assert output.equals(expected), "`dfAnonymizer.numeric_noise` returned unexpected values"

 
def test_numeric_binning(anonym_small):
    output = anonym_small.numeric_binning('salary', bins=2, inplace = False)
    expected = pd.cut(anonym_small.to_df()['salary'], bins=2, precision=0)
    assert output.equals(expected)
    
