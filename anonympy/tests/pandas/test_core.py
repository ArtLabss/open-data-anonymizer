import pytest
import pandas as pd
from anonympy.pandas import dfAnonymizer
from anonympy.pandas.utils import load_dataset


@pytest.fixture(scope = 'module')
def load_df():
    '''
    Returns a DataFrame from anonympy.pandas.utils.load_dataset('small')
    '''
    df = load_dataset('small')
    return df 


@pytest.fixture(scope = 'module')
def anonym_obj(load_df):
    '''
    Returns an dfAnonymizer instance 
    '''
    anonym = dfAnonymizer(load_df)
    return anonym


def test_load_df(load_df):
    assert isinstance(load_df, pd.DataFrame), "`load_dataset` should return a pandas DataFrame"


def test_anonym_obj(anonym_obj):
    assert isinstance(anonym_obj, dfAnonymizer), "should have returned `dfAnonymizer` object"


#pytest -v test_core.py
