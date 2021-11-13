import cape_privacy as cp
from faker import Faker
import pandas as pd
from typing import List


df = pd.DataFrame({
        "name": ["alice", "bob"],
        "age": [34, 55],
        "birthdate": [pd.Timestamp(1985, 2, 23), pd.Timestamp(1963, 5, 10)],
        "salary": [59234.32, 49324.53],
        "ssn": ["343554334", "656564664"],
    })


class dfAnonymizer(object):
    """
    Initializes a pd.DataFrame as a dfAnonymize object.

    Args:
        df: 
        numeric_columns: 
        categorical_columns: 
        datetime_columns: 
        inplace: 
        
    Returns:
        dfAnonymizer object
    """


    def __init__(self,
                 df: pd.DataFrame,
                 numeric_columns: List[str] = None,
                 categorical_columns: List[str] = None,
                 datetime_columns: List[str] = None,
                 inplace: bool = False):    
        if df.__class__.__name__ != "DataFrame":
            raise Exception(f"{df} is not a pandas DataFrame.")

        self.df = df
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.datetime_columns = datetime_columns
        self.inplace = inplace


















        
        
