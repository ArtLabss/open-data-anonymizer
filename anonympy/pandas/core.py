import cape_privacy as cp
from faker import Faker
import pandas as pd
from typing import List, Dict


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
    Set ``seed`` to an integer for reproducible results 

    Parameters:
    ----------
        df: pd.DataFrame
        numeric_columns : list, default None
        categorical_columns : list, default None
        datetime_columns : list, default None
        inplace : bool, default False
         
    Returns:
    ----------
        dfAnonymizer object

    Raises
    ----------
        Exception:
            * If ``df`` is not a  DataFrame
    """


    def __init__(self,
                 df: pd.DataFrame,
                 numeric_columns: List[str] = None,
                 categorical_columns: List[str] = None,
                 datetime_columns: List[str] = None,
                 inplace: bool = False,
                 ):    
        if df.__class__.__name__ != "DataFrame":
            raise Exception(f"{df} is not a pandas DataFrame.")

        self._df = df
        self._inplace = inplace
        if numeric_columns == None:
            self._numeric_columns = df.select_dtypes(exclude=['object', 'datetime', 'category']).columns.tolist()
        if categorical_columns == None:
            self._categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if datetime_columns == None:
            self._datetime_columns = df.select_dtypes(include=['datetime']).columns.tolist()


    def get_numeric_columns(self) -> List:
        '''
Return a subset of the DataFrame's columns which are numeric

Returns
----------
    List of columns with numeric values 
'''
        return self._df[self._numeric_columns]


    def get_categorical_columns(self) -> List:
        '''
Return a subset of the DataFrame's columns which are categorical

Returns
----------
List of columns with categorical values 
'''
        return self._categorical_columns


    def get_datetime_columns(self) -> List:
        '''
Return a subset of the DataFrame's columns which are datetime 

Returns
----------
List of columns with datetime values 
'''
        return self._datetime_columns


    def anonymize(self,
                  methods: Dict[str, str] = None):

        if methods == None:
            pass
        

    def fake_data_manual(self, ser:pd.Series, method: str, locale: List[str] = ['en_US']) -> pd.Series:
        fake = Faker(locale=locale)
        method = getattr(fake, method)
        faked = ser.apply(lambda x: method())
        return faked










        
        
