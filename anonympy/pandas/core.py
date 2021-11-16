import cape_privacy as cp
from faker import Faker
import pandas as pd
from typing import List, Dict, Union
from texttable import Texttable
from utils import fake_methods

df = pd.DataFrame({
        "name": ["alice", "bob"],
        "age": [34, 55],
        "birthdate": [pd.Timestamp(1985, 2, 23), pd.Timestamp(1963, 5, 10)],
        "salary": [59234.32, 49324.53],
        "ssn": ["343554334", "656564664"],
    })


class dfAnonymizer(object):
    """
    Initializes pandas DataFrame as a dfAnonymizer object.

    Parameters:
    ----------
        df: pandas DataFrame
        numeric_columns : list, default None
        categorical_columns : list, default None
        datetime_columns : list, default None
         
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
                 ):    
        if df.__class__.__name__ != "DataFrame":
            raise Exception(f"{df} is not a pandas DataFrame.")

        self._df = df.copy()
        self._methods_applied = {}
        if numeric_columns == None:
            self._numeric_columns = self._df.select_dtypes(exclude=['object', 'datetime', 'category']).columns.tolist()
        if categorical_columns == None:
            self._categorical_columns = self._df.select_dtypes(include=['object', 'category']).columns.tolist()
        if datetime_columns == None:
            self._datetime_columns = self._df.select_dtypes(include=['datetime']).columns.tolist()
        self.anonymized_columns = []
        self.unanonymized_columns = self._df.columns.to_list()


    def get_numeric_columns(self) -> List:
        '''
        Return a subset of the DataFrame's columns which are numeric.

        Returns
        ----------
            List of columns with numeric values 
        '''
        return self._numeric_columns


    def get_categorical_columns(self) -> List:
        '''
        Return a subset of the DataFrame's columns which are categorical.

        Returns
        ----------
            List of columns with categorical values 
        '''
        return self._categorical_columns


    def get_datetime_columns(self) -> List:
        '''
        Return a subset of the DataFrame's columns which are datetime. 

        Returns
        ----------
            List of columns with datetime values 
'''
        return self._datetime_columns


    def anonymize(self,
                  methods: Dict[str, str] = None):

        if methods == None:
            pass
        

    def fake_data_manual(self, column: str, method: str, locale: Union[str, List[str]] = ['en_US'], inplace = False) -> pd.Series:
        '''
        Anonymize pandas Series object using synthetic data generator.

        Parameters
        ----------
            column : str
            method : str
            locale : List[str], default ['en_US']
            inplace : bool

        Returns
        ----------
            faked : pandas Series

        '''
        
        fake = Faker(locale=locale)
        method = getattr(fake, method)
        faked = self._df[column].apply(lambda x: method())
        self._methods_applied[column] = 'Synthetic Data'
        if inplace:
            self._df[column] = faked
            self.anonymized_columns.append(column)
            self.unanonymized_columns.remove(column)
        else:
            return faked

    def fake_data_auto(self):
        pass
    

    def to_DataFrame(self):
        ''' 
        Convert dfAnonymizer object back to pandas DataFrame

        Returns
        ----------
        DataFrame object
        '''
        
        return self._df


    def info(self):
        '''
        Print a summary of the a DataFrame, which columns have been anonymized and which haven't.

        Returns
        ----------
            None
        '''
        t = Texttable()
        header = f'Total number of columns: {self._df.shape[1]}'

        row1 = 'Anonymized Column -> Method: \n'
        for column in self.anonymized_columns:
            row1 += '\n- ' + column + ' -> ' + self._methods_applied.get(column)

        row2 = 'Unanonymized Columns: \n\n'
        row2 +='\n'.join([f'- {i}' for i in self.unanonymized_columns])

        t.add_rows([[header], [row1], [row2]])

        print(t.draw())
        
        
        
