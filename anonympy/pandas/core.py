import pandas as pd
import numpy as np

from typing import List
from typing import Dict
from typing import Union
from typing import Optional
from typing import Tuple

from texttable import Texttable

import cape_privacy as cp
from cape_privacy.pandas import dtypes 
from cape_privacy.pandas.transformations import NumericPerturbation
from cape_privacy.pandas.transformations import DatePerturbation

from faker import Faker
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
                 numeric_columns: Optional[List[str]] = None,
                 categorical_columns: Optional[List[str]] = None,
                 datetime_columns: Optional[List[str]] = None):
        
        if df.__class__.__name__ != "DataFrame":
            raise Exception(f"{df} is not a pandas DataFrame.")

        # Private Attributes
        self._df = df.copy()
        self._methods_applied = {}
        self._synthetic_data = 'Synthetic Data (fake)'
        self._numeric_perturbation = 'Numeric Perturbation (noise)'
        self._datetime_perturbation = 'Datetime Perturbation (noise)'
        self._round = 'Generalization - Rounding'
        self._bin = 'Generalization - Binning '
        self._sample = 'Resampling'
        self._PCA = 'Masking - PCA'
        self._drop = 'Suppression (drop)'
        
        if numeric_columns == None:
            self._numeric_columns = self._df.select_dtypes(exclude=['object', 'datetime', 'category']).columns.tolist()
        if categorical_columns == None:
            self._categorical_columns = self._df.select_dtypes(include=['object', 'category']).columns.tolist()
        if datetime_columns == None:
            self._datetime_columns = self._df.select_dtypes(include=['datetime']).columns.tolist()
            
        # Public Attributes
        self.columns = self._df.columns.tolist()
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


    def _dtype_checker(self,
                    column: str):
        '''
        Returns the dtype of the column

        Parameters
        ----------
            column: str

        Returns
        ----------
            dtype: numpy dtype 
        '''
        dtype = self._df[column].dtype

        if dtype == np.float32:
            return dtypes.Float
        elif dtype == np.float64:
            return dtypes.Double
        elif dtype == np.byte:
            return dtypes.Byte
        elif dtype == np.short:
            return dtypes.Short
        elif dtype == np.int32:
            return dtypes.Integer
        elif dtype == np.int64:
            return dtypes.Long
        else:
            return None
        


    def anonymize(self,
                  methods: Optional[Dict[str, str]] = None,
                  inplace: Optional[bool] = True):
        '''

        '''
        
        if methods == None:
            pass



    def fake_data(self,
                  column: str,
                  method: str,
                  locale: Optional[Union[str, List[str]]] = ['en_US'],
                  inplace: Optional[bool] = True):
        '''
        Anonymize pandas Series object using synthetic data generator.

        Parameters
        ----------
            column : str
            method : str
            locale : str or List[str], default ['en_US']
            inplace : bool, defaulty True

        Returns
        ----------
            faked : None if inplace = True, else pandas Series

        '''
        
        fake = Faker(locale=locale)
        method = getattr(fake, method)
        faked = self._df[column].apply(lambda x: method())
        if inplace:
            self._df[column] = faked
            self.anonymized_columns.append(column)
            self.unanonymized_columns.remove(column)
            self._methods_applied[column] = self._synthetic_data
        else:
            return faked


    def _fake_data_auto(self,
                       locale: Optional[Union[str, List[str]]] = ['en_US'],
                        inplace: Optional[bool]= True):
        '''
        Compare a column's name to faker's method from utils.fake_method.
        Anonymize if column name is similar to the method. 

        Parameters
        ----------
            locale : str or List[str], default ['en_US']
            inplace : bool, default True

        Returns
        ----------
            None if inplace = True, else an anonymized pandas Series or
            pandas DataFrame depending on the number of columns.
        '''
        
        temp = pd.DataFrame()
        
        for column in self.unanonymized_columns:
            func = column.strip().lower()
            if func in fake_methods:
                if inplace:
                    self.fake_data(column, func, inplace = True)
                else:
                    temp[column] = self.fake_data(column, func, inplace = False)
        if not inplace:
            if len(temp.columns) > 1:
                return temp
            else:
                return pd.Series(temp[temp.columns[0]])


    def numeric_noise(self,
                      column: Union[str, List[str]],
                      MIN: Union[int, float] = -10,
                      MAX: Union[int, float] = 10,
                      seed: Optional[int] = None,
                      inplace: Optional[bool] = True):
        '''
        Add uniform random noise to a numeric Pandas series.
        Based on cape-privacy's pandas.NumericPerturbation 

        Parameters
        ----------
            column : Union[str, List[str]]
            min : (int, float), default -10
            max : (int, float), default 10 
            seed : int, default None
            inplace : bool, default True

        Returns
        ----------
            ser: pandas Series with uniform random noise added
        '''
        
        dtype = self._dtype_checker(column)
        noise = NumericPerturbation(dtype = dtype, min = MIN, max = MAX)
        ser = noise(self._df[column])
        
        if inplace:
            self._df[column] = ser
            self.anonymized_columns.append(column)
            self.unanonymized_columns.remove(column)
            self._methods_applied[column] = self._numeric_perturbation
        else:
            return ser


    def datetime_noise(self,
                       column: Union[str, List[str]],
                       frequency: Union[str, Tuple[str, ...]]  = ("YEAR", "MONTH", "DAY"),
                       MIN:  Union[int, Tuple[int, ...]] = (-10, -5, -5),
                       MAX:  Union[int, Tuple[int, ...]] = (10, 5, 5),
                       seed: Optional[int] = None,
                       inplace: Optional[bool] = True):
        '''
        Add uniform random noise to a Pandas series of timestamps
        Based on cape-privacy's pandas.DatePerturbation 

        Parameters
        ----------
            column : Union[str, List[str]]
            frequency : Union[str, Tuple[str, ...]] , default  ("YEAR", "MONTH", "DAY")
            min : Union[int, Tuple[int, ...]], default (-10, -5, -5)
            max : Union[int, Tuple[int, ...]], default (10, 5, 5)
            seed : int, default None
            inplace : bool, default True

        Returns
        ----------
            ser: pandas Series with uniform random noise added
        '''

        noise = DatePerturbation(frequency = frequency, min = MIN, max = MAX)
        ser = noise(self._df[column])

        if inplace:
            self._df[column] = ser
            self.anonymized_columns.append(column)
            self.unanonymized_columns.remove(column)
            self._methods_applied[column] = self._datetime_perturbation
        else:
            return ser
        

    def info(self):
        '''
        Print a summary of the a DataFrame, which columns have been anonymized and which haven't.

        Returns
        ----------
            None
        '''
        t = Texttable(max_width=150)
        header = f'Total number of columns: {self._df.shape[1]}'

        row1 = 'Anonymized Column -> Method: '
        for column in self.anonymized_columns:
            row1 += '\n- ' + column + ' -> ' + self._methods_applied.get(column)

        row2 = 'Unanonymized Columns: \n'
        row2 +='\n'.join([f'- {i}' for i in self.unanonymized_columns])

        t.add_rows([[header], [row1], [row2]])

        print(t.draw())

        
    def to_df(self):
        ''' 
        Convert dfAnonymizer object back to pandas DataFrame

        Returns
        ----------
        DataFrame object
        '''
        
        return self._df.copy()
    
        
        
