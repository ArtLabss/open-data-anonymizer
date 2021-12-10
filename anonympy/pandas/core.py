import pandas as pd
import numpy as np
import datetime

from typing import List
from typing import Dict
from typing import Union
from typing import Optional
from typing import Tuple

from texttable import Texttable

from cape_privacy.pandas import dtypes 
from cape_privacy.pandas.transformations import NumericPerturbation
from cape_privacy.pandas.transformations import DatePerturbation
from cape_privacy.pandas.transformations import NumericRounding
from cape_privacy.pandas.transformations import Tokenizer

import utils as _utils 
from utils import timer_func, load_dataset
from faker import Faker

from sklearn.decomposition import PCA


class dfAnonymizer(object):
    """
    Initializes pandas DataFrame as a dfAnonymizer object.

    Parameters:
    ----------
        df: pandas DataFrame
         
    Returns:
    ----------
        dfAnonymizer object

    Raises
    ----------
        Exception:
            * If ``df`` is not a DataFrame

    Examples
    ----------
    Contructing dfAnonymizer object from a DataFrame.

    >>> df = load_dataset()
    >>> anonym = dfAnonymizer(df)
    >>> anonym.to_df()
         name   age  ...                 email          ssn
    0   Bruce   33   ...  josefrazier@owen.com    343554334
    1   Tony    48   ...       eryan@lewis.com    656564664
    """
    def __init__(self,
                 df: pd.DataFrame):
        
        if df.__class__.__name__ != "DataFrame":
            raise Exception(f"{df} is not a pandas DataFrame.")
        
        # Private Attributes
        self._df = df.copy()
        self._methods_applied = {}
        self._synthetic_data = 'Synthetic Data'
        self._tokenization = 'Tokenization'
        self._numeric_perturbation = 'Numeric Perturbation'
        self._datetime_perturbation = 'Datetime Perturbation'
        self._round = 'Generalization - Rounding'
        self._bin = 'Generalization - Binning'
        self._drop = 'Column Suppression'
        self._sample = 'Resampling'
        self._PCA = 'PCA Masking'
        self._email = 'Partial Masking'
            
        # Public Attributes
        self.anonymized_columns = []
        self.columns = self._df.columns.tolist()
        self.unanonymized_columns = self.columns.copy()
        
        self.numeric_columns = _utils.get_numeric_columns(self._df)
        self.categorical_columns = _utils.get_categorical_columns(self._df)
        self.datetime_columns = _utils.get_datetime_columns(self._df)

        self._available_methods = _utils.available_methods 
        self._fake_methods = _utils.fake_methods


    def __str__(self):
        return self._info().draw()


    def __repr__(self):
        return self._info().draw()


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
                  methods = None,
                  locale = ['en_US'],
                  inplace = True):
        '''
        Anonymize all columns using different method for each dtype.

        If dictionary is not provided, for numerical columns ``numeric_rounding`` is applied. ``categorical_fake`` and ``categorical_tokenization``
        for categorical columns and ``datetime_noise`` or ``datetime_fake`` are applied for datetime columns.

        Parameters
        ----------
        methods : Optional[Dict[str, str]], default None
            {column_name: anonympy_method}. List of all methods ``dfAnonymizer.available_methods``.
        locale : str or List[str], default ['en_US']
            See https://faker.readthedocs.io/en/master/locales.html for all faker's locales.
        inplace : bool, default True
            If True, the changes will be applied to DataFrame (access using ``dfAnonymizer.to_df``)
            Else, output is returned. 

        Returns
        ----------
            None if inplace = True, else pandas Series or pandas DataFrame.

        Examples
        ----------
        >>> df = load_dataset()
        >>> anonym = dfAnonymizer(df)
        
        If methods None.

        >>> anonym.anonymize(inplace = False)
                       name      age  ...                  email        ssn
        0  Douglas Williams      30   ...  dcampbell@example.org     718-51-5290
        1     Nicholas Hall      50   ...  orichards@example.com     684-81-8137

        Dictionary for specifying which methods to apply.
        
        >>> anonym.available_methods('numeric')
            numeric_noise   numeric_binning	numeric_masking	  numeric_rounding
        >>> anonym.anonymize({'name':'categorical_fake', 'age':'numeric_noise', 'email':'categorical_email_masking'}, inplace = False)
                    name  age               email
        0  Kyle Thompson   38    j*****r@owen.com
        1  Michael Smith   47   e*****n@lewis.com        
        '''
        if not methods:
            if inplace:
                # try synthetic data 
                self.categorical_fake_auto(locale = locale)   # anonymize using fake data if any column's name is similar to Faker's method (print(fake_methods) for all available methods)
                # if there are still columns left unanonymized 
                if self.unanonymized_columns:
                    for column in self.unanonymized_columns.copy():
                        
                        if column in self.numeric_columns:
                            self.numeric_rounding(column)
                            
                        elif column in self.categorical_columns:
                            self.categorical_tokenization(column)
                        
                        elif column in self.datetime_columns:
                            self.datetime_noise(column)
            else:
                # try synthetic data
                temp = self.categorical_fake_auto(locale = locale, inplace = False)
                unanonymized = self.unanonymized_columns.copy()
                
                if isinstance(temp, pd.DataFrame):
                    unanonymized = [column for column in unanonymized if column not in temp.columns.to_list()]
                elif isinstance(temp, pd.Series):
                    unanonymized.remove(temp.name)
                    temp = pd.DataFrame(temp)
                else: # if temp is a already  a dataframe
                    temp = pd.DataFrame()
                    
                if unanonymized:
                    for column in unanonymized:
                        if column in self.numeric_columns:
                            temp[column] = self.numeric_rounding(column, inplace = False)
                            
                        elif column in self.categorical_columns:
                            temp[column] = self.categorical_tokenization(column, inplace = False)
                        
                        elif column in self.datetime_columns:
                            temp[column] = self.datetime_noise(column, inplace = False)
                return temp 
        # if dictionary with methods was passed
        else:
            if inplace: 
                for key, value in methods.items():
                    # numeric
                    if value == "numeric_noise":
                        self.numeric_noise(key)
                    elif value == "numeric_binning":
                        self.numeric_binning(key)
                    elif value == "numeric_masking":
                        self.numeric_masking(key)
                    elif value == "numeric_rounding":
                        self.numeric_rounding(key)
                    # categorical
                    elif value == "categorical_fake":
                        self.categorical_fake(key)
                    elif value == "categorical_resampling":
                        self.categorical_resampling(key)
                    elif value == "categorical_tokenization":
                        self.categorical_tokenization(key)
                    elif value == "categorical_email_masking":
                        self.categorical_email_masking(key)
                    # datetime
                    elif value == "datetime_fake":
                        self.datetime_fake(key)
                    elif value == "datetime_noise":
                        self.datetime_noise(key)
                    # drop 
                    elif value == "column_suppression":
                        self.column_suppression(key)           
            else:
                temp = pd.DataFrame()
                for key, value in methods.items():
                    # numeric
                    if value == "numeric_noise":
                        temp[key] = self.numeric_noise(key, inplace = False)
                    elif value == "numeric_binning":
                        temp[key] = self.numeric_binning(key, inplace = False)
                    elif value == "numeric_masking":
                        temp[key] = self.numeric_masking(key, inplace = False)
                    elif value == "numeric_rounding":
                        temp[key] = self.numeric_rounding(key, inplace = False)
                    # categorical
                    elif value == "categorical_fake":
                        temp[key] = self.categorical_fake(key, inplace = False)
                    elif value == "categorical_resampling":
                        temp[key] = self.categorical_resampling(key, inplace = False)
                    elif value == "categorical_tokenization":
                        temp[key] = self.categorical_tokenization(key, inplace = False)
                    elif value == 'categorical_email_masking':
                        temp[key] = self.categorical_email_masking(key, inplace = False)
                    # datetime
                    elif value == "datetime_fake":
                        temp[key] = self.datetime_fake(key, inplace = False)
                    elif value == "datetime_noise":
                        temp[key] = self.datetime_noise(key, inplace = False)
                    # drop 
                    elif value == "column_suppression":
                        temp[key] = self.column_suppression(key, inplace = False)

                if len(temp.columns) > 1:
                    return temp
                elif len(temp.columns) == 1:
                    return pd.Series(temp[temp.columns[0]])
                    
    
    def _fake_column(self,
                  column,
                  method,
                  locale = ['en_US'],
                  inplace = True):
        '''
        Anonymize pandas Series object using synthetic data generator
        Based on faker.Faker.
        
        Parameters
        ----------
            column : str
            method : str
            locale : str or List[str], default ['en_US']
            inplace : bool, default True

        Returns
        ----------
            None if inplace = True, else pandas Series.
        '''
        fake = Faker(locale=locale)
        method = getattr(fake, method)
        faked = self._df[column].apply(lambda x: method())
        if inplace:
            if column in self.anonymized_columns:
                print(f'`{column}` column already anonymized!')
            else:
                self._df[column] = faked
                self.unanonymized_columns.remove(column)
                self.anonymized_columns.append(column)
                self._methods_applied[column] = self._synthetic_data
        else:
            return faked

    
    def categorical_fake(self,
                  columns,
                  locale = ['en_US'],
                  inplace = True):
        '''
        Anonymize pandas Series or pandas DataFrame using synthetic data generator
        To see the list of all faker's methods, call ``faker_methods()``. 

        Parameters
        ----------
            columns : Union[str, List[str], Dict[str, str]]
                * if column string or list of column strings is passed, faker's method which is the same as the column name will be applied.
                * dictionary with column as a key and method name as value should be used if column name doesn't correspond to any faker's methods.
            locale : str or List[str], default ['en_US']
            inplace : bool, default True

        Returns
        ----------
            faked : None if inplace = True, else pandas Series or pandas DataFrame

        Examples
        ----------
        >>> df = load_dataset()
        >>> anonym = dfAnonymizer(df)

        If methods are not specified
        
        >>> anonym.categorical_fake(['name', 'email', 'ssn'], inplace = False)
                       name                     email          ssn
        0  Veronica Nichols  michealscott@example.com  741-09-3939
        1       Brent Smith    amberyoder@example.net  443-53-3660

        Passing a specific faker's method
        
        >>> anonym.fake_methods('n')
            name, name_female, name_male, name_nonbinary, nic_handle, nic_handles, null_boolean, numerify
        >>> anonym.categorical_fake({'name':'name_nonbinary'}, inplace = False)
                      name
        0         Ann Snow
        1  Lindsay Roberts        
        '''
        # if a single column is passed (str)
        if isinstance(columns, str) or (len(columns) == 1 and isinstance(columns, list)):
            if isinstance(columns, list):
                columns = columns[0]
            if inplace:
                self._fake_column(columns, columns, inplace = True, locale = locale)
            else:
                return self._fake_column(columns, columns, inplace = False, locale = locale)
        
        # if a list of columns is passed
        elif isinstance(columns, list):
            temp = pd.DataFrame()
            if inplace:
                for column in columns:
                    self._fake_column(column, column, inplace = True, locale = locale)
            else:
                for column in columns:
                    faked = self._fake_column(column, column, inplace = False, locale = locale)
                    temp[column] = faked
                return temp
            
        # if a dictionary with column name and method name is passed 
        elif isinstance(columns, dict):
            temp = pd.DataFrame()
            if inplace:
                for column, method in columns.items():
                    self._fake_column(column, method, inplace = True, locale = locale)
            else:
                for column, method in columns.items():
                    faked = self._fake_column(column, method, inplace = False, locale = locale)
                    temp[column] = faked
                return temp

            
    def categorical_fake_auto(self,
                        locale = ['en_US'],
                        inplace = True):
        '''
        Anonymize if column name is similar to the method.

        In order to produce synthetic data, column name should have same name as faker's method name.

        Parameters
        ----------
            locale : str or List[str], default ['en_US']
            inplace : bool, default True

        Returns
        ----------
            None if inplace = True, else an anonymized pandas Series or pandas DataFrame
        '''
        temp = pd.DataFrame()
        
        for column in self.columns:
            func = column.strip().lower()
            if func in _utils._fake_methods:
                if inplace:
                    if column in self.anonymized_columns:
                        print(f'`{column}` column already anonymized!')
                    else:
                        self._fake_column(column, func, inplace = True, locale = locale)
                else:
                    temp[column] = self._fake_column(column, func, inplace = False, locale = locale)
        if not inplace:
            if len(temp.columns) > 1:
                return temp
            elif len(temp.columns) == 1:
                return pd.Series(temp[temp.columns[0]])
            else:
                return None


    def numeric_noise(self,
                      columns,
                      MIN = -10,
                      MAX = 10,
                      seed = None,
                      inplace = True):
        '''
        Add uniform random noise to a numeric Pandas series.
        Based on cape-privacy's NumericPerturbation function

        Parameters
        ----------
            columns : Union[str, List[str]]
            MIN : (int, float), default -10
            MAX : (int, float), default 10 
            seed : int, default None
            inplace : bool, default True

        Returns
        ----------
            ser: pandas Series or pandas DataFrame with uniform random noise added
        '''
        # If a single column is passed
        if isinstance(columns, str) or (len(columns) == 1 and isinstance(columns, list)):
            if isinstance(columns, list):
                columns = columns[0]
            dtype = self._dtype_checker(columns)
            noise = NumericPerturbation(dtype = dtype, min = MIN, max = MAX)
            ser = noise(self._df[columns].copy()).astype(dtype)

            if inplace:
                if columns in self.anonymized_columns:
                    print(f'`{columns}` column already anonymized!')
                else:
                    self._df[columns] = ser
                    self.anonymized_columns.append(columns)
                    self.unanonymized_columns.remove(columns)
                    self._methods_applied[columns] = self._numeric_perturbation
            else:
                return ser.astype(dtype)
        # if a list of columns is passed
        else:
            temp = pd.DataFrame()
            for column in columns:
                
                dtype = self._dtype_checker(column)
                noise = NumericPerturbation(dtype = dtype, min = MIN, max = MAX)
                ser = noise(self._df[column].copy()).astype(dtype)

                if inplace:
                    if column in self.anonymized_columns:
                         print(f'`{column}` column already anonymized!')
                    else:
                        self._df[column] = ser
                        self.anonymized_columns.append(column)
                        self.unanonymized_columns.remove(column)
                        self._methods_applied[column] = self._numeric_perturbation
                else:
                    temp[column] = ser
            if not inplace:
                return temp


    def datetime_noise(self,
                       columns,
                       frequency = ("MONTH", "DAY"),
                       MIN = (-10, -5, -5),
                       MAX = (10, 5, 5),
                       seed = None,
                       inplace = True):
        '''
        Add uniform random noise to a Pandas series of timestamps
        Based on cape-privacy's DatePerturbation function

        Parameters
        ----------
            columns : Union[str, List[str]]
            frequency : Union[str, Tuple[str, ...]] , default ("MONTH", "DAY")
            MIN : Union[int, Tuple[int, ...]], default (-10, -5, -5)
            MAX : Union[int, Tuple[int, ...]], default (10, 5, 5)
            seed : int, default None
            inplace : bool, default True

        Returns
        ----------
            ser: pandas Series or pandas DataFrame
        '''
        # if a single column is passed
        if isinstance(columns, str) or (len(columns) == 1 and isinstance(columns, list)):
            if isinstance(columns, list):
                columns = columns[0]
            noise = DatePerturbation(frequency = frequency, min = MIN, max = MAX)
            ser = noise(self._df[columns].copy())

            if inplace:
                if columns in self.anonymized_columns:
                    print(f'`{columns}` column already anonymized!')
                else:
                    self._df[columns] = ser
                    self.anonymized_columns.append(columns)
                    self.unanonymized_columns.remove(columns)
                    self._methods_applied[columns] = self._datetime_perturbation
            else:
                return ser
        # if a list of columns is passed
        else:
            temp = pd.DataFrame()
            
            for column in columns:
                noise = DatePerturbation(frequency = frequency, min = MIN, max = MAX)
                ser = noise(self._df[column].copy())

                if inplace:
                        if column in self.anonymized_columns:
                            print(f'`{column}` column already anonymized!')
                        else:
                            self._df[column] = ser
                            self.anonymized_columns.append(column)
                            self.unanonymized_columns.remove(column)
                            self._methods_applied[column] = self._datetime_perturbation
                else:
                    temp[column] = ser
        if not inplace:
            return temp


    def numeric_rounding(self,
                         columns,
                         precision = None,
                         inplace = True):
        '''
        Round each value in the Pandas Series to the given number
        Based on cape-privacy's NumericRounding

        Parameters
        ----------
            columns : Union[str, List[str]]
            precision : int, default None
            inplace : bool, default True

        Returns
        ----------
            ser: pandas Series or pandas DataFrame
        '''
        # if a single column is passed
        if isinstance(columns, str) or (len(columns) == 1 and isinstance(columns, list)):
            if isinstance(columns, list):
                columns = columns[0]
            dtype = self._dtype_checker(columns)
            precision = len(str(int(self._df[columns].mean()))) - 1
            rounding = NumericRounding(dtype = dtype, precision = -precision)
            ser = rounding(self._df[columns].copy()).astype(dtype)

            if inplace:
                if columns in self.anonymized_columns:
                    print(f'`{columns}` column already anonymized!')
                else:
                    self._df[columns] =  ser
                    self.anonymized_columns.append(columns)
                    self.unanonymized_columns.remove(columns)
                    self._methods_applied[columns] = self._round
            else:
                return ser
        # if a list of columns is passed
        else:
            temp = pd.DataFrame()
            
            for column in columns:
                dtype = self._dtype_checker(column)
                precision = len(str(int(self._df[column].mean()))) - 1
                rounding = NumericRounding(dtype = dtype, precision = -precision)
                ser = rounding(self._df[column].copy())

                if inplace:
                    if column in self.anonymized_columns:
                        print(f'`{column}` column already anonymized!')
                    else:
                        self._df[column] = ser
                        self.anonymized_columns.append(column)
                        self.unanonymized_columns.remove(column)
                        self._methods_applied[column] = self._round
                else:
                    temp[column] = ser.astype(dtype)
            if not inplace:
                return temp

    
    def numeric_masking(self,
                        columns,
                        inplace = True):
        '''
        Apply PCA masking to a column/columns
        Based on sklearn's PCA function

        Parameters
        ----------
            columns : Union[str, List[str]]
            inplace : bool, default True

      Returns
      ----------
            ser : pandas Series or pandas DataFrame
        '''
        # if a single column is passed
        if isinstance(columns, str) or (len(columns) == 1 and isinstance(columns, list)):
            if isinstance(columns, list):
                columns = columns[0]
            pca = PCA(n_components=1)
            ser = pd.Series(pca.fit_transform(self._df[columns]))

            if inplace:
                if columns in self.anonymized_columns:
                    print(f'`{columns}` column already anonymized!')
                else:
                    self._df[columns] = ser
                    self.anonymized_columns.append(columns)
                    self.unanonymized_columns.remove(columns)
                    self._methods_applied[columns] = self._PCA
            else:
                return ser
        # if a list of columns is passed
        else:
            if not inplace:
                 pca = PCA(n_components=len(columns))
                 return pd.DataFrame(pca.fit_transform(self._df[columns]), columns=columns)

            else:
                for column in columns:
                    if column in self.anonymized_columns:
                        print(f'`{column}` column already anonymized!')
                        columns.remove(column)
                    else:
                        self.anonymized_columns.append(column)
                        self.unanonymized_columns.remove(column)
                        self._methods_applied[column] = self._PCA
                        
                pca = PCA(n_components=len(columns))
                self._df[columns] = pca.fit_transform(self._df[columns])
    
    
    def categorical_tokenization(self,
                  columns,
                  max_token_len = 10,
                  key = b"my secret",
                  inplace = True):
        '''
        Maps a string to a token (hexadecimal string) to obfuscate it.

        Parameters
        ----------
            columns : Union[str, List[str]]
            max_token_len : int, default 10
            key : str, default b"my secret"
            inplace : bool, default True

        Returns
        ----------
            ser : pandas Series or pandas DataFrame
        '''
        
        # if a single column is passed
        if isinstance(columns, str) or (len(columns) == 1 and isinstance(columns, list)):
            if isinstance(columns, list):
                columns = columns[0]
            tokenize = Tokenizer(max_token_len = max_token_len, key = b"my secret")
            ser = tokenize(self._df[columns])

            if inplace:
                if columns in self.anonymized_columns:
                    print(f'`{columns}` column already anonymized!')
                else:
                    self._df[columns] = ser
                    self.anonymized_columns.append(columns)
                    self.unanonymized_columns.remove(columns)
                    self._methods_applied[columns] = self._tokenization
            else:
                return ser
        # if a list of columns is passed
        else:
            temp = pd.DataFrame()
            
            for column in columns:
                tokenize = Tokenizer(max_token_len = max_token_len, key = b"my secret")
                ser = tokenize(self._df[column])

                if inplace:
                    if column in self.anonymized_columns:
                        print(f'`{column}` column already anonymized!')
                    else:
                        self._df[column] = ser
                        self.anonymized_columns.append(column)
                        self.unanonymized_columns.remove(column)
                        self._methods_applied[column] = self._tokenization
                else:
                    temp[column] = ser
            if not inplace:
                return temp

            
    def _mask(self, s):
        '''
        Mask a single email

        Parameters
        ----------
            s : str

        Returns
        ----------
            masked : str 
        '''
        lo = s.find('@')

        if lo > 0:
            masked = s[0] + '*****' + s[lo-1:]
            return masked
        else:
            raise Exception('Invalid Email')
    
    
    def categorical_email_masking(self,
                      columns,
                      inplace = True):
        '''
        Apply Partial Masking to emails.

        Parameters
        ----------
            columns: Union[str, List[str]]
            inplace: Optional[bool] = True

        Returns
        ----------
            ser : pandas Series or pandas DataFrame 
        '''
        # if a single column is passed
        if isinstance(columns, str) or (len(columns) == 1 and isinstance(columns, list)):
            if isinstance(columns, list):
                columns = columns[0]
            ser = self._df[columns].apply(lambda x: self._mask(x))

            if inplace:
                if columns in self.anonymized_columns:
                    print(f'`{columns}` column already anonymized!')
                else:
                    self._df[columns] = ser
                    self.anonymized_columns.append(columns)
                    self.unanonymized_columns.remove(columns)
                    self._methods_applied[columns] = self._email
            else:
                return ser
        # if a list of columns is passed
        else:
            temp = pd.DataFrame()

            for column in columns:
                ser = self._df[columns].apply(lambda x: self._mask(x))

                if inplace:
                    if column in self.anonymized_columns:
                        print(f'`{column}` column already anonymized!')
                    else:
                        self._df[column] = ser
                        self.anonymized_columns.append(column)
                        self.unanonymized_columns.remove(column)
                        self._methods_applied[column] = self._email
                else:
                    temp[column] = ser

            if not inplace:
                return temp
                        
        
    def datetime_fake(self,
                  columns,
                  pattern = '%Y-%m-%d',
                  end_datetime = None,
                  locale = ['en_US'],
                  inplace = True):
        '''
        Replace Column's values with synthetic dates between January 1, 1970 and now.
        Based on faker `date()` method

        Parameters
        ----------
            columns : Union[str, List[str]]
            pattern : str, default  '%Y-%m-%d'
            end_datetime : Union[datetime.date, datetime.datetime, datetime.timedelta, str, int, None], default None
            locale : str or List[str], default ['en_US']
            inplace : bool, default True

        Returns
        ----------
            ser : pandas Series or pandas DataFrame
        '''
        fake = Faker(locale=locale)

        # if a single column is passed 
        if isinstance(columns, str) or (len(columns) == 1 and isinstance(columns, list)):
            if isinstance(columns, list):
                columns = columns[0]
            ser = self._df[columns].apply(lambda x: pd.to_datetime(fake.date(pattern=pattern, end_datetime=end_datetime)))
            if inplace:
                if columns in self.anonymized_columns:
                    print(f'`{columns}` column already anonymized!')
                else:
                    self._df[columns] = ser
                    self.anonymized_columns.append(columns)
                    self.unanonymized_columns.remove(columns)
                    self._methods_applied[columns] = self._synthetic_data
            else:
                return ser
        # if a list of columns is passed 
        else:
            temp = pd.DataFrame()
            
            for column in columns:
                ser = self._df[column].apply(lambda x: pd.to_datetime(fake.date(pattern=pattern, end_datetime=end_datetime)))

                if inplace:
                    if column in self.anonymized_columns:
                        print(f'`{column}` column already anonymized!')
                    else:
                        self._df[column] = ser
                        self.anonymized_columns.append(column)
                        self.unanonymized_columns.remove(column)
                        self._methods_applied[column] = self._synthetic_data
                else:
                    temp[column] = ser
            if not inplace:
                return temp

    
    def column_suppression(self,
                          columns,
                          inplace = True):
        '''
        Redact a column (drop)
        Based on pandas `drop` method 

        Parameters
        ----------
            columns : Union[str, List[str]]
            inplace : bool, default True

        Returns
        ----------
            ser : None if inplace = True, else pandas Series or pandas DataFrame
        '''
        # if single column is passed
        if isinstance(columns, str) or (len(columns) == 1 and isinstance(columns, list)):
            if isinstance(columns, list):
                columns = columns[0]

            if inplace:
                if columns in self.anonymized_columns:
                    print(f'`{columns}` column already anonymized!')
                else:
                    self._df.drop(columns, axis = 1, inplace = True)
                    self.anonymized_columns.append(columns)
                    self.unanonymized_columns.remove(columns)
                    self._methods_applied[columns] = self._drop
            else:
                return self._df.drop(columns, axis = 1, inplace = False)

        # if a list of columns is passed
        else:
            if inplace:
                for column in columns:
                    if column in self.anonymized_columns:
                        print(f'`{column}` column already anonymized!')
                    else:
                        self._df.drop(column, axis = 1, inplace = True)
                        self.anonymized_columns.append(column)
                        self.unanonymized_columns.remove(column)
                        self._methods_applied[column] = self._drop
            else:
                return self._df.drop(columns, axis = 1, inplace = False)


    def numeric_binning(self,
                        columns,
                        bins = 4,
                        inplace = True):
        '''
        Bin values into discrete intervals.
        Based on pandas `cut` method 

        Parameters
        ----------
            columns : Union[str, List[str]]
            bins : int, default 4 
            inplace : bool, default True

        Returns
        ----------
            ser : None if inplace = True, else pandas Series or pandas DataFrame
        '''
        # if a single column is passed 
        if isinstance(columns, str) or (len(columns) == 1 and isinstance(columns, list)):
            if isinstance(columns, list):
                columns = columns[0]

            ser = pd.cut(self._df[columns], bins=bins, precision=0)
    
            if inplace:
                if columns in self.anonymized_columns:
                    print(f'`{columns}` column already anonymized!')
                else:
                    self._df[columns] = ser
                    self.anonymized_columns.append(columns)
                    self.unanonymized_columns.remove(columns)
                    self._methods_applied[columns] = self._bin
            else:
                return ser
            
        # if a list of columns is passed 
        else:
            temp = pd.DataFrame()

            for column in columns:
                ser = pd.cut(self._df[column], bins=bins, precision=0)

                if inplace:
                    if column in self.anonymized_columns:
                        print(f'`{column}` column already anonymized!')
                    else:
                        self._df[column] = ser
                        self.anonymized_columns.append(column)
                        self.unanonymized_columns.remove(column)
                        self._methods_applied[column] = self._bin
                else:
                    temp[column] = ser

            if not inplace:
                return temp
    
    
    def categorical_resampling(self,
                               columns,
                               inplace = True):
        '''
        Sampling from the same distribution

        Parameters
        ----------
            columns : Union[str, List[str]]
            inplace : bool, default True

        Returns
        ----------
            ser : None if inplace = True, else pandas Series or pandas DataFrame
        '''
        # if a single column is passed 
        if isinstance(columns, str) or (len(columns) == 1 and isinstance(columns, list)):
            if isinstance(columns, list):
                columns = columns[0]

            counts = self._df[columns].value_counts(normalize = True)

            if inplace:
                if columns in self.anonymized_columns:
                    print(f'`{columns}` column already anonymized!')
                else:
                    self._df[columns] = np.random.choice(counts.index, p = counts.values, size = len(self._df))
                    self.anonymized_columns.append(columns)
                    self.unanonymized_columns.remove(columns)
                    self._methods_applied[columns] = self._sample
            else:
                return pd.Series(np.random.choice(counts.index, p = counts.values, size = len(self._df)))

        # if a list of columns is passed
        else:
            temp = pd.DataFrame()

            for column in columns:
                counts = self._df[column].value_counts(normalize = True)

                if inplace:
                    if column in self.anonymized_columns:
                        print(f'`{column}` column already anonymized!')
                    else:
                        self._df[column] = np.random.choice(counts.index, p = counts.values, size = len(self._df))
                        self.anonymized_columns.append(column)
                        self.unanonymized_columns.remove(column)
                        self._methods_applied[column] = self._sample
                else:
                    temp[column] = np.random.choice(counts.index, p = counts.values, size = len(self._df))
            if not inplace:
                return temp


    def _info(self):
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

        return t
    

    def info(self):
        '''
        Print a summary of the a DataFrame.
        `status = 1 ` - anonymized
        `status = 0 ` - unanonymized

        Returns
        ----------
            None
        '''
        t  = Texttable(150)
        t.header(['Column', 'Status', 'Method'])

        for i in range(len(self.columns)):
            column = self.columns[i]
            
            if column in self.anonymized_columns:
                status = 1
                method = self._methods_applied[column]
            else:
                status = 0
                method = ''

            row = [column, status, method]
            t.add_row(row)

        print(t.draw())


    def available_methods(self, dtype = None):
        '''
        Print a list of available methods
        '''
        text = self._available_methods
        splitted = {'numeric': ['numeric_noise', 'numeric_binning', 'numeric_masking', 'numeric_rounding'],
                    'categorical': ['categorical_fake', 'categorical_fake_auto', 'categorical_resampling', 'categorical_tokenization', 'categorical_email_masking'],
                    'datetime': ["datetime_fake", "datetime_noise"],
                    'general': ['column_suppression']}
        
        if dtype == None or dtype == 'all':
            print(text)
        else:
            print(*splitted[dtype], sep='\t')
            
    
    def fake_methods(self, letter = None):
        '''
        Print a list of faker's methods
        '''
        text = self._fake_methods

        if letter == None or letter == 'all':
            print(text)
        else:
            for line in text.split('\n'):
                if line[0] == letter.upper():
                    print(line[3:])

        
    def to_df(self):
        ''' 
        Convert dfAnonymizer object back to pandas DataFrame

        Returns
        ----------
        DataFrame object
        '''
        
        return self._df.copy()
