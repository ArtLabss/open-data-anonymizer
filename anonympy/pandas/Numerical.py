import cape_privacy as cp
import pandas as pd

class AnonymNum(object):
      """
Initialize pandas DataFrame as an AnonymNum object.
To be used if the data is of the following format:
- np.float32, np.float64, np.byte, np.short, np.int32, np.int64

Args:
      df: pandas DataFrame

Returns:
      AnonymNum object
"""
      def __init__(self, df):
            if df.__class__.__name__ != "DataFrame":
                  raise Exception(f"{df} is not a pandas DataFrame.")
            self._df = df
      
