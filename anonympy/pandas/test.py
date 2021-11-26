from cape_privacy.pandas import dtypes
from cape_privacy.pandas.transformations import ColumnRedact
from cape_privacy.pandas.transformations import DatePerturbation
from cape_privacy.pandas.transformations import NumericRounding
from cape_privacy.pandas.transformations import NumericPerturbation
from cape_privacy.pandas.transformations import Tokenizer
import pandas as pd


def load_dataset(sess=None):
    dataset = pd.DataFrame({
        "name": ["alice", "bob"],
        "age": [34, 55],
        "birthdate": [pd.Timestamp(1985, 2, 23), pd.Timestamp(1963, 5, 10)],
        "salary": [59234.32, 49324.53],
        "ssn": ["343554334", "656564664"],
    })
    if sess is not None:
        return sess.createDataFrame(dataset)
    else:
        return dataset

# Load Pandas DataFrame
df = load_dataset()
print("Original Dataset:")
print(df.head())

# Define the transformations
tokenize = Tokenizer(max_token_len=10, key=b"my secret")
perturb_numric = NumericPerturbation(dtype=dtypes.Integer, min=-100, max=100)
perturb_date = DatePerturbation(frequency=("YEAR", "MONTH", "DAY"), min=(-10, -5, -5), max=(10, 5, 5))
round_numeric = NumericRounding(dtype=dtypes.Float, precision=-3)
redact_column = ColumnRedact(columns="ssn")

# Apply the transformations
df["name"] = tokenize(df["name"])
df["age"] = perturb_numric(df["age"])
df["salary"] = round_numeric(df["salary"])
df["birthdate"] = perturb_date(df["birthdate"])
df = redact_column(df)

print("Masked Dataset:")
print(df.head())


##class A:
##    def __init__(self):
##        pass
##
##    def sampleFunc(self, arg):
##        print('you called sampleFunc({})'.format(arg))
##
##m = globals()['A']()
##func = getattr(m, 'sampleFunc')
##func('sample arg')
##
### Sample, all on one line
##getattr(globals()['A'](), 'sampleFunc')('sample arg')
