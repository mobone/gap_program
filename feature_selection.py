import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, mutual_info_regression

df = pd.read_csv('nosql_data_longer.csv')
df_corr = df.corr()
features = list(df_corr['Day 5'].index)[:-7]
df = df[features+['Day 5']]

# remove features that have too many nans
null_counts = df.isnull().sum()

too_many = float(len(df))*.1

null_counts = null_counts[null_counts<too_many]


df = df[list(null_counts.index)+['Day 5']]

df = df.dropna()
#print(df)
X = df.ix[:,:-1]
y = df['Day 5']


k = SelectKBest(f_regression, k=18)
k = k.fit(X,y)

print(list(X.columns[k.get_support()]))
