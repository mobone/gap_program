import requests as r
import requests_cache
import pandas as pd
from requests_toolbelt.threaded import pool

requests_cache.install_cache('nosql_cache')
docs = r.get('http://mobone:C00kie32!@24.7.228.52:5984/finviz_data/_all_docs').content
docs = eval(docs)['rows']

urls = []
for doc in docs:
    urls.append('http://mobone:C00kie32!@24.7.228.52:5984/finviz_data/' + doc['id'])


p = pool.Pool.from_urls(urls, num_processes=20)
p.join_all()

def p2f(x):
    return x.str.strip('%').astype(float)/100

def convert_to_float(value):
    if 'K' in value:
        value = float(value[:-1])*1000
    elif 'M' in value:
        value = float(value[:-1])*1000000
    elif 'B' in value:
        value = float(value[:-1])*1000000000
    return float(value)

df = None
prices_df = None
for response in p.responses():
    doc = eval(response.content)
    symbol, date = doc['_id'].split('_')
    if prices_df is None:
        prices_df = pd.DataFrame([[symbol,date, float(doc['Price'])]])
    else:
        prices_df = prices_df.append(pd.DataFrame([[symbol,date, float(doc['Price'])]]))

    if abs(float(doc['Change'].strip('%'))/100)<.049:
        continue

    if df is None:
        df = pd.DataFrame(doc, index=[doc['_id']])
    else:
        df = df.append(pd.DataFrame(doc, index=[doc['_id']]))

prices_df.columns = ['Symbol', 'Date', 'Price']
prices_df = prices_df.sort_values(by=['Symbol', 'Date'])
prices_df = prices_df.reset_index(drop=True)
for row in df.iterrows():

    symbol, date = row[1]['_id'].split('_')
    start_price = float(row[1]['Price'])
    price_df = prices_df[prices_df['Symbol']==symbol]
    cur_date_index = int(price_df[price_df['Date']==date].index.values[0])


    for i in range(cur_date_index+1,cur_date_index+6):
        try:
            df.loc[row[0], 'Day '+str(i-cur_date_index)] = (price_df.loc[i]['Price'] / start_price) - 1
        except:
            pass

df.to_csv("nosql_data_before.csv")
df['Change'] = p2f(df['Change'])

df['Week'], df['Month'] = df['Volatility'].str.split(' ',1).str
df['52W Low'], df['52W High'] = df['52W Range'].str.split(' - ',1).str
df = df.drop(['52W Range', 'Volatility', '_rev', '_id'], axis=1)
df = df.replace('-', df.replace(['-'], [None]))

# remove features that have too many nans
null_counts = df.isnull().sum()
too_many = float(len(df))*.3

null_counts = null_counts[null_counts<too_many]
df = df[null_counts.index.values]

for col in df.columns:
    try:
        if df[col].str.contains('%').any():
            df[col] = p2f(df[col])
    except:
        pass

convert_me = ['Market Cap', 'Income', 'Avg Volume', 'Shs Outstand', 'Shs Float']
for row in df.iterrows():
    for row_to_convert in convert_me:
        try:
            df.ix[row[0],row_to_convert] = convert_to_float(row[1][row_to_convert])
        except:
            continue

for col in df.columns:
    try:
        df[col] = df[col].apply(pd.to_numeric)
    except:
        pass
df.to_csv("nosql_data_longer.csv")



corr = df.corr()[['Day 1','Day 2','Day 3','Day 4','Day 5']].dropna().sort_values(by=['Day 5'])
print(corr)
corr.to_csv('corr.csv')
print('complete')
