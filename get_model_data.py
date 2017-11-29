from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
import pandas as pd
import requests as r
import sqlite3
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from requests_toolbelt.threaded import pool
import re
import pandas_datareader as pdr
import warnings
warnings.filterwarnings('ignore')
"""
Gets back test data from yahoo finance, to be used to generate model
"""

conn = sqlite3.connect('gap_data.db')
yahoo_conn = sqlite3.connect('yahoo_data.db')

def get_changes(df):

    df['Percent_change'] = df['Open'] / df['Close'].shift(1) - 1


    df = df.reset_index()
    all_changes = []
    for i in df[df['Percent_change'].abs()>.05].index:
        if len(df.ix[i-61:i-1])<60:
            continue

        signal_date = str(df.ix[i,'Date']).split(' ')[0]
        signal = df.ix[i,'Percent_change']
        start_price = df.ix[i,'Open']
        volume = df.ix[i-1,'Volume']
        avg_vol = df.ix[i-61:i-2,'Volume'].mean()
        change_in_vol = volume / avg_vol - 1

        change = [symbol, signal_date, signal, avg_vol, change_in_vol, start_price]
        if start_price < 1 or avg_vol == 0 or volume == 0 or avg_vol<500000:
            return

        for j in [5,20,60]:
            try:
                change.append(df.ix[i-j]['Close'] / start_price - 1)
            except:
                change.append(None)

        for j in range(0,6):
            try:
                change.append(df.ix[i+j]['Close'] / start_price - 1)
            except:
                change.append(None)

        all_changes.append(change)
    out_df = pd.DataFrame(all_changes, columns=['Symbol','Date','Signal','Avg_Vol', 'Vol_Chg', 'Start_Price', 'Wk','Mth','Qtr','0','1','2','3','4','5'])
    if out_df.empty:
        return

    out_df.to_sql('gap_data', conn, if_exists='append', index=False)

try:
    cur = conn.cursor()
    cur.execute('''DROP TABLE gap_data''')
    conn.commit()
except Exception as e:
    print(e)


# get list of symbols that have split
df = pd.read_sql("select * from "+'A', yahoo_conn).tail(300).reset_index()
month = datetime.strptime(df.ix[0,'Date'].split(' ')[0], '%Y-%m-%d')
urls = []
while month<datetime.now():
    urls.append('https://eresearch.fidelity.com/eresearch/conferenceCalls.jhtml?tab=splits&begindate='+month.strftime('%m/%d/%y'))
    month = month + relativedelta(months=1)

p = pool.Pool.from_urls(urls, num_processes=20)
p.join_all()

split_symbols = []
for response in p.responses():
    symbols = pd.read_html(response.content)[1]['Symbol '].str.split(':')
    for i in symbols:
        split_symbols.append(i[0])


# get all symbols
urls = []
pages = r.get('https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o500,sh_opt_short').content
pages = int(re.findall(r'Total: </b>[0-9]*', str(pages))[0].split('>')[1])
print(pages)
for count in range(1,pages,20):
    url = 'https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o500,sh_opt_short&r=' + str(count)
    urls.append(url)

p = pool.Pool.from_urls(urls, num_processes=20)
p.join_all()


symbols = None
for response in p.responses():
    start = response.content.find(b'Total: ')
    end = response.content.find(b'Filters: ')
    df = pd.read_html(response.content[start:end])[0]
    df = df.ix[1:,1]

    if symbols is None:
        symbols = df
    else:
        symbols = symbols.append(df)


urls = []
for symbol in symbols.values:
    if symbol in split_symbols:
        print('split!', symbol)
        continue
    try:
        try:
            df = pd.read_sql("select * from "+symbol, yahoo_conn)
        except Exception as e:
            df = pdr.get_data_yahoo(symbol)
            df = df.reset_index()
            print('getting yahoo!', symbol)
            df.to_sql(symbol, yahoo_conn, if_exists='replace', index=False)
        df = df.tail(350)

        get_changes(df)
    except Exception as e:
        print(e)
        continue
