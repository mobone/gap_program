from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
import pandas as pd
import requests as r
import sqlite3
from requests_toolbelt.threaded import pool
import re


"""
Gets back test data from yahoo finance, to be used to generate model
"""

conn = sqlite3.connect('gap_data.db')
def get_historical_data(response, number_of_days):
    data = []
    start = response.text.find('>Currency in')
    end = response.text.find('</table>',start)+20
    rows = bs(response.text[start:end], 'lxml').findAll('table')[0].tbody.findAll('tr')

    for each_row in rows:
        divs = each_row.findAll('td')
        if divs[1].span.text  != 'Dividend' and divs[1].span.text != 'Stock Split':
            data.append({'Date': divs[0].span.text,
                         'Open': float(divs[1].span.text.replace(',','')),
                         'Close': float(divs[4].span.text.replace(',','')),
                         'Volume': float(divs[6].span.text.replace(',',''))})

    df = pd.DataFrame(data[1:number_of_days])
    df = df[['Date', 'Open', 'Close', 'Volume']]

    return df



def perc_change(final, initial):
    perc_change = float((final-initial) / initial)
    return perc_change

def get_changes(response):

    symbol = response.request_kwargs['url'].split('/')[4]
    df = get_historical_data(response, 250)


    df['Overnight_change'] = None
    for i in range(1,len(df)):
        df.ix[i-1,'Overnight_change'] = perc_change(df.iloc[i-1]['Open'],df.iloc[i]['Close'])

    df = df.dropna()
    out_df = None
    for i in df[df['Overnight_change'].abs()>.05].index:

        signal_date = df.ix[i,'Date']
        signal = df.ix[i,'Overnight_change']
        start_price = df.ix[i,'Open']
        change = [symbol, signal_date, signal]

        volume = df.ix[i,'Volume']

        avg_volume = df.ix[i+1:,'Volume'].mean()


        change_in_volume = perc_change(volume, avg_volume)
        change.append(avg_volume)
        change.append(change_in_volume)

        for j in [5,20,60]:
            try:
                change_from_start = perc_change(df.ix[i+j]['Close'], start_price)
                change.append(change_from_start)
            except:
                change.append(None)

        for j in range(0,5):
            try:
                change_from_start = perc_change(df.ix[i-j]['Close'], start_price)
                change.append(change_from_start)
            except:
                change.append(None)

        out_df = pd.DataFrame([change], columns=['Symbol','Date','Signal','Avg_Vol', 'Vol_Chg', 'Wk','Mth','Qtr','0','1','2','3', '4'])
        store(out_df)

    return out_df

def store(df):
    df.to_sql('gap_data', conn, if_exists='append', index=False)


try:
    cur = conn.cursor()
    cur.execute('''DROP TABLE gap_data''')
    conn.commit()
except Exception as e:
    print(e)

symbols = None
urls = []
#2653
pages = r.get('https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o300,sh_opt_short').content
pages = int(re.findall(r'Total: </b>[0-9]*', str(pages))[0].split('>')[1])
print(pages)
for count in range(1,pages,20):
    url = 'https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o300,sh_opt_short&r=' + str(count)
    urls.append(url)

p = pool.Pool.from_urls(urls, num_processes=20)
p.join_all()

for response in p.responses():
    start = response.content.find(b'Total: ')
    end = response.content.find(b'Filters: ')
    df = pd.read_html(response.content[start:end])[0]
    df = df.ix[1:,1]

    if symbols is None:
        symbols = df
    else:
        symbols = symbols.append(df)


print(symbols)
urls = []
for symbol in symbols.values:
    url = "http://finance.yahoo.com/quote/" + symbol + "/history"
    urls.append(url)
print(len(urls))
p = pool.Pool.from_urls(urls, num_processes=20)
p.join_all()

for response in p.responses():
    try:
        print(response.request_kwargs['url'])
        get_changes(response)
    except Exception as e:
        print(e)
        pass
