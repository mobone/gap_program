from flask import Flask, session, request, flash, jsonify, url_for, redirect, render_template, abort, g
from datetime import datetime
from flask_apscheduler import APScheduler
import pandas as pd
import sqlite3
import requests as r
from requests_toolbelt.threaded import pool
from finviz_alerts import *
app = Flask(__name__)

@app.route('/')
def index():
    df = pd.read_html(r.get('https://finviz.com/').content)

    df[5].columns = df[5].ix[0]
    df[6].columns = df[6].ix[0]
    df[5] = df[5].drop(0)
    df[6] = df[6].drop(0)
    df[5] = df[5].append(df[6])
    df = df[5].reset_index(drop=True)

    df = df[(df['Signal']=='Top Losers') | (df['Signal'] == 'Top Gainers')]
    df.index = df['Ticker']
    df = df[['Ticker', 'Last', 'Change', 'Volume', 'Signal']]
    urls = []
    for symbol in df['Ticker']:
        url = 'https://finviz.com/quote.ashx?t='+symbol
        urls.append(url)


    p = pool.Pool.from_urls(urls, num_processes=10)
    p.join_all()

    for response in p.responses():
        symbol = response.request_kwargs['url'].split('=')[1]
        start = response.content.find(b'snapshot-table2')
        end = response.content.find(b'</table>', start)
        ticker_df = pd.read_html(response.content[start-100:end])[0]

        ticker_table = None
        for i in range(0,len(ticker_df.columns),2):
            table_section = ticker_df.ix[:,i:i+1]
            table_section.columns = ['Key', 'Value']
            table_section.index = table_section['Key']
            table_section = table_section['Value']
            if ticker_table is None:
                ticker_table = table_section
            else:
                ticker_table = ticker_table.append(table_section)

        if 'K' in ticker_table['Avg Volume']:
            ticker_table['Avg Volume'] = float(ticker_table['Avg Volume'][:-1])*1000
        elif 'M' in ticker_table['Avg Volume']:
            ticker_table['Avg Volume'] = float(ticker_table['Avg Volume'][:-1])*1000000
        elif 'B' in ticker_table['Avg Volume']:
            ticker_table['Avg Volume'] = float(ticker_table['Avg Volume'][:-1])*1000000000

        print(symbol, ticker_table['Avg Volume'], ticker_table['Perf Week'], ticker_table['Perf Month'])


    return render_template('index.html')


if __name__ == '__main__':
    app.config.from_pyfile('gap_app.cfg')
    scheduler = APScheduler()
    scheduler.init_app(app)
    scheduler.start()
    scheduler.add_job(id='get_finviz_alerts',func='finviz_alerts:start_process')
    app.run()
