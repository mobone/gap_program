from flask import Flask, session, request, flash, jsonify, url_for, redirect, render_template, abort, g
from datetime import datetime
from flask_apscheduler import APScheduler
import pandas as pd
import sqlite3
import requests as r
from requests_toolbelt.threaded import pool
from finviz_alerts import *
from time import sleep
app = Flask(__name__)

def process_manager():
    start_date = datetime.today()

    while True:
        # get next start date
        while start_date.strftime('%y%m%d') == NYSE_holidays()[0].strftime('%y%m%d') or start_date.weekday()>=5:
            start_date = start_date + timedelta(days=1)

        # sleep until 9am
        finviz_alert_start = start_date.strftime('%Y-%m-%d') + ' 14:00:00'
        print('Next alert time: ',finviz_alert_start)
        while datetime.now()<datetime.strptime(finviz_alert_start, '%Y-%m-%d %H:%M:%S'):
            sleep(5)
        try:
            finviz_alerts()
        except Exception as e:
            print(e)

        closer_start = start_date.strftime('%Y-%m-%d') + ' 15:01:00'
        print('Next close time: ',closer_start)
        while datetime.now()<datetime.strptime(closer_start, '%Y-%m-%d %H:%M:%S'):
            sleep(5)
        try:
            closer()
        except Exception as e:
            print(e)
        start_date = start_date + timedelta(days=1)


@app.route('/', methods=['GET', 'POST'])
def index(this_table_name=None):

    conn = sqlite3.connect('alerts.db')
    # get table names
    table_names = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)

    table_names_formatted = table_names.name.str.split('__',0)[1:]

    if request.method != 'POST':
        this_table_name = table_names['name'][1]

    model_name = this_table_name.split('__')[0]
    # get todays alerts
    today = datetime.now().strftime('%b %d, %Y')
    df = pd.read_sql('select * from "%s" where Start_Date = "%s"' % (this_table_name,today), conn)
    metric_cols = list(df.columns[:6].values)+['Change', 'Prediction']
    ordered_columns = ['Ticker', 'Change']+list(df.columns[:6].values)+['Price', 'Prediction', 'Play']

    print(ordered_columns)


    df = df[ordered_columns]

    for col in metric_cols:
        print(col)
        df[col] = pd.Series(["{0:.2f}%".format(val * 100) for val in df[col]], index = df.index)
    #df = df[['Symbol', 'Play', 'Signal', 'Prediction', 'Vol_Chg', 'Wk', 'Mth','Start_Price']]
    print(df)
    return render_template('index.html', model_name = model_name, todays_columns = ordered_columns, todays_alerts=df.values, today=datetime.now().strftime('%B %d, %Y'))


if __name__ == '__main__':
    app.config.from_pyfile('gap_app.cfg')
    scheduler = APScheduler()
    scheduler.init_app(app)
    scheduler.start()

    if app.config['DEBUG'] is False:
        scheduler.add_job(id='get_finviz_alerts',func='gap_site:process_manager')
        app.run(host='0.0.0.0', port=8090)
    else:
        app.run()
