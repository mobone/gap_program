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
        finviz_alert_start = start_date.strftime('%Y-%m-%d') + ' 9:30:00'
        print('Next alert time: ',finviz_alert_start)
        while datetime.now()<datetime.strptime(finviz_alert_start, '%Y-%m-%d %H:%M:%S'):
            sleep(5)
        try:
            finviz_alerts()
        except Exception as e:
            print(e)

        closer_start = start_date.strftime('%Y-%m-%d') + ' 15:05:00'
        print('Next close time: ',closer_start)
        while datetime.now()<datetime.strptime(closer_start, '%Y-%m-%d %H:%M:%S'):
            sleep(5)
        try:
            closer()
        except Exception as e:
            print(e)
        start_date = start_date + timedelta(days=1)


@app.route('/')
def index():
    conn = sqlite3.connect('gap_data.db')
    today = datetime.now().strftime('%b %d, %Y')
    df = pd.read_sql('select * from alerts where Start_Date = "%s"' % today, conn)

    df = df[['Symbol', 'Play', 'Signal', 'Prediction', 'Vol_Chg', 'Wk', 'Mth','Start_Price']]

    return render_template('index.html', todays_alerts=df.values, today=datetime.now().strftime('%B %d, %Y'))


if __name__ == '__main__':
    app.config.from_pyfile('gap_app.cfg')
    scheduler = APScheduler()
    scheduler.init_app(app)
    scheduler.start()

    if app.config['DEBUG'] is False:
        scheduler.add_job(id='get_finviz_alerts',func='gap_site:process_manager')

        app.run(host='0.0.0.0', port=8080)
    else:
        app.run()
