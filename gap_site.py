from flask import Flask, session, request, flash, jsonify, url_for, redirect, render_template, abort, g
from datetime import datetime
from flask_apscheduler import APScheduler
import pandas as pd
import sqlite3
import requests as r
from requests_toolbelt.threaded import pool
from finviz_alerts import *
app = Flask(__name__)

def process_manager():
    print("hey")

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.config.from_pyfile('gap_app.cfg')
    scheduler = APScheduler()
    scheduler.init_app(app)
    scheduler.start()
    scheduler.add_job(id='get_finviz_alerts',func='gap_site:process_manager')
    app.run()
