import pandas as pd
import requests as r
from requests_toolbelt.threaded import pool
from datetime import datetime, date, timedelta
from sklearn.externals import joblib
import warnings
import re
from nyse_holidays import *
import sqlite3
import os
warnings.filterwarnings('ignore')

class finviz_alerts(object):
    """
    Triggers each morning at 9am, collects big movers of the day,
    runs their data through prediction, stores in database
    """
    def __init__(self):
        # check if alerts have already been collected for today
        # TODO
        """
        conn = sqlite3.connect('gap_data.db')
        today = datetime.now().strftime('%b %d, %Y')
        df = pd.read_sql('select * from alerts where Start_Date = "%s"' % today, conn)
        if not df.empty:
            return
        """


        print("Getting alerts", datetime.now())
        self.get_alerts()
        conn = sqlite3.connect('gap_data.db')
        machine_names = sorted(os.listdir('./models'))[1:]
        original_companies = self.companies.copy()
        for machine_name in machine_names:
            self.companies = original_companies.copy()
            machine_ids = machine_name.split('_')
            self.machine_id = machine_name.replace("_both","").replace(".pkl","")
            self.hold_days = int(machine_ids[3][:1])
            self.cut_off = float(machine_ids[2])

            if self.cut_off == .1:
                self.companies = self.companies[self.companies['Signal'].abs()>.1]

            self.get_machines(machine_name)
            self.get_predictions()
            self.get_close_dates()

            self.companies.to_sql(self.machine_id, conn, if_exists='replace', index=False)

    def get_close_dates(self):
        for company in self.companies.iterrows():

            start_date = datetime.strptime(company[1]['Start_Date'], '%b %d, %Y')

            for i in range(self.hold_days-1):
                start_date = start_date+timedelta(days=1)
                while start_date.strftime('%y%m%d') == NYSE_holidays()[0].strftime('%y%m%d') or start_date.weekday()>=5:
                    start_date = start_date+timedelta(days=1)

            self.companies.loc[company[0],'Close_Date'] = start_date.strftime('%b %d, %Y')
            self.companies.loc[company[0],'Close_Price'] = None
            self.companies.loc[company[0],'Percent_Change'] = None
            self.companies.loc[company[0],'Hold_Days'] = self.hold_days

    def get_predictions(self):
        self.companies = self.companies.dropna()
        self.companies['Play'] = None
        self.companies['Prediction'] = self.clf.predict(self.companies[['Signal', 'Wk', 'Mth', 'Vol_Chg']])
        self.companies.loc[self.companies['Prediction']==0, 'Play'] = 'Short'
        self.companies.loc[self.companies['Prediction']==4, 'Play'] = 'Long'
        self.companies = self.companies.dropna(subset=['Play'])

    def get_machines(self, name):
        self.clf = joblib.load('models/%s' % name)


    def get_alerts(self):
        url_up = 'https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o500,sh_opt_short,ta_perf_d5o&ft=3&r='
        url_down = 'https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o500,sh_opt_short,ta_perf_d5u&ft=3&r='

        # get total pages
        up_count = int(re.findall(b'Total: </b>[0-9]*', r.get(url_up).content)[0].split(b'>')[1])
        down_count = int(re.findall(b'Total: </b>[0-9]*', r.get(url_down).content)[0].split(b'>')[1])

        # get all alerts
        df = None
        for url, count in [(url_up, up_count), (url_down,down_count)]:
            for page in range(1,count,20):
                finviz_page = r.get(url+str(page)).content
                start = finviz_page.find(b'Total:')
                end = finviz_page.find(b'Filters', start)
                if df is None:
                    df = pd.read_html(finviz_page[start:end], header=0)[0]
                else:
                    df = df.append(pd.read_html(finviz_page[start:end], header=0)[0])

        df = df.reset_index()

        # get all the quotes relating to each stock, for other related data
        urls = []
        for symbol in df['Ticker']:
            urls.append('https://finviz.com/quote.ashx?t=' + symbol)

        p = pool.Pool.from_urls(urls, num_processes=20)
        p.join_all()

        companies = []
        for response in p.responses():
            company = self.get_company_data(response)
            companies.append(company)

        self.companies = pd.DataFrame(companies, columns = ['Symbol', 'Start_Date', 'Signal', 'Avg_Vol', 'Vol', 'Vol_Chg', 'Wk', 'Mth', 'Qtr', 'Start_Price'])
        self.companies = self.companies[self.companies['Start_Price']>2]




    def get_company_data(self, response):
        # formats all the other related data
        start = response.content.find(b'snapshot-table2')-200
        end = response.content.find(b'</table>',start+300)
        df = pd.read_html(response.content[start:end])[0]

        finviz_data = None
        for i in range(0,len(df.columns),2):
            table_section = df.ix[:,i:i+1]
            table_section.columns = ['Key', 'Value']
            table_section.index = table_section['Key']
            table_section = table_section['Value']
            if finviz_data is None:
                finviz_data = table_section
            else:
                finviz_data = finviz_data.append(table_section)
        #'Symbol', 'Date', 'Signal', 'Wk', 'Mth', 'Vol_Chg',
        symbol = response.request_kwargs['url'].split('=')[1]
        date = datetime.now().strftime("%b %d, %Y")
        signal = float(finviz_data['Change'][:-1])/100
        price = float(finviz_data['Price'])
        avg_vol = self.convert_to_float(finviz_data['Avg Volume'])
        vol = float(finviz_data['Volume'])
        vol_chg = (vol-avg_vol)/avg_vol
        try:
            chg_wk = float(finviz_data['Perf Week'][:-1])/100
        except:
            chg_wk = None
        try:
            chg_mth = float(finviz_data['Perf Month'][:-1])/100
        except:
            chg_mth = None
        try:
            chg_qtr = float(finviz_data['Perf Quarter'][:-1])/100
        except:
            chg_qtr = None

        company_data = [symbol, date, signal, avg_vol, vol, vol_chg, chg_wk, chg_mth, chg_qtr, price]

        return company_data

    def convert_to_float(self, value):
        if 'K' in value:
            value = float(value[:-1])*1000
        elif 'M' in value:
            value = float(value[:-1])*1000000
        elif 'B' in value:
            value = float(value[:-1])*1000000000
        return float(value)


class closer(object):
    def __init__(self):
        print("Getting close prices", datetime.now())
        # get closing companies
        conn = sqlite3.connect('gap_data.db')
        close_date = datetime.now().strftime('%b %d, %Y')
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
        for table_name in tables.values:
            table_name = table_name[0]

            print(table_name)
            try:
                input()
                df = pd.read_sql('select * from "%s" where Close_Date = "%s"' % (table_name, close_date), conn)
            except Exception as e:
                print(e)


            for company in df.iterrows():
                symbol = company[1]['Symbol']
                finviz_page = r.get('https://finviz.com/quote.ashx?t=' + symbol).content

                start = finviz_page.find(b'snapshot-table2')-200
                end = finviz_page.find(b'</table>',start+300)

                finviz_df = pd.read_html(finviz_page[start:end])[0]
                close_price = float(finviz_df.loc[10,11])
                start_price = float(company[1]['Start_Price'])
                percent_change = (close_price-start_price)/start_price
                cur = conn.cursor()
                cur.execute('update alerts set Close_Price = %f, Percent_Change = %f where Symbol="%s" and Close_Date="%s"' % (close_price, percent_change, symbol, close_date))
            conn.commit()

x = finviz_alerts()
#x = closer()
