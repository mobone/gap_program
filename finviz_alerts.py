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

        conn = sqlite3.connect('gap_data.db')
        # get machines
        sql = 'select * from nosql_data_pruned where Neg_Median<-.01 and Pos_Median>.01 order by diff_mean desc limit 20;'
        machines = pd.read_sql(sql,conn).ix[:,['Features','Neg_Cutoff','Pos_Cutoff','Hold_Time']]

        # get all features we might use
        features_list = []
        for machine in machines['Features'].values:
            machine = machine.replace('\\','')
            features_list.extend(machine.split('_'))

        self.total_features_set = list(set(features_list))
        self.get_backtest_dataset()

        print("Getting alerts", datetime.now())
        self.get_alerts()
        self.company_data = pd.concat([self.total_dataset, self.company_data])


        for self.features,self.neg_cutoff,self.pos_cutoff,self.hold_days in machines.values:
            self.features = '['+self.features.replace('\/','/').replace(' ',', ')+']'
            print(self.features)

            self.machine_id = [self.features.replace('/','-o-').replace(' ','-'),
                               str(self.neg_cutoff), str(self.pos_cutoff)]
            self.machine_id = '__'.join(self.machine_id)
            print(self.machine_id)
            input()

            self.get_machine()
            self.get_predictions()
            self.get_close_dates()

            self.companies.to_sql(self.machine_id.split('_')[0], conn, if_exists='append', index=False)
            break

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
        self.companies = self.company_data[self.features]
        self.companies = self.companies.dropna()
        print(self.companies)
        exit()
        self.companies['Play'] = None
        self.companies['Prediction'] = self.clf.predict(self.companies[['Signal', 'Wk', 'Mth', 'Vol_Chg']])
        self.companies.loc[self.companies['Prediction']<self.neg_cutoff, 'Play'] = 'Short'
        self.companies.loc[self.companies['Prediction']>self.pos_cutoff, 'Play'] = 'Long'
        self.companies = self.companies.dropna(subset=['Play'])

    def get_machine(self):
        self.clf = joblib.load('models/%s.pkl' % self.machine_id)

    def p2f(self,x):
        return x.str.strip('%').astype(float)/100

    def get_alerts(self):
        url_up = 'https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o300,sh_opt_short,ta_perf_d5o&ft=3&r='
        url_down = 'https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o300,sh_opt_short,ta_perf_d5u&ft=3&r='

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
                break
            break

        df = df.reset_index()

        # get all the quotes relating to each stock, for other related data
        urls = []
        for symbol in df['Ticker']:
            urls.append('https://finviz.com/quote.ashx?t=' + symbol)

        p = pool.Pool.from_urls(urls, num_processes=20)
        p.join_all()

        # generate total company df
        self.company_data = []
        index_num = 0
        for response in p.responses():
            symbol = response.request_kwargs['url'].split('=')[1]
            start = response.content.find(b'snapshot-table2')-200
            end = response.content.find(b'</table>',start+300)
            df = pd.read_html(response.content[start:end])[0]
            df = pd.DataFrame(df.values.reshape(-1, 2), columns=['key', index_num])
            df = df.set_index('key').T
            df = df[self.total_features_set]
            df = df.replace('-', df.replace(['-'], [None]))
            df['Ticker'] = symbol
            self.company_data.append(df)
            index_num+=1
        self.company_data = pd.concat(self.company_data)

        # format data
        for col in self.company_data.columns[:-1]:
            if self.company_data[col].str.contains('%').any():
                self.company_data[col] = self.p2f(self.company_data[col])
            self.company_data[col] = self.company_data[col].apply(pd.to_numeric)
        print(self.company_data)

    def get_backtest_dataset(self):
        df = pd.read_csv('nosql_data.csv')
        df.index.name = 'Index'
        col_str = str(df.columns).replace('\/','/').split('[')[1].split(']')[0]
        df.columns = eval('['+col_str+']')
        self.total_dataset = df[self.total_features_set]

    def get_company_data(self, response):
        # formats all the other related data

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
            try:
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
finviz_alerts()
