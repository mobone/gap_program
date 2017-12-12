from sklearn.svm import SVR, SVC
import pandas as pd
import requests as r
from requests_toolbelt.threaded import pool
from datetime import datetime, date, timedelta
from sklearn.externals import joblib
from sklearn import preprocessing
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
        results_conn = sqlite3.connect('alerts.db')
        # get machines
        sql = 'select * from nosql_data_machines_pruned_3 order by diff_mean desc limit 25;'
        machines = pd.read_sql(sql,conn).ix[:,['Features','Neg_Cutoff_Mean','Pos_Cutoff_Mean','Hold_Time', 'Model_Type']]

        # save machines to another table
        machines.to_sql('selected_models', results_conn, if_exists='replace', index=False)

        # get all features we might use
        features_list = []
        for machine in machines['Features'].values:
            machine = machine.replace('\\','')
            features_list.extend(machine.split('_'))
        self.total_features_set = list(set(features_list))
        self.hold_lenghts = list(set(machines['Hold_Time']))

        print(self.total_features_set)

        self.get_backtest_dataset()

        print("Getting alerts", datetime.now())
        self.get_alerts()
        self.company_data = pd.concat([self.backtest_dataset, self.company_data])
        print('Input Len:',len(self.company_data))

        for self.features,self.neg_cutoff,self.pos_cutoff,self.hold_days,self.model_type in machines.values:
            self.hold_days = int(self.hold_days.split(' ')[1])
            self.features = self.features.replace('\/','/').split('_')
            print(self.model_type, self.features, self.hold_days, self.neg_cutoff, self.pos_cutoff)
            self.machine_id = ['_'.join(self.features).replace('/','-o-').replace(' ','-'),
                               str(self.neg_cutoff), str(self.pos_cutoff)]
            self.machine_id = '__'.join(self.machine_id)

            self.get_machine()
            self.get_predictions()
            self.get_close_dates()
            print('Alerts Len:', len(self.companies))
            print(self.machine_id)
            self.companies.to_sql(self.machine_id, results_conn, if_exists='append', index=False)


    def get_close_dates(self):
        for company in self.companies.iterrows():

            start_date = datetime.strptime(company[1]['Start_Date'], '%b %d, %Y')

            for i in range(self.hold_days):
                start_date = start_date+timedelta(days=1)
                while start_date.strftime('%y%m%d') == NYSE_holidays()[0].strftime('%y%m%d') or start_date.weekday()>=5:
                    start_date = start_date+timedelta(days=1)

            self.companies.loc[company[0],'Close_Date'] = start_date.strftime('%b %d, %Y')
            self.companies.loc[company[0],'Hold_Days'] = self.hold_days
            self.companies.loc[company[0],'Close_Price'] = None
            self.companies.loc[company[0],'Percent_Change'] = None

    def get_predictions(self):

        self.companies = self.company_data[self.features+['Change','Ticker','Start_Date','Price']]

        self.companies = self.companies.dropna(subset=self.features)

        self.companies['Play'] = None
        if self.model_type == 'Single':
            self.companies['Prediction'] = self.clf.predict(preprocessing.scale(self.companies[self.features]))
            self.companies.loc[self.companies['Prediction']<self.neg_cutoff, 'Play'] = 'Short'
            self.companies.loc[self.companies['Prediction']>self.pos_cutoff, 'Play'] = 'Long'
            self.companies = self.companies.dropna(subset=['Play','Ticker'])
            self.companies.index = self.companies['Ticker']
        elif self.model_type == 'Dual':
            neg_dual_data = self.companies[self.companies['Change']<0]
            neg_predictions = self.positive_clf.predict(preprocessing.scale(neg_dual_data[self.features]))
            pos_dual_data = self.companies[self.companies['Change']>0]
            pos_predictions = self.negative_clf.predict(preprocessing.scale(pos_dual_data[self.features]))
            neg_dual_data['Prediction'] = neg_predictions
            pos_dual_data['Prediction'] = pos_predictions
            self.companies = pd.concat([neg_dual_data, pos_dual_data])
            self.companies.loc[self.companies['Prediction']<self.neg_cutoff, 'Play'] = 'Short'
            self.companies.loc[self.companies['Prediction']>self.pos_cutoff, 'Play'] = 'Long'
            self.companies = self.companies.dropna(subset=['Play','Ticker'])
            self.companies.index = self.companies['Ticker']

    def get_model_data(self):
        target = 'Day '+str(self.hold_days)
        cleaned_backtest = self.backtest_dataset[self.features+['Change',target]].dropna()
        # select the data specific to this feature set
        if self.model_type == 'Single':
            self.model_data, self.target_data = (preprocessing.scale(cleaned_backtest[self.features]),
                                                                     cleaned_backtest[target])
        elif self.model_type == 'Dual':
            positive_set = cleaned_backtest[cleaned_backtest['Change']>0]
            negative_set = cleaned_backtest[cleaned_backtest['Change']<0]

            self.positive_model_data, self.pos_target_data = (preprocessing.scale(positive_set[self.features]),
                                                                             positive_set[target])
            self.negative_model_data, self.neg_target_data = (preprocessing.scale(negative_set[self.features]),
                                                                             negative_set[target])


    def get_machine(self):
        self.get_model_data()
        if self.model_type == 'Single':
            self.clf = SVR(C=1.0, epsilon=0.2) # regression only
            self.clf.fit(self.model_data, self.target_data)
        elif self.model_type == 'Dual':
            self.positive_clf = SVR(C=1.0, epsilon=0.2) # regression only
            self.negative_clf = SVR(C=1.0, epsilon=0.2) # regression only
            self.positive_clf.fit(self.positive_model_data, self.pos_target_data)
            self.negative_clf.fit(self.negative_model_data, self.neg_target_data)

    def p2f(self,x):
        return x.str.strip('%').astype(float)/100

    def get_alerts(self):
        url_up = 'https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o300,sh_opt_short,ta_perf_d5o&ft=3&r='
        url_down = 'https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o300,sh_opt_short,ta_perf_d5u&ft=3&r='

        # get total pages
        up_count = int(re.findall(b'Total: </b>[0-9]*', r.get(url_up).content)[0].split(b'>')[1])
        down_count = int(re.findall(b'Total: </b>[0-9]*', r.get(url_down).content)[0].split(b'>')[1])
        if up_count == 0:
            print('No +5% companies ')
        if down_count == 0:
            print('No -5% companies ')




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
            df['Week'], df['Month'] = df['Volatility'].str.split(' ',1).str

            df = df[self.total_features_set+['Price','Change']]
            df = df.replace('-', df.replace(['-'], [None]))
            df['Ticker'] = symbol
            df['Closed'] = False
            df['Start_Date'] = datetime.now().strftime("%b %d, %Y")
            self.company_data.append(df)
            index_num+=1
        self.company_data = pd.concat(self.company_data)

        # format data
        for col in self.company_data[self.total_features_set+['Change']]:
            if self.company_data[col].str.contains('%').any():
                self.company_data[col] = self.p2f(self.company_data[col])
            self.company_data[col] = self.company_data[col].apply(pd.to_numeric)

    def get_backtest_dataset(self):
        df = pd.read_csv('nosql_data_longer.csv')
        df.index.name = 'Index'
        col_str = str(df.columns).replace('\/','/').split('[')[1].split(']')[0]
        df.columns = eval('['+col_str+']')
        self.backtest_dataset = df[self.total_features_set+['Change']+self.hold_lenghts]


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
        for table_name in ['alerts.db', 'alerts_first_week.db']:
            print("Getting close prices for", table_name, datetime.now())
            # get closing companies
            conn = sqlite3.connect(table_name)
            close_date = datetime.now().strftime('%b %d, %Y')
            tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
            for table_name in tables.values:
                table_name = table_name[0]
                #print(table_name)
                try:
                    df = pd.read_sql('select * from "%s" where Close_Date >= "%s"' % (table_name, close_date), conn)
                except Exception as e:
                    continue

                urls = []
                for symbol in df['Ticker']:
                    urls.append('https://finviz.com/quote.ashx?t=' + symbol)
                p = pool.Pool.from_urls(urls, num_processes=20)
                p.join_all()
                analysis = []
                for response in p.responses():
                    symbol = response.request_kwargs['url'].split('=')[1]
                    start = response.content.find(b'snapshot-table2')-200
                    end = response.content.find(b'</table>',start+300)
                    finviz_df = pd.read_html(response.content[start:end])[0]

                    if len(df[df['Ticker']==symbol])>1:
                        continue

                    start_price = float(df[df['Ticker']==symbol]['Price'].values[0])
                    play = df[df['Ticker']==symbol]['Play'].values[0]

                    close_price = float(finviz_df.loc[10,11])

                    if play=='Long':
                        change = (close_price-start_price)/close_price
                    elif play=='Short':
                        change = ((close_price-start_price)/close_price)*-1

                    #print(symbol, play, start_price, close_price, change)

                    cur = conn.cursor()
                    sql = 'update "%s" set Close_Price = %f, Percent_Change = %f where Ticker = "%s"' % (table_name,close_price, change, symbol)
                    analysis.append([symbol, play, change])

                    cur.execute(sql)

                analysis = pd.DataFrame(analysis, columns = ['Symbol', 'Play', 'Change'])
                for play in set(analysis['Play']):

                    this_analysis = analysis[analysis['Play']==play]
                    desc = this_analysis.describe()
                    print(play, desc['Change']['count'], desc['Change']['50%'], desc['Change']['mean'], end = ' ')
                print()

                conn.commit()


#finviz_alerts()
closer()
