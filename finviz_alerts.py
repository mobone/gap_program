import pandas as pd
import requests as r
from requests_toolbelt.threaded import pool
from datetime import datetime
from sklearn.externals import joblib
import warnings
warnings.filterwarnings('ignore')

class finviz_alerts(object):
    """
    Triggers each morning at 9am, collects big movers of the day,
    runs their data through prediction, stores in database
    """

    def start_process(self):

        with open('machine_cutoffs.json', 'r') as f:
            self.param_dict = eval(f.read())
        self.get_machines()

        # TODO: put into white true, add dates and times for triggering
        self.get_alerts()
        self.get_predictions()
        # TODO: add storage into sqlite

    def get_predictions(self):
        self.up_companies = self.companies[self.companies['Signal']>0]
        self.down_companies = self.companies[self.companies['Signal']<0]

        up_predictions = self.clf_up.predict(self.up_companies[['Signal', 'Wk', 'Mth', 'Vol_Chg']])
        self.up_companies['Prediction'] = up_predictions
        down_predictions = self.clf_down.predict(self.down_companies[['Signal', 'Wk', 'Mth', 'Vol_Chg']])
        self.down_companies['Prediction'] = down_predictions

        print(self.up_companies)
        print(self.down_companies)

    def get_machines(self):
        self.clf_up = joblib.load('models/gap_up.pkl')
        self.clf_down = joblib.load('models/gap_down.pkl')

    def get_alerts(self):
        # TODO: iterate through pages
        url_up = 'https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o300,sh_opt_short,ta_perf_d5o&ft=3&r=1'
        url_down = 'https://finviz.com/screener.ashx?v=111&f=sh_avgvol_o300,sh_opt_short,ta_perf_d5u&ft=3&r=1'
        df = None
        for url in [url_up, url_down]:
            finviz_page = r.get(url).content
            start = finviz_page.find(b'Total:')
            end = finviz_page.find(b'Filters', start)
            if df is None:
                df = pd.read_html(finviz_page[start:end], header=0)[0]
            else:
                df = df.append(pd.read_html(finviz_page[start:end], header=0)[0])

        df = df.reset_index()
        urls = []
        for symbol in df['Ticker']:
            urls.append('https://finviz.com/quote.ashx?t=' + symbol)
        p = pool.Pool.from_urls(urls, num_processes=5)
        p.join_all()

        companies = []
        for response in p.responses():
            # find current share price
            symbol = response.request_kwargs['url'].split('=')[1]
            share_price = df[df['Ticker']==symbol]['Price']
            company = self.get_company_data(response, share_price)
            companies.append(company)

        self.companies = pd.DataFrame(companies, columns = ['Symbol', 'Date', 'Signal', 'Avg_Vol', 'Vol', 'Vol_Chg', 'Wk', 'Mth', 'Qtr', 'Start_Price'])

    def get_company_data(self, response, share_price):
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
        price = share_price.values[0]
        date = datetime.now().strftime("%b %d, %Y")
        avg_vol = self.convert_to_float(finviz_data['Avg Volume'])
        vol = float(finviz_data['Volume'])
        signal = float(finviz_data['Change'][:-1])/100
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





finviz_alerts()
