from sklearn.svm import SVR
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.externals import joblib

conn = sqlite3.connect('gap_data.db')

for signal in ['<','>']:
    for target in [2]:
        x = []
        x_mean = []
        x2 = []
        x2_mean = []
        df = pd.read_sql('select * from gap_data where "2"<1 and "2">-1', conn)
        df = df.ix[:,['Symbol', 'Date', 'Signal', 'Wk', 'Mth', 'Vol_Chg', str(target)]]

        df[str(target)] = pd.to_numeric(df[str(target)])

        if signal=='>':
            df = df[df['Signal']>0]
            gap = "up"
        elif signal=='<':
            df = df[df['Signal']<0]
            gap = "down"

        df = df[df['Signal'].abs()>.1]
        df = df.dropna()


        #df.ix[:, 2:5] = scale(df.ix[:, 2:5])
        clf = SVR(C=1.0, epsilon=0.2)
        clf.fit(df.ix[:,2:], df.ix[:,len(df.columns)-1])
        joblib.dump(clf, 'models/gap_%s.pkl' % gap)

        for i in range(20):
            a_train, a_test, b_train, b_test = train_test_split(df.ix[:,:-1], df.ix[:,len(df.columns)-1], test_size=.35)
            clf = SVR(C=1.0, epsilon=0.2)
            clf.fit(a_train.ix[:,2:], b_train)

            predicted = clf.predict(a_test.ix[:,2:])


            a_test['Predicted'] = predicted
            a_test['Bin'] = np.round(a_test['Predicted'].rank(pct=True)*(4))
            a_test['Actual'] = b_test


            # create box plot
            df_box = a_test[['Bin','Actual']]

            dtf = df_box.boxplot(by='Bin')
            dtf.plot()
            fig = dtf.get_figure()
            fig.savefig("histograms/%s_%s.png" % (gap, i))
            plt.close()

            a_test = a_test.sort_values('Predicted')

            a_test = a_test.dropna()

            out_df = a_test[a_test['Bin']==0]
            cut_off = max(out_df['Predicted'])
            for j in out_df.iterrows():
                x.append([cut_off, j[1]['Actual']])

            out_df = a_test[a_test['Bin']==4]
            cut_off = min(out_df['Predicted'])
            for j in out_df.iterrows():
                x2.append([cut_off, j[1]['Actual']])

            #a_test.to_csv('test_output_%i_%s.csv' % (target, signal))

        x = pd.DataFrame(x, columns=['Cutoff', 'Return'])
        x2 = pd.DataFrame(x2, columns=['Cutoff', 'Return'])

        #x = str(pd.Series(x).mean()*100)[:6]
        #x2 = str(pd.Series(x2).mean()*100)[:6]
        x_r_mean = str(x['Return'].median()*100)[:6]

        x_c_mean = str(x['Cutoff'].median()*100)[:6]
        x2_r_mean = str(x2['Return'].median()*100)[:6]

        x2_c_mean = str(x2['Cutoff'].median()*100)[:6]

        print(target, signal, len(x)/20, len(x2)/20, x_c_mean, x2_c_mean, x_r_mean,  x2_r_mean, float(x2_r_mean)+(float(x_r_mean)*-1))




        """
        x = str(pd.Series(x).mean()*100)[:6]
        x_mean = str(pd.Series(x_mean).mean()*100)[:6]
        x2 = str(pd.Series(x2).mean()*100)[:6]
        x2_mean = str(pd.Series(x2_mean).mean()*100)[:6]
        """
