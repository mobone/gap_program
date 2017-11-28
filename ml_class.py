from sklearn.svm import SVR, SVC
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.externals import joblib
import warnings
warnings.filterwarnings('ignore')

class machine(object):
    """
    Used to find the best machine learning model, reading back test data from
    datebase
    """

    def __init__(self, gap_direction, gap_percentage, hold_days, machine, sim_num):
        self.sim_num = sim_num
        self.machine_id = machine+'_'+gap_direction+'_'+str(gap_percentage)+'_'+str(hold_days)
        self.gap_direction = gap_direction
        self.gap_percentage = gap_percentage
        self.hold_days = str(hold_days)
        self.machine = machine

        self.main()


    def main(self):
        self.get_data()

        if self.machine=='Classifier':
            self.bin_data()
            target = self.df.ix[:,'Bin']
            self.data = self.df.ix[:,:-1]
        elif self.machine=='Regression':
            target = self.df.ix[:,self.hold_days]
            self.data = self.df

        a_train, a_test, b_train, b_test = train_test_split(self.data, target, test_size=.35)
        self.get_machine(a_train, b_train)
        self.predict(a_test)
        self.get_results()

    def get_data(self):
        conn = sqlite3.connect('gap_data.db')
        df = pd.read_sql('select * from gap_data where "%s"<1 and "%s">-1' % (self.hold_days, self.hold_days), conn)

        df = df.ix[:,['Symbol', 'Date', 'Signal', 'Wk', 'Mth','Vol_Chg', self.hold_days]]

        #df = df.fillna(df.mean())

        df[self.hold_days] = pd.to_numeric(df[self.hold_days])
        if self.gap_direction=="up":
            df = df[df['Signal']>0]
        elif self.gap_direction=="down":
            df = df[df['Signal']<0]

        df = df[df['Signal'].abs()>self.gap_percentage]
        self.df = df.dropna()


    def bin_data(self):
        self.df['Bin'] = np.round(self.df[self.hold_days].rank(pct=True)*(2))

    def get_machine(self, a_train, b_train):
        if self.machine=='Classifier':
            self.clf = SVC(C=1.0)
            self.out_clf = SVC(C=1.0)
        elif self.machine=='Regression':
            self.clf = SVR(C=1.0, epsilon=0.2)
            self.out_clf = SVR(C=1.0, epsilon=0.2)
        try:

            self.clf.fit(a_train.ix[:,2:-1], b_train)
        except Exception as e:
            print('um', e)
            b_train.to_csv("bad_data_a.csv")
            b_train.to_csv("bad_data_b.csv")
            input()

        if self.sim_num == 0:
            if self.machine=='Classifier':
                self.out_clf.fit(self.df.ix[:,2:-2], self.df.ix[:,'Bin'])
            elif self.machine=='Regression':
                self.out_clf.fit(self.df.ix[:,2:-1], self.df.ix[:,self.hold_days])

            joblib.dump(self.out_clf, 'models/%s.pkl' % self.machine_id)

    def predict(self, a_test):
        try:
            predicted = self.clf.predict(a_test.ix[:,2:-1])
        except:
            a_test.to_csv('bad_data_c.csv')
            return
        a_test['Predicted'] = predicted
        self.result_df = a_test

    def get_results(self):
        if self.machine=='Classifier':
            negative = self.result_df[self.result_df['Predicted']==0]
            positive = self.result_df[self.result_df['Predicted']==2]
        elif self.machine=='Regression':
            self.result_df = self.result_df.sort_values(by='Predicted')
            negative = self.result_df.head(20)
            positive = self.result_df.tail(20)
            #negative = self.result_df[self.result_df['Predicted']<-.07]
            #positive = self.result_df[self.result_df['Predicted']>.036]

        self.negative = negative[self.hold_days]
        self.positive = positive[self.hold_days]
        #print(negative)
        #print('----')
        #print(positive)
        #input()
        if self.machine=='Classifier':
            self.cutoff_0 = 0
            self.cutoff_1 = 4
        elif self.machine=='Regression':
            self.cutoff_0 = max(negative['Predicted'])
            self.cutoff_1 = min(positive['Predicted'])





if __name__ == '__main__':
    total_results = []
    for gap_percentage in [.05]:
        for gap_direction in ['up']:
            for hold_days in [2,3,4,5]:
                for machine_type in ['Classifier']:

                    negative_results = None
                    positive_results = None
                    for j in range(30):
                        try:
                            x = machine(gap_direction, gap_percentage, hold_days, machine_type, j)
                            if negative_results is None:
                                negative_results = x.negative
                                positive_results = x.positive
                            else:
                                negative_results = negative_results.append(x.negative)
                                positive_results = positive_results.append(x.positive)
                        except Exception as e:
                            print(e)
                            continue



                    total_results.append([x.machine_id, x.cutoff_0, x.cutoff_1, len(negative_results)/30, len(positive_results)/30, negative_results.median(), positive_results.median(), negative_results.mean(), positive_results.mean()])
                    df = pd.DataFrame(total_results, columns = ['Machine', 'Cutoff_0', 'Cutoff_1', 'Neg_Count', 'Pos_Count','Neg_Median', 'Pos_Median', 'Neg_Mean', 'Pos_Mean'])




    df = pd.DataFrame(total_results, columns = ['Machine', 'Cutoff_0', 'Cutoff_1', 'Neg_Count', 'Pos_Count','Neg_Median', 'Pos_Median', 'Neg_Mean', 'Pos_Mean'])
    df = df[(df['Pos_Count']>8) & (df['Neg_Count']>8)]
    df['Diff_Median'] = (df['Neg_Median']-df['Pos_Median'])*-1
    df['Diff_Mean'] = (df['Neg_Mean']-df['Pos_Mean'])*-1
    df = df.sort_values(by='Diff_Median')
    print(df)
    df.to_csv("machine_results.csv", index=False)
