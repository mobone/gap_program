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

    def __init__(self, gap_direction, gap_percentage, hold_days, machine):
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

        df = df.ix[:,['Symbol', 'Date', 'Signal', 'Wk', 'Mth', 'Vol_Chg', self.hold_days]]

        df[self.hold_days] = pd.to_numeric(df[self.hold_days])
        if self.gap_direction=="up":
            df = df[df['Signal']>0]
        elif self.gap_direction=="down":
            df = df[df['Signal']<0]

        df = df[df['Signal'].abs()>self.gap_percentage]
        self.df = df.dropna()

    def bin_data(self):
        self.df['Bin'] = np.round(self.df[self.hold_days].rank(pct=True)*(4))

    def get_machine(self, a_train, b_train):
        if self.machine=='Classifier':
            self.clf = SVC(C=1.0)
            self.out_clf = SVC(C=1.0)
        elif self.machine=='Regression':
            self.clf = SVR(C=1.0, epsilon=0.2)
            self.out_clf = SVR(C=1.0, epsilon=0.2)

        self.clf.fit(a_train.ix[:,2:-1], b_train)

        if self.machine=='Classifier':
            self.out_clf.fit(self.df.ix[:,2:-2], self.df.ix[:,'Bin'])
        elif self.machine=='Regression':
            self.out_clf.fit(self.df.ix[:,2:-1], self.df.ix[:,self.hold_days])
        joblib.dump(self.out_clf, 'models/gap_%s.pkl' % self.gap_direction)

    def predict(self, a_test):
        predicted = self.clf.predict(a_test.ix[:,2:-1])
        a_test['Predicted'] = predicted
        self.result_df = a_test

    def get_results(self):
        if self.machine=='Classifier':
            negative = self.result_df[self.result_df['Predicted']==0]
            positive = self.result_df[self.result_df['Predicted']==4]
        elif self.machine=='Regression':
            self.result_df = self.result_df.sort_values(by='Predicted')
            negative = self.result_df.head(20)
            positive = self.result_df.tail(20)

        self.negative = negative[self.hold_days]
        self.positive = positive[self.hold_days]
        self.cutoff_0 = max(negative['Predicted'])
        self.cutoff_1 = max(positive['Predicted'])





if __name__ == '__main__':
    total_results = []
    for gap_percentage in [.05,.1]:
        for gap_direction in ['up', 'down']:
            for hold_days in [0,1,2,3,4]:
                results = []
                for j in range(30):
                    x = machine(gap_direction, gap_percentage, hold_days, 'Regression')
                    for result in range(len(x.negative)):
                        results.append([gap_percentage, gap_direction, hold_days, x.cutoff_0, x.cutoff_1, x.negative.iloc[result], x.positive.iloc[result]])

                df = pd.DataFrame(results)
                total_results.append([df[0].values[0], df[1].values[0], df[2].values[0], df[3].median(), df[4].median(), df[5].median(), df[6].median()])


    df = pd.DataFrame(total_results, columns = ['Gap_Perc', 'Gap_Dir', 'Hold_days', 'Cutoff_0', 'Cutoff_1', 'Neg_Mean', 'Pos_Mean'])
    df['Diff'] = (df['Neg_Mean']-df['Pos_Mean'])*-1
    df = df.sort_values(by='Diff')
    print(df)
