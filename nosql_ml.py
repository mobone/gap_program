from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split
import pandas as pd
import sqlite3
import itertools
from random import shuffle
from time import sleep
from sklearn import preprocessing
from sklearn.externals import joblib
import numpy as np
from multiprocessing import Process, Queue

class machine(Process):
    def __init__(self,process_q):
        Process.__init__(self)
        self.process_q = process_q
        self.sim_count = 6
        self.trade_count = 10
        self.bins = 2


    def run(self):
        self.conn = sqlite3.connect('gap_data.db')
        self.original_data = pd.read_csv('nosql_data.csv')
        while self.process_q.qsize():
            self.target,self.machine,self.features = self.process_q.get()
            self.features = list(self.features)
            self.data = self.original_data[self.features+[self.target]].dropna()

            self.negative = []
            self.positive = []
            self.predictions_neg = []
            self.predictions_pos = []

            for self.sim_num in range(self.sim_count):
                self.get_data()
                self.get_model()
                self.get_predictions()

                if len(set(self.predictions))<5 and self.machine=='Regression':
                    break

            self.get_metrics()

            # discard models with unsatisfactory metrics
            if self.metric_df['Pos_Mean'][0]<.025 or self.metric_df['Neg_Mean'][0]>-.025:
                continue

            if self.machine=='Classifier':
                if len(self.negative)<self.sim_count*(self.trade_count-3):
                    continue
                if len(self.positive)<self.sim_count*(self.trade_count-3):
                    continue

            # save model if diff is great enough
            if self.metric_df['Diff_Mean'][0]>.05 and self.metric_df['Diff_Median'][0]>.03:
                self.save_model()

            self.metric_df.to_sql('nosql_data_pruned', self.conn, if_exists='append',index=False)

    def save_model(self):
        self.get_out_model()
        filename = [self.metric_df['Features'][0].replace(' ','-').replace('\/', '-o-'),
                    str(float(self.metric_df['Neg_Cutoff'][0])),
                    str(float(self.metric_df['Pos_Cutoff'][0]))]
        filename = '__'.join(filename)
        joblib.dump(self.out_clf, 'models/%s.pkl' % filename)

    def get_metrics(self):
        df1 = pd.DataFrame({'Neg_Cutoff': self.predictions_neg})
        df2 = pd.DataFrame({'Pos_Cutoff': self.predictions_pos})
        df3 = pd.DataFrame({'Negative': self.negative})
        df4 = pd.DataFrame({'Positive': self.positive})

        df = pd.concat([df1,df2,df3,df4], ignore_index=True,axis=1)
        df.columns = ['Neg_Cutoff', 'Pos_Cutoff', 'Negative', 'Positive']
        metrics = df.describe().transpose()

        row = [self.machine, '_'.join(self.features)]
        row.extend([metrics['max']['Neg_Cutoff'], metrics['min']['Pos_Cutoff'],
                   metrics['count']['Negative'], metrics['count']['Negative'],
                   metrics['mean']['Negative'], metrics['50%']['Negative'],
                   metrics['mean']['Positive'], metrics['50%']['Positive']])

        df = pd.DataFrame([row], columns = ['Machine_Type','Features', 'Neg_Cutoff', 'Pos_Cutoff',
                                            'Neg_Count','Pos_Count', 'Neg_Mean',
                                            'Neg_Median', 'Pos_Mean', 'Pos_Median'])
        df['Diff_Mean'] = df['Pos_Mean'] + (df['Neg_Mean']*-1)
        df['Diff_Median'] = df['Pos_Median'] + (df['Neg_Median']*-1)
        df['Hold_Time'] = self.target
        self.metric_df = df

    def bin_data(self):
        self.data['Bin'] = np.round(self.data[self.target].rank(pct=True)*(self.bins))

    def get_data(self):
        if self.machine=='Classifier':
            self.bin_data()
            split_data = train_test_split(preprocessing.scale(self.data[self.features]),
                                          self.data.ix[:,['Bin',self.target]], test_size=.4)
        elif self.machine=='Regression':
            split_data = train_test_split(preprocessing.scale(self.data[self.features]),
                                          self.data[self.target], test_size=.4)
        self.a_train, self.a_test, self.b_train, self.b_test = split_data

    def get_model(self):
        if self.machine=='Classifier':
            self.clf = SVC(C=1.0)
            self.clf.fit(self.a_train, self.b_train['Bin'])
        elif self.machine=='Regression':
            self.clf = SVR(C=1.0, epsilon=0.2)
            self.clf.fit(self.a_train, self.b_train)

    def get_out_model(self):
        if self.machine=='Classifier':
            self.out_clf = SVC(C=1.0)
            self.out_clf.fit(preprocessing.scale(self.data[self.features]),
                             self.data['Bin'])
        elif self.machine=='Regression':
            self.out_clf = SVR(C=1.0, epsilon=0.2)
            self.out_clf.fit(preprocessing.scale(self.data[self.features]),
                             self.data[self.target])

    def get_predictions(self):
        self.predictions = self.clf.predict(self.a_test)
        self.a_test = pd.DataFrame(self.a_test, columns = self.features)
        self.a_test['Predicted'] = list(self.predictions)


        self.a_test = self.a_test.sort_values(by=['Predicted'])
        if self.machine=='Classifier':
            self.a_test['Return'] = self.b_test[self.target].values
            self.negative.extend(self.a_test[self.a_test['Predicted']==0]['Return'].values)
            self.positive.extend(self.a_test[self.a_test['Predicted']==self.bins]['Return'].values)
            self.predictions_neg.extend(self.a_test[self.a_test['Predicted']==0]['Predicted'].values)
            self.predictions_pos.extend(self.a_test[self.a_test['Predicted']==self.bins]['Predicted'].values)
        elif self.machine=='Regression':
            self.a_test['Return'] = self.b_test.values
            self.negative.extend(self.a_test['Return'].head(self.trade_count).values)
            self.positive.extend(self.a_test['Return'].tail(self.trade_count).values)
            self.predictions_neg.extend(self.a_test['Predicted'].head(self.trade_count).values)
            self.predictions_pos.extend(self.a_test['Predicted'].tail(self.trade_count).values)


if __name__ == '__main__':
    process_q = Queue()

    features = ['Perf Half Y', 'Beta', 'Inst Trans', 'Sales Q\/Q', 'Gross Margin',
                'P\/S', 'EPS next 5Y', 'Perf YTD', 'P\/B', 'Sales past 5Y',
                'RSI (14)', 'Perf Quarter', 'SMA20', 'Recom', 'EPS past 5Y',
                'ROI', 'Change', 'Perf Year']


    if features is None:
        correls = pd.read_csv('corr.csv')
        correls.index = correls[correls.columns[0]]
        correls = correls.ix[:-5]
        corr = pd.DataFrame(correls.ix[:,'Day '+str(i)])
        features = list(corr[corr['Day '+str(i)].abs() >.03].index)

    feature_set = []
    for permute_length in range(3,7):
        for num_days in range(4,6):
            for machine_type in ['Regression']:
                for feature in list(itertools.permutations(features[:-1], r=permute_length)):
                    feature_set.append(['Day '+str(num_days),machine_type,feature])
    shuffle(feature_set)
    for feature in feature_set:
        process_q.put(feature)
    print(process_q.qsize())

    for process_id in range(8):
        p = machine(process_q)
        p.start()

    while process_q.qsize():
        sleep(60)
        print(process_q.qsize())
    while process_q.qsize():
        process_q.get()
