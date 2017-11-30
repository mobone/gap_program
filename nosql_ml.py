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
        # get the processing queue
        self.process_q = process_q
        # set initial params, universal to all processors
        self.sim_count = 6      # n experiements per model
        self.trade_count = 8   # set the number of trades per model
        self.bins = 2           # used for classifiers, chooses the bin size

    def run(self):
        print('started')
        # connect to db for storage and read the input data csv
        self.conn = sqlite3.connect('gap_data.db')
        self.original_data = pd.read_csv('nosql_data.csv')

        # iterate over the process queue
        while not self.process_q.empty():
            sleep(.05)
            self.target,self.machine,self.features = self.process_q.get()

            self.features = list(self.features)
            # select the data specific to this feature set
            self.data = self.original_data[self.features+[self.target]].dropna()

            # used for storing the results of n experiements
            self.negative = []
            self.positive = []
            self.predictions_neg = []
            self.predictions_pos = []

            # run n experiements
            self.run_experiments(self.sim_count)

            # take in the results from the experience and get stats about performance
            if self.get_metrics()==False:
                continue


            # discard models with unsatisfactory metrics
            if self.metric_df['Pos_Mean'][0]<.02 or self.metric_df['Neg_Mean'][0]>-.02:
                continue
            if self.machine=='Classifier':
                if len(self.negative)<self.sim_count*(self.trade_count-3):
                    continue
                if len(self.positive)<self.sim_count*(self.trade_count-3):
                    continue
            #if self.metric_df['Diff_Mean'][0]<.05 or self.metric_df['Diff_Median'][0]<.03:
            #    continue
            print(self.metric_df)
            # save model if diff between neg predictions and pos predictions is great enough
            # this model is fit using 100% of the data, and will be used to make future predictions
            self.save_model()

            # store result in the database
            self.metric_df.to_sql('nosql_data_pruned_dual_model', self.conn, if_exists='append',index=False)

    def run_experiments(self, sim_count):
        for self.sim_num in range(sim_count):
            for self.model_type in ['Negative','Positive']:
                self.get_data()     # splits data into random train/test splits
                self.get_model()    # trains the Classifier or Regression model
                self.get_predictions()  # uses the test input to get output predictions

                # discard the model if its regression and producing only one prediction
                if len(set(self.predictions))<5 and self.machine=='Regression':
                    return False
        return True



    def get_data(self):
        if self.model_type == ['Negative']:
            self.data = self.data[self.data['Change']<0].copy()
        elif self.model_type == ['Positive']:
            self.data = self.data[self.data['Change']>0].copy()

        if self.machine=='Classifier':
            self.bin_data()
            split_data = train_test_split(preprocessing.scale(self.data[self.features]),
                                          self.data.ix[:,['Bin',self.target]], test_size=.4)
        elif self.machine=='Regression':
            split_data = train_test_split(preprocessing.scale(self.data[self.features]),
                                          self.data[self.target], test_size=.4)
        self.a_train, self.a_test, self.b_train, self.b_test = split_data

    def bin_data(self):
        # this turns continuous data into n classes, for us in classifier
        self.data['Bin'] = np.round(self.data[self.target].rank(pct=True)*(self.bins))

    def get_model(self):
        if self.machine=='Classifier':
            self.clf = SVC(C=1.0)
            self.clf.fit(self.a_train, self.b_train['Bin'])
        elif self.machine=='Regression':
            self.clf = SVR(C=1.0, epsilon=0.2)
            self.clf.fit(self.a_train, self.b_train)

    def get_predictions(self):
        self.predictions = self.clf.predict(self.a_test)
        self.a_test = pd.DataFrame(self.a_test, columns=self.features)
        self.a_test['Predicted'] = list(self.predictions)

        self.a_test = self.a_test.sort_values(by=['Predicted'])
        if self.machine=='Classifier':
            self.a_test['Return'] = self.b_test[self.target].values
            if self.model_type == 'Negative':
                self.negative.extend(self.a_test[self.a_test['Predicted']==0]['Return'].values)
                self.predictions_neg.extend(self.a_test[self.a_test['Predicted']==0]['Predicted'].values)
            elif self.model_type == 'Positive':
                self.positive.extend(self.a_test[self.a_test['Predicted']==self.bins]['Return'].values)
                self.predictions_pos.extend(self.a_test[self.a_test['Predicted']==self.bins]['Predicted'].values)
        elif self.machine=='Regression':
            self.a_test['Return'] = self.b_test.values
            if self.model_type == 'Negative':
                self.predictions_neg.extend(self.a_test['Predicted'].head(self.trade_count).values)
                self.negative.extend(self.a_test['Return'].head(self.trade_count).values)
            elif self.model_type == 'Positive':
                self.predictions_pos.extend(self.a_test['Predicted'].tail(self.trade_count).values)
                self.positive.extend(self.a_test['Return'].tail(self.trade_count).values)

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

    def save_model(self):
        self.get_out_model()
        filename = [self.metric_df['Features'][0].replace(' ','-').replace('\/', '-o-'),
                    str(float(self.metric_df['Neg_Cutoff'][0])),
                    str(float(self.metric_df['Pos_Cutoff'][0]))]
        filename = '__'.join(filename)
        joblib.dump(self.out_clf, 'models/%s.pkl' % filename)

    def get_out_model(self):
        if self.machine=='Classifier':
            self.out_clf = SVC(C=1.0)
            self.out_clf.fit(preprocessing.scale(self.data[self.features]),
                             self.data['Bin'])
        elif self.machine=='Regression':
            self.out_clf = SVR(C=1.0, epsilon=0.2)
            self.out_clf.fit(preprocessing.scale(self.data[self.features]),
                             self.data[self.target])


if __name__ == '__main__':
    process_q = Queue()

    features = ['Perf Half Y', 'Beta', 'Inst Trans', 'Sales Q\/Q', 'Gross Margin',
                'P\/S', 'EPS next 5Y', 'Perf YTD', 'P\/B', 'Sales past 5Y',
                'RSI (14)', 'Perf Quarter', 'SMA20', 'Recom', 'EPS past 5Y',
                'ROI', 'Change', 'Perf Year']

    # ability to toggle preset features vs finding those that have the highest correlation
    if features is None:
        correls = pd.read_csv('corr.csv')
        correls.index = correls[correls.columns[0]]
        correls = correls.ix[:-5]
        corr = pd.DataFrame(correls.ix[:,'Day '+str(i)])
        features = list(corr[corr['Day '+str(i)].abs() >.03].index)


    feature_set = []
    # permute all the different combinations of the possible features, of length 3 to n
    for permute_length in range(4,6):
        #hold stock for 4, 5 or 6 days, this is our target variable
        for num_days in range(4,6):
            # ability to change from Classifier and Regression models
            for machine_type in ['Regression']:
                for feature in list(itertools.permutations(features[:-1], r=permute_length)):

                    feature_set.append(['Day '+str(num_days),machine_type,feature])

    # shuffle the list so that the multiprocessing can discover good vs bad models randomly
    shuffle(feature_set)
    print(len(feature_set))

    # start the processors
    for process_id in range(1):
        print("starting")
        p = machine(process_q)
        p.start()



    # put them in the queue
    list = [process_q.put(feature) for feature in feature_set]








    # wait for the queue to empty
    while not process_q.empty():
        sleep(60)
        print('im')

    # if ctrl+c is encountered, the queue must be cleared before terminating
    while process_q.qsize():
        process_q.get()
