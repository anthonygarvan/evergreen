__author__ = 'root'
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from pandas import *
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.cross_validation import train_test_split
from pandas.io.parsers import read_csv
from sklearn.linear_model.perceptron import Perceptron
from sklearn.svm import LinearSVC
import cPickle as pickle
import nltk
from TopicModel import TopicModel
from time import time


class Evergreen:
    def __init__(self):

        self.trainPath = 'train.tsv'
        self.testPath = 'test.tsv'
        self.submissionPath = 'submission_05.csv'
        self.estimators = 100
        self.testSize = .25

        self.distinctCategories = self.getDistinct(self.trainPath, True, 'alchemy_category')
        self.distinctIsNews = self.getDistinct(self.trainPath, True, 'is_news')

        #self.trainAndTest()
        #self.TopicModel = TopicModel(getTitle=True, getBody=True)

        print 'Done'

    def printMetrics(self, y_test, y_predicted):
        print "Final AUC Score: %f" % roc_auc_score(y_test, y_predicted)

    def getRaw(self, path):
        raw = read_csv(path, sep='\t', na_values=['?']).fillna(-5)
        return raw

    def trainChildren(self, df):
        print 'training child models...'
        self.topicModelBoth = TopicModel(getTitle=True, getBody=True)
        self.topicModelBoth.trainFromDataFrame(df)

        self.topicModelBody = TopicModel(getTitle=False, getBody=True)
        self.topicModelBody.trainFromDataFrame(df)

        self.topicModelTitle = TopicModel(getTitle=True, getBody=False)
        self.topicModelTitle.trainFromDataFrame(df)

    def preprocess(self, df):
        print 'extracting features...'
        X = []
        y = []

        for s in df:
            if s == 'label':
                y = np.array(df[s].tolist())
            else:
                if s == 'urlid':
                    urlid = df[s].tolist()
                if df[s].dtype == 'float64':
                    X.append(df[s].tolist())
                if df[s].dtype == 'int64':
                    X.append(df[s].tolist())
                if s == 'alchemy_category':
                    categories = df[s]

            #print '%s : %s' % (s, raw[s].dtype)

        X = self.addDistinctColumns(self.distinctCategories, X, df['alchemy_category'])
        X = self.addDistinctColumns(self.distinctIsNews, X, df['is_news'])
        X = self.topicModelBoth.addTopicModel(X, df['boilerplate'])
        X = self.topicModelTitle.addTopicModel(X, df['boilerplate'])
        X = self.topicModelBody.addTopicModel(X, df['boilerplate'])

        X = np.array(X).transpose()
        y = np.array(y)

        return X, y, urlid

    def addDistinctColumns(self, distinct, X, distinctSeries):
        for category in distinct:
            newCol = []
            for row in distinctSeries:
                if row == category:
                    newCol.append(1)
                else:
                    newCol.append(0)
            X.append(newCol)

        newCol = []
        for row in distinctSeries:
            newCol.append(distinct[row])
        X.append(newCol)
        return X

    def getDistinct(self, path, makeNew, columnName):
        distinctPath = '%s.pkl' % columnName

        if makeNew:
            raw = read_csv(path, sep='\t',na_values=['?']).fillna(-5)
            categories = raw[columnName]
            distinct = {}
            distinctNum = 0
            for category in categories:
                if category not in distinct:
                    distinct[category] = distinctNum
                    distinctNum += 1

            f = open(distinctPath, 'w')
            pickle.dump(distinct, f)
        else:
            f = open(distinctPath, 'r')
            distinct = pickle.load(f)
        return distinct

    def generateSubmission(self):
        print 'starting submission'
        raw = self.getRaw(self.testPath)

        self.trainChildren(raw)
        X, y, urlid = self.preprocess(raw)
        self.train(X, y)

        print 'generating prediction'
        X_test, y_test, urlid = self.preprocess(raw)
        y_predicted = self.rf.predict(X_test)
        self.printMetrics(y_test, y_predicted)

        data = []
        for x in xrange(0, len(urlid)):
            data.append([urlid[x], y_predicted[x]])

        submission = DataFrame(data=data, columns = ['urlid', 'label'])
        submission.to_csv(self.submissionPath, index=False)
        print 'submission writted to %s' % self.submissionPath

    def train(self, X, y):
        print 'training random forest...'
        self.rf = RandomForestRegressor(n_estimators=self.estimators)
        self.rf.fit(X, y)

    def benchmark(self):
        print 'starting benchmark'
        raw = self.getRaw(self.trainPath)

        rawRecords = raw.to_records()
        dfr_train, dfr_test = train_test_split(rawRecords, test_size=self.testSize)
        df_train = DataFrame().from_records(dfr_train)
        df_test = DataFrame().from_records(dfr_test)

        self.trainChildren(df_train)
        X_train, y_train, urlid = self.preprocess(df_train)
        self.train(X_train, y_train)

        print 'generating prediction'
        X_test, y_test, urlid = self.preprocess(df_test)
        y_predicted = self.rf.predict(X_test)
        self.printMetrics(y_test, y_predicted)



start = time()
e = Evergreen()
e.benchmark()
finish = time()
elapsed = finish-start
print 'elapsed time: %f seconds' % elapsed

