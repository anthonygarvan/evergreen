__author__ = 'root'

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LogisticRegression
from pandas import *
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from pandas.io.parsers import read_csv
from sklearn.linear_model.perceptron import Perceptron
from sklearn.svm import LinearSVC
from sklearn.svm import SVR, SVC
import cPickle as pickle
import nltk
from TopicModel import TopicModel
from time import time
import pylab as P
import json
from Preprocessor import Preprocessor

class Evergreen:
    def __init__(self):

        self.submissionPath = 'submission_08.csv'
        self.estimators = 50
        self.testSize = .25

        #self.trainAndTest()
        #self.TopicModel = TopicModel(getTitle=True, getBody=True)

        print 'Done'

    def printMetrics(self, y_test, y_predicted):
        print "AUC Score: %f" % roc_auc_score(y_test, y_predicted)
        P.figure()
        P.hist(y_test-y_predicted)
        P.show()

    def trainChildren(self, df):
        print 'training child models...'
        self.topicModelBoth = TopicModel(getTitle=True, getBody=True, getUrl=True)
        self.topicModelBoth.trainFromDataFrame(df)

        self.topicModelBody = TopicModel(getTitle=False, getBody=True, getUrl=False)
        self.topicModelBody.trainFromDataFrame(df)

        self.topicModelTitle = TopicModel(getTitle=True, getBody=False, getUrl=False)
        self.topicModelTitle.trainFromDataFrame(df)

    def generateSubmission(self):
        print '\n***starting submission***'
        self.benchmarkMetaModel(.01)
        raw = self.getRaw(self.testPath)

        print 'generating prediction'
        X_test, y_test, urlid = self.preprocess(raw)
        #y_predicted = self.model.predict(X_test)
        y_predicted = self.topicModelBoth.addTopicModel(raw['boilerplate'])
        if y_test:
            self.printMetrics(y_test, y_predicted)

        data = []
        for x in xrange(0, len(urlid)):
            data.append([urlid[x], y_predicted[x]])

        submission = DataFrame(data=data, columns = ['urlid', 'label'])
        submission.to_csv(self.submissionPath, index=False)
        print 'submission written to %s' % self.submissionPath

    def train(self, X, y):
        print 'training parent model...'
        self.model = RandomForestRegressor(n_estimators=self.estimators)
        #self.model = AdaBoostRegressor(n_estimators=self.estimators)
        #self.model = ExtraTreesRegressor(n_estimators=self.estimators)
        #self.model = SVR()
        self.model.fit(X, y)

    def benchmarkMetaModel(self, testSize):
        print '\n***starting benchmark***'
        raw = self.getRaw(self.trainPath)

        rawRecords = raw.to_records(index=False)
        dfr_train, dfr_test = train_test_split(rawRecords, test_size=testSize, random_state=5)
        df_train = DataFrame().from_records(dfr_train)
        df_test = DataFrame().from_records(dfr_test)

        self.trainChildren(df_train)
        X_train, y_train, urlid = self.preprocess(df_train)
        self.train(X_train, y_train)

        print 'generating prediction'
        X_test, y_test, urlid = self.preprocess(df_test)
        y_predicted = self.model.predict(X_test)
        print 'random forest benchmark: '
        self.printMetrics(y_test, y_predicted)

        X_topicModelBoth = self.topicModelBoth.addTopicModel(df_train['boilerplate'])
        X_topicModelTitle = self.topicModelTitle.addTopicModel(df_train['boilerplate'])
        X_topicModelBody = self.topicModelBody.addTopicModel(df_train['boilerplate'])
        X_rf = self.model.predict(X_train)
        X_train_meta = np.array([X_rf, X_topicModelBoth, X_topicModelTitle, X_topicModelBody]).transpose()

        metaModel = LogisticRegression()
        #metaModel = SVC(probability=True)
        #metaModel = AdaBoostRegressor(n_estimators=20)
        metaModel.fit(X_train_meta, y_train)
        X_test_topicModelBoth = self.topicModelBoth.addTopicModel(df_test['boilerplate'])
        X_test_topicModelTitle = self.topicModelTitle.addTopicModel(df_test['boilerplate'])
        X_test_topicModelBody = self.topicModelBody.addTopicModel(df_test['boilerplate'])

        print 'topic model result:'
        self.printMetrics(y_test, X_test_topicModelBoth)
        self.writeReport(df_test, y_test, X_test_topicModelBoth, '/home/tony/topicModelReport.csv')
        X_test_rf = self.model.predict(X_test)
        X_test_meta = np.array([X_test_rf, X_test_topicModelBoth, X_test_topicModelTitle, X_test_topicModelBody]).transpose()
        y_predicted_meta = metaModel.predict_proba(X_test_meta)[:,1]
        #y_predicted_meta = metaModel.predict(X_test_meta)

        print 'meta model result:'
        self.printMetrics(y_test, y_predicted_meta)

    def benchmark(self, testSize):
        print '\n***starting benchmark***'
        P = Preprocessor()
        raw = P.getRaw(P.trainPath)

        rawRecords = raw.to_records(index=False)
        dfr_train, dfr_test = train_test_split(rawRecords, test_size=testSize, random_state=5)
        df_train = DataFrame().from_records(dfr_train)
        df_test = DataFrame().from_records(dfr_test)
        #self.trainChildren(df_train)
        X_train, y_train, urlid = P.preprocess(df_train)
        self.train(X_train, y_train)

        importances = DataFrame({'feature': Series(X_train.columns),'importance':Series(self.model.feature_importances_)})
        print importances.sort('importance', ascending=False)
        print 'generating prediction'
        X_test, y_test, urlid = P.preprocess(df_test)
        y_predicted = self.model.predict(X_test)
        print 'random forest benchmark: '
        self.printMetrics(y_test, y_predicted)

start = time()
e = Evergreen()
e.benchmark(.25)
#e.benchmarkMetaModel(.25)
#e.generateSubmission()
finish = time()
elapsed = finish-start
print 'elapsed time: %f seconds' % elapsed

