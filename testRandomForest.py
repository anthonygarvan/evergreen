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
import pickle


class Evergreen:
    def __init__(self):

        self.trainPath = 'train.tsv'
        self.testPath = 'test.tsv'
        self.submissionPath = 'submission_03.csv'
        self.estimators = 100
        self.testSize = 0
        self.debugMode = True

        self.distinctCategories = self.getDistinct(self.trainPath, True, 'alchemy_category')
        self.distinctIsNews = self.getDistinct(self.trainPath, True, 'is_news')

        self.trainAndTest()

        print 'Done'

    def printMetrics(self, y_test, y_predicted):
        print "AUC Score: %f" % roc_auc_score(y_test, y_predicted)

    # Load training file
    def getData(self, path):
        raw = read_csv(path, sep='\t', na_values=['?']).fillna(-5)
        X = []
        y = []

        for s in raw:
            if s == 'label':
                y = np.array(raw[s].tolist())
            else:
                if s == 'urlid':
                    urlid = raw[s].tolist()
                if raw[s].dtype == 'float64':
                    X.append(raw[s].tolist())
                if raw[s].dtype == 'int64':
                    X.append(raw[s].tolist())
                if s == 'alchemy_category':
                    categories = raw[s]

            #print '%s : %s' % (s, raw[s].dtype)

        X = self.addDistinctColumns(self.distinctCategories, X, raw['alchemy_category'])
        X = self.addDistinctColumns(self.distinctIsNews, X, raw['is_news'])
        X = np.array(X).transpose()
        y = np.array(y)

        n = 5
        #print 'X:'
        #print X[n]
        #print 'X shape:'
        #print X.shape
        #print "Raw:"
        #print raw.irow(n)
        #print raw['alchemy_category_score']
        #print raw['is_news']
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

    def generateSubmission(self, model):
        X , y, urlid = self.getData(self.testPath)
        y_predicted = model.predict(X)

        data = []

        for x in xrange(0, len(urlid)):
            data.append([urlid[x], y_predicted[x]])

        submission = DataFrame(data=data, columns = ['urlid', 'label'])
        submission.to_csv(self.submissionPath, index=False)


    def trainAndTest(self):
        X, y, urlid = self.getData(self.trainPath)

        # Random Forest
        rf = RandomForestRegressor(n_estimators=self.estimators)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.testSize)
        rf.fit(X_train, y_train)
        if self.testSize == 0:
            self.generateSubmission(rf)
        else:
            y_predicted = rf.predict(X_test)
            self.printMetrics(y_test, y_predicted)


e = Evergreen()
