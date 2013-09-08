__author__ = 'root'

from pandas import *
import numpy as np
from pandas.io.parsers import read_csv
import cPickle as pickle
import json
import pylab as P
from sklearn.feature_selection import SelectKBest, chi2


class Preprocessor:
    def __init__(self):
        self.trainPath = 'train.tsv'
        self.testPath = 'test.tsv'

        self.distinctCategories = self.getDistinct(self.trainPath, True, 'alchemy_category')
        self.distinctIsNews = self.getDistinct(self.trainPath, True, 'is_news')

    def addDistinctColumns(self, distinct, X, distinctSeries):
        for category in distinct:
            newCol = []
            #colNames.append(category)
            for row in distinctSeries:
                if row == category:
                    newCol.append(1)
                else:
                    newCol.append(0)
            X[category] = Series(newCol)

    #newCol = []
    #for row in distinctSeries:
    #    newCol.append(distinct[row])
    #X.append(newCol)
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


    def preprocess(self, df):
        print 'extracting features...'
        X = {}
        y = []
        for s in df:
            if s == 'label':
                y = np.array(df[s].tolist())
            else:
                if s == 'urlid':
                    urlid = df[s].tolist()
                if df[s].dtype == 'float64':
                    X[s] = Series(df[s].tolist())
                if df[s].dtype == 'int64':
                    X[s] = Series(df[s].tolist())
                if s == 'alchemy_category':
                    categories = df[s]

                    #print '%s : %s' % (s, raw[s].dtype)

        X = self.addDistinctColumns(self.distinctCategories, X, df['alchemy_category'])

        X = self.addDistinctColumns(self.distinctIsNews, X, df['is_news'])


        #X.append(self.topicModelBoth.addTopicModel(df['boilerplate']))
        #X.append(self.topicModelTitle.addTopicModel(df['boilerplate']))
        #X.append(self.topicModelBody.addTopicModel(df['boilerplate']))

        #X.append(self.topicModelBoth.addTotalWordCounts(df['boilerplate']))
        #X.append(self.topicModelTitle.addTotalWordCounts(df['boilerplate']))
        #X.append(self.topicModelBody.addTotalWordCounts(df['boilerplate']))
        #X = np.array(X).transpose()
        X = DataFrame(X)
        y = np.array(y)

        return X, y, urlid

    def transformFeatures(self, X):
        print "transforming features..."
        X = self.ch2.transform(X)
        return X

    def fitKBest(self, X, y, numFeatures=10000):
        print 'fitting k best...'
        self.ch2 = SelectKBest(chi2, k=numFeatures)
        self.ch2.fit(X, y)

    def preprocessAndExtract(self, df):
        X_raw, y, urlid = self.preprocess(df)

        #X = DataFrame({'non_markup_alphanum_characters': X_raw['non_markup_alphanum_characters'],
        #               'recreation':X_raw['recreation'],
        #               'business':X_raw['business'],
        #               'alchemy_category_score': X_raw['alchemy_category_score'],
        #               'frameTagRatio': X_raw['frameTagRatio'],
        #               'linkwordscore': X_raw['linkwordscore']})
        X = DataFrame({'recreation':X_raw['recreation']})


        """
        26  non_markup_alphanum_characters    0.409029
        30                      recreation    0.139793
        5                         business    0.102396
        2           alchemy_category_score    0.092127
        14                   frameTagRatio    0.043546
        24                   linkwordscore    0.040073
        """

        return X, y, urlid

    def generateSubmission(self, submissionPath, urlid, label):
        submission = DataFrame({'urlid':urlid, 'label': label})
        submission.to_csv(submissionPath, index=False)
        print 'submission written to %s' % submissionPath
        print 'submission has %d rows' % len(label)

    def getRaw(self, path):
        raw = read_csv(path, sep='\t', na_values=['?']).fillna(-5)
        return raw

    def writeReport(self, df, y_test, y_predicted, path):
        f = open(path, 'w')
        body = []
        for row in df['boilerplate']:
            bodyRow = json.loads(row)['body']
            body.append(bodyRow)

        N = 100
        outDf = DataFrame({'url': df['url'][:N], 'y_test': Series(y_test)[:N], 'y_predicted':Series(y_predicted)[:N], 'y_predicted_rounded':Series(np.round(y_predicted))[:N], 'absolute-error': Series(np.abs(y_test-y_predicted))[:N], 'body' : Series(body)[:N]})
        outDf.sort(column='absolute-error', ascending=False).to_csv(path, encoding='utf-16')
