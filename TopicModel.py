__author__ = 'root'

from pandas.io.parsers import read_csv
import random
import json
import cPickle as pickle
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score


class TopicModel:
    def __init__(self, getTitle, getBody):
        self.getTitle = getTitle
        self.getBody = getBody

    def getRaw(self, path):
        raw = read_csv(path, sep='\t', na_values=['?']).fillna(-5)
        return raw

    def getColumns(self, raw):
        boilerplate = raw['boilerplate']
        if 'label' in raw:
            labels = raw['label']
            return boilerplate, labels
        return boilerplate

    def getDocs(self, boilerplate):
        docs = []
        for row in boilerplate:
            rowObject = json.loads(row)
            doc = ''
            if 'title' in rowObject and rowObject['title'] and self.getTitle:
                doc += rowObject['title'].lower()
            if 'body' in rowObject and rowObject['body'] and self.getBody:
                doc += rowObject['body'].lower()
            docs.append(doc)
        return docs

    def vectorize(self, docs):
        vectorizer = HashingVectorizer(stop_words='english')
        return vectorizer.transform(docs)

    def trainFromDataFrame(self, df):
        boilerplate, y = self.getColumns(df)
        docs = self.getDocs(boilerplate)
        X = self.vectorize(docs)
        self.fit(X, y)

    def fit(self, X, y):
        self.model = BernoulliNB()
        self.model.fit(X, y)

    def addTopicModel(self, X, boilerplate):
        docs = self.getDocs(boilerplate)
        X_extracted = self.vectorize(docs)
        y_predicted = self.predict(X_extracted)
        X.append(y_predicted)
        return X

    def predict(self, X):
        return self.model.predict_proba(X)[:,1]

    def benchmark(self, testSize):
        raw = self.getRaw('train.tsv')
        boilerplate, y = self.getColumns(raw)
        docs = self.getDocs(boilerplate)
        X = self.vectorize(docs)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize)
        self.fit(X_train, y_train)
        y_predicted = self.predict(X_test)
        print 'AUC Score: %f' % roc_auc_score(y_test, y_predicted)

T = TopicModel(getTitle=True, getBody=True)
T.benchmark(.25)

