__author__ = 'root'

from pandas.io.parsers import read_csv
import random
import json
import cPickle as pickle
from sklearn.feature_extraction.text import HashingVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVR, SVC, LinearSVC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, AdaBoostRegressor
import pylab as P
import numpy as np
import random
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.ensemble import AdaBoostRegressor
from time import time
from sklearn.grid_search import GridSearchCV
from SparseAdaBoost import SparseAdaBoost
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix
from scipy.sparse import hstack
from Preprocessor import Preprocessor
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn.base import BaseEstimator
from numpy import linspace, logspace
from sklearn.cross_validation import cross_val_score

class TopicModel(BaseEstimator):
    def __init__(self, C=.5, numFeatures=5000, n_estimators=50, gamma=1):
        self.C = C
        self.numFeatures = numFeatures
        self.n_estimators = n_estimators
        self.gamma = gamma
        self.alpha=0.05

    def fit(self,X, y):
        self.clf = LogisticRegression(penalty='l2', dual=True, tol=0.0001,
                                      C=self.C, fit_intercept=True, intercept_scaling=1.0,
                                      class_weight=None, random_state=None)
        #self.ch2 = SelectKBest(chi2, k=self.numFeatures)
        #self.ch2.fit(X, y)
        #X = self.ch2.transform(X)
        self.clf.fit(X,y)

    def predict(self, X):
        #X = self.ch2.transform(X)
        return self.binResults(self.clf.predict_proba(X)[:, 1], self.alpha)
        #return self.clf.predict(X)

    def decision_function(self, X):
        #X = self.ch2.transform(X)
        return self.binResults(self.clf.predict_proba(X)[:,1], self.alpha)

    def predict_proba(self,X):
        #X = self.ch2.transform(X)
        #return self.clf.predict_proba_lr(X)
        return self.clf.predict_proba(X)

    def binResults(self, y_predicted, alpha):
        out = []
        for y in y_predicted:
            if y < alpha:
                out.append(0)
            elif y > (1-alpha):
                out.append(1)
            else:
                out.append(y)
        return out

class TopicModelHarness:
    def __init__(self, getTitle, getBody, getUrl):
        self.getTitle = getTitle
        self.getBody = getBody
        self.getUrl = getUrl

    def getRaw(self, path):
        raw = read_csv(path, sep='\t', na_values=['?']).fillna(-5)
        return raw

    def getColumns(self, raw):
        boilerplate = raw['boilerplate']
        urlid = raw['urlid']
        if 'label' in raw:
            labels = raw['label']
            return boilerplate, labels, urlid
        return boilerplate, urlid

    def getDocs(self, boilerplate):
        docs = []
        for row in boilerplate:
            rowObject = json.loads(row)
            doc = ''
            if 'title' in rowObject and rowObject['title'] and self.getTitle:
                doc += rowObject['title']
            if 'body' in rowObject and rowObject['body'] and self.getBody:
                doc += ' ' + rowObject['body']
            if 'url' in rowObject and rowObject['url'] and self.getUrl:
                doc += ' ' + rowObject['url']
            docs.append(doc)
        return docs

    def tag(self, str, tag):
        strList = str.split(' ')
        newstr = ''
        for s in strList:
            if s.lower() not in ENGLISH_STOP_WORDS:
                newstr += tag + '__' + s + ' ' + s + ' '
        return newstr

    def preprocessDocs(self, docs):
        preprocessed_docs = []
        for doc in docs:
            punctuation = [',','.', ';', '!', '?', ':']
            for p in punctuation:
                doc = doc.replace(p, ' ' + p + ' ')

            doc = doc.lower()
            preprocessed_docs.append(doc)

        return preprocessed_docs

    def replaceRareWords(self, docs, rareWords):
        processed_docs = []
        for doc in docs:
            punctuation = [',','.', ';', '!', '?', ':']
            for p in punctuation:
                doc = doc.replace(p, ' ' + p + ' ')

            docList = [(self.classifyRareWord(d) if (d in rareWords) else d) for d in doc.split(' ')]
            doc = ''

            for d in docList:
                doc += ' %s ' % d

            processed_docs.append(doc)

        return processed_docs

    def removeStopWords(self, docs, stopWords):
        processed_docs = []
        for doc in docs:
            punctuation = [',','.', ';', '!', '?', ':']
            for p in punctuation:
                doc = doc.replace(p, ' ' + p + ' ')

            docList = [('' if (d in stopWords) else d) for d in doc.split(' ')]
            doc = ''

            for d in docList:
                doc += ' %s ' % d

            processed_docs.append(doc)

        return processed_docs

    def getStopWords(self, freqs, threshold):
        stopWords = set()
        for token in freqs:
            if freqs[token] > threshold:
                stopWords.add(token)
        return stopWords

    def classifyRareWord(self, word):
        if word.find('-') >= 0:
            words = word.split('-')
            out = ''
            for w in words:
                out += w + ' '
            return out
        if word.isdigit():
            return '__ISDIGIT__'
        return '__RARE__'

    def countTokens(self, docs, y):
        freqCounts = {}
        freqCountsWithClass = {}
        freqsByClass = {}
        #for doc in docs, yi in y:
        for i in xrange(0,len(docs)):
            doc = docs[i]
            tokenList = doc.split(' ')
            for token in tokenList:
                if token in freqCounts:
                    freqCounts[token] += 1
                    if y[i] == 1:
                        freqCountsWithClass[token] += 1
                    else:
                        freqCountsWithClass[token] += -1
                else:
                    freqCounts[token] = 1
                    if y[i] == 1:
                        freqCountsWithClass[token] = 1
                    else:
                        freqCountsWithClass[token] = -1
        for token in freqCountsWithClass:
            freqsByClass[token] = float(freqCountsWithClass[token])/float(freqCounts[token])
        return freqCounts, freqsByClass

    def getAmbiguousTokens(self, freqsByClass):
        ambiguousTokens = set()
        for token in freqsByClass:
            if np.abs(freqsByClass[token]) < 0.1:
                ambiguousTokens.add(token)
        return ambiguousTokens


    def getFreqs(self, freqCounts):
        # get total token count
        totalTokenCount = 0
        for token in freqCounts:
            totalTokenCount += freqCounts[token]

        freqs = {}
        for token in freqCounts:
            freqs[token] = float(freqCounts[token]) / float(totalTokenCount)
        return freqs

    def getRareWords(self, freqCounts):
        #freqCounts = self.countTokens(docs)
        rareWords = set()
        for token in freqCounts:
            if freqCounts[token] <= 1:
                rareWords.add(token)
        return rareWords

    def vectorize(self, docs, stopWords, y):
        print "vectorizing..."
        #vectorizer = HashingVectorizer(stop_words='english', non_negative=True)
        #vectorizer = HashingVectorizer(stop_words=stopWords, non_negative=True, norm='l2')
        if y is not None:
            self.vectorizer =TfidfVectorizer(min_df=3,  max_features=None, strip_accents='unicode',
                                    analyzer='word',token_pattern=r'\w{1,}',ngram_range=(1, 2), use_idf=1,smooth_idf=1,
                                    sublinear_tf=1)
            self.vectorizer.fit(docs)

        vectorizedDocs = self.vectorizer.transform(docs)

        #print vectorizedDocs
        return vectorizedDocs

    def standardizeVecs(self, vectorizedDocs):
        print "standardizing vectors..."
        s = vectorizedDocs
        #s_lil = vectorizedDocs.tolil()

        """
        col_sum = s.sum(axis=0)
        (rows, cols) = s.nonzero()
        s_normalized = lil_matrix(s.shape, dtype='float64')
        for i in xrange(0,len(rows)):
            s_normalized[rows[i], cols[i]] = s[rows[i], cols[i]] / col_sum[0, cols[i]]
            if i%50000==0:
                print i
        return s_normalized.tocsr()
        """
        # standardize
        #means = s.mean(axis=0)

        # initialize mean matrix
        #mean_lil = lil_matrix(s.shape, dtype='float64')

        (rows, cols)  = s.nonzero()
        #for i in xrange(0, len(rows)):
        #    mean_lil[rows[i], cols[i]] = means[0, cols[i]]
        #mean_csr = mean_lil.tocsr()
        #s_zeroMean = (s - mean_csr)
        #s_stdDev = (s_zeroMean.multiply(s_zeroMean)).mean(axis=0)
        norm = (s.multiply(s)).sum(axis=0)
        s_standardized = lil_matrix(s.shape, dtype='float64')
        #print s_stdDev.shape
        print s_standardized.shape
        #print s_zeroMean.shape
        for i in xrange(0,len(rows)):
            #s_standardized[rows[i], cols[i]] = s_zeroMean[rows[i], cols[i]] / s_stdDev[0, cols[i]]
            s_standardized[rows[i], cols[i]] = float(np.abs(s[rows[i], cols[i]])) / np.sqrt(float(norm[0, cols[i]]))
            if i%50000==0:
                print i
        for i in xrange(0,50):
            print s_standardized[i, 0]

        return s_standardized.tocsr()

    def trainFromDataFrame(self, df):
        print "training from data frame..."
        boilerplate, y = self.getColumns(df)
        docs = self.getDocs(boilerplate)
        #rareWords = self.getRareWords(docs)
        docs = self.preprocessDocs(docs)
        X = self.vectorize(docs)
        self.fit(X, y)


    def fit(self, X, y):
        print 'training topic model...'
        #self.model = TopicModel()
        #self.model = LogisticRegression(penalty='l2', dual=True, C=.8)
        #self.model.fit(X,y)

        #params = {'C': linspace(.3, .8, 1), 'numFeatures': linspace(1000, 15000, 5)}
        #params = {'C': linspace(.5, 1, 2), 'numFeatures': linspace(25000, 75000, 2)}
        #print params
        #params = {'C': logspace(-1,4,10), 'gamma':logspace(0,0,1)}
        params = {'C': linspace(.8,1.3,5)}
        clf = TopicModel()
        self.model = GridSearchCV(clf, param_grid=params, scoring='roc_auc', cv=10, verbose=2, n_jobs=4)
        self.model.fit(X, y)

        try:
            print 'Best Params:'
            print self.model.best_params_
            print 'Best Score: '
            print self.model.best_score_
            print self.model.grid_scores_
        except:
            pass

        self.model = self.model.best_estimator_

    def addTopicModel(self, boilerplate):
        docs = self.getDocs(boilerplate)
        X_extracted = self.vectorize(docs)
        #y_predicted = self.predict(X_extracted)
        y_predicted = self.binResults(self.predict(X_extracted), .05)
        return y_predicted

    def addTotalWordCounts(self, boilerplate):
        docs = self.getDocs(boilerplate)
        wordCounts = [(len(doc.split(' '))) for doc in docs]
        return wordCounts


    def predict(self, X):
        return self.model.predict(X)
        #return self.model.predict_proba(X)[:,1]


    def addAlchemyCategories(self, docs, alchemyCategory):
        for i in xrange(0, len(docs)):
            docs[i] = docs[i] + ' __' + str(alchemyCategory[i]) + ' '
        return docs

    def getXy(self, path):
        raw = self.getRaw(path)
        docs = self.getDocs(raw['boilerplate'])
        if 'label' in raw:
            y = raw['label']
            self.vectorize(docs=docs, stopWords='english', y=y)
        else:
            y = None

        docs = self.preprocessDocs(docs)
        X_text = self.vectorize(docs=docs, stopWords='english', y=None)
        self.Pre = Preprocessor()
        X_meta, y, urlid = self.Pre.preprocess(raw)
        #X_meta = np.abs(X_meta)
        #X = hstack([X_meta,X_text])
        X = X_text
        d = {'X': X, 'y':y, 'urlid': urlid}
        return d

    def runModel(self, testSize, debug):
        d = self.getXy('train.tsv')
        if debug:
            X_train, X_test, y_train, y_test = train_test_split(d['X'], d['y'], test_size=testSize, random_state=5)
        else:
            X_train = d['X']
            y_train = d['y']
            d_test = self.getXy('test.tsv')
            X_test = d_test['X']
            urlid = d_test['urlid']

        self.fit(X_train, y_train)
        print "20 Fold CV Score: ", np.mean(cross_val_score(self.model, d['X'], d['y'], cv=10, scoring='roc_auc'))
        y_predicted = self.predict(X_test)
        #y_predicted = self.binResults(y_predicted, 0.05)

        if debug:
            print 'Topic Model AUC Score: %f' % roc_auc_score(y_test, y_predicted)
        else:
            Pre = Preprocessor()
            Pre.generateSubmission('submission_11.csv', urlid, y_predicted)

        P.figure()
        P.hist(y_predicted, bins=100)
        P.show()

start = time()
T = TopicModelHarness(getTitle=True, getBody=True, getUrl=True)
T.runModel(.25, debug=False)
stop = time()
elapsed = stop-start
print 'elapsed time to train topic mode: %f' % elapsed
