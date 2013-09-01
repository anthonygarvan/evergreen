__author__ = 'root'

import nltk
from nltk.corpus import movie_reviews
from pandas.io.parsers import read_csv
import random
import json
import cPickle as pickle

class BagOfWords:
    def __init__(self, newModel):
        self.modelPath = 'bagOfWords.pkl'
        self.wordFeaturesPath = 'wordFeatures.pkl'

        if newModel:
            self.GetTrainingData()
            #self.GetMovieCorpusData()
            self.classifier = self.TrainClassifier()
            self.GetAccuracy()
            f = open(self.modelPath, 'w')
            pickle.dump(self.classifier, f)
            f = open(self.wordFeaturesPath, 'w')
            pickle.dump(self.word_features, f)
            f.close()
        else:
            print 'loading bag of words model...'
            f = open(self.modelPath,'r')
            self.classifier = pickle.load(f)
            print 'loading word features...'
            f = open(self.wordFeaturesPath,'r')
            self.word_features = pickle.load(f)


    def GetMovieCorpusData(self):
        self.documents = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]
        all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
        self.word_features = all_words.keys()[:2000]

    def GetTrainingBoilerplate(self):
        raw = read_csv('train.tsv', sep='\t', na_values=['?']).fillna(-5)
        self.boilerplate = raw['boilerplate']
        self.labels = raw['label']

    def GetTestBoilerplate(self):
        raw = read_csv('test.tsv', sep='\t', na_values=['?']).fillna(-5)
        self.boilerplate = raw['boilerplate']

    def GetWordsFromBoilerPlate(self, getTitle, getBody, makeNew):
        print 'getting words from boilerplate...'
        picklePath = 'wordList_%d_%s_%s.pkl' % (len(self.boilerplate), getTitle, getBody)

        if makeNew:
            self.words = []
            for row in self.boilerplate:
                rowObject = json.loads(row)
                wordsRow = []
                if 'title' in rowObject and rowObject['title'] and getTitle:
                    wordList = rowObject['title'].lower().split(' ')
                    wordsRow.extend(wordList)
                if 'body' in rowObject and rowObject['body'] and getBody:
                    wordList = rowObject['body'].lower().split(' ')
                    wordsRow.extend(wordList)
                self.words.append(wordsRow)
            f = open(picklePath, 'w')
            pickle.dump(self.words, f)
        else:
            print 'unpickling'
            f = open(picklePath, 'r')
            self.words = pickle.load(f)
            print 'unpickled'
        print 'finished getting words from boilerplate'

    def GetTrainingData(self):
        self.GetTrainingBoilerplate()
        self.GetWordsFromBoilerPlate()
        self.GetTrainingSet()

    def GetTrainingSet(self):
        print 'getting training set...'
        self.documents = []
        for x in xrange(0, len(self.words)):
            self.documents.append((self.words[x], self.labels[x]))
        all_words_list = []
        for wordRow in self.words:
            all_words_list.extend(wordRow)
        all_words = nltk.FreqDist(w.lower() for w in all_words_list)
        self.word_features = all_words.keys()[:2000]
        self.train_set = [(self.document_features(d), c) for (d,c) in self.documents]

    def document_features(self, document):
        document_words = set(document)
        features = {}
        for word in self.word_features:
            features['contains(%s)' % word] = (word in document_words)
        return features

    def TrainClassifier(self):
        classifier = nltk.NaiveBayesClassifier.train(self.train_set)

        return classifier

    def GetFeatureSetFromWords(self):
        self.featureset = []
        for doc in self.words:
            self.featureset.append(self.document_features(doc))

    def Classify(self, X, boilerplate):
        print 'classifying with naive bayes...'
        self.boilerplate = boilerplate
        self.GetWordsFromBoilerPlate(getTitle=True, getBody=True, makeNew=True)
        #self.GetFeatureSetFromWords()
        print 'getting predictions...'
        predictions = [self.classifier.prob_classify(self.document_features(doc)).prob(1) for doc in self.words]
        X.append(predictions)
        return X

    def GetAccuracy(self):
        print nltk.classify.accuracy(self.classifier, self.train_set)

#BagOfWords(False)