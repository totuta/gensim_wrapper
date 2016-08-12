#!usr/bin/python
# -*- coding: utf-8 -*-

'''
filename: genwrap.py
description: a python wrapper for gensim APIs
version : 0.2 (basic funciton with support for generator)
author: Wonchang Chung
date: 08/12/2016
'''

import os
import sys
import math
import time

import gensim
from gensim import corpora, models, utils

import nltk
from nltk.corpus import stopwords

import string
import argparse

import numpy as np

from six import iteritems

from utils import read_large_file


def get_arguments():

    print "----------------------------------------"
    print "parsing arguments..."
    parser = argparse.ArgumentParser()

    # choose mode
    parser.add_argument('--model', type=str, default='lda',
                        help='choose mode : lda/lsi')

    # choose vectorization
    parser.add_argument('--vec', type=str, default='bow',
                        help='choose vectorization : bow/tfidf')
    
    args = parser.parse_args()
    print "arguments parsing done."

    # printout all arguments
    print "----------------------------------------"
    print "LIST OF ARGUMENTS"
    for arg in vars(args):
        print arg, ":", getattr(args, arg)

    return args


def make_dictionary(raw_file):

	stoplist = stopwords.words('english')

	try:
		dictionary = corpora.Dictionary(map(lambda x : x.split('/')[0], utils.lemmatize(line)) for line in read_large_file(open(raw_file)))
		# dictionary = corpora.Dictionary(utils.lemmatize(line) for line in read_large_file(open(raw_file)))
		# dictionary = corpora.Dictionary(line.lower().split() for line in read_large_file(open(raw_file)))
		# dictionary = corpora.Dictionary(line.lower().split() for line in open(raw_file))

		# remove stopwords and once-words
		stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
		once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
		dictionary.filter_tokens(stop_ids + once_ids)
		dictionary.compactify()

		return dictionary

	except (IOError, OSError):
		print("Error opening / processing file")


def make_corpus(raw_file, dictionary):

	try:
		corpus = [dictionary.doc2bow(text.lower().split()) for text in read_large_file(open(raw_file))]
		# corpus = [dictionary.doc2bow(text.lower().split()) for text in open(raw_file)]

		# when using tf-idf
		tfidf = models.TfidfModel(corpus)
		corpus = tfidf[corpus]

		return corpus, tfidf

	except (IOError, OSError):
		print("Error opening / processing file")


def look_up(model, dictionary, tfidf, doc):
	# topic lookup function

	doc_value = model[tfidf[dictionary.doc2bow(doc.lower().split())]]

	return doc_value


def word_to_vec(sentences, size):
	# TODO : implement word2vec wrapper

	# sentences = list of list of words
	return w2v = models.word2vec.Word2Vec(sentences, size=size, window=5, min_count=5, workers=4)


def doc_to_vec(documents, size):
	# TODO : implement doc2vec wrapper

	# documents = ???
	return d2v = models.doc2vec.Doc2Vec(documents, size=size, window=8, min_count=5, workers=4)
	
	

def main():

	args = get_arguments()

	INPUT_FILE = 'data/sample_corpus.txt'

	print "----------------------------------------"
	print "making dictionary.."
	dictionary    = make_dictionary(INPUT_FILE)
	print "dictionary done."

	print "----------------------------------------"
	print "making corpus.."
	corpus, tfidf = make_corpus(INPUT_FILE, dictionary)
	print "corpus done."

	# TODO : add any other possible options

	if args.model == 'lda':
		print "----------------------------------------"
		print "doing LDA.."
		# lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=100, update_every=1, chunksize=10000, passes=1)

		# load/save
		lda.save('model/model.lda')
		dictionary.save('model/dict.lda')
		tfidf.save('model/tfidf.lda')
		# lda = models.LdaModel.load('model/model.lda')
		# dictionary = corpora.Dictionary.load('model/dict.lda')
		# tfidf = models.TfidfModel.load('model/tfidf.lda')

		topics = lda.print_topics(10)
		for topic in topics:
			print dictionary.id2token[topic[0]], topic[1]

		value = look_up(lda, dictionary, tfidf, "I like scissors and hammer and control and earthquake")

		for topic in value:
			print dictionary.id2token[topic[0]], topic[1]

		print "LDA done."

	elif args.model == 'lsi':
		print "----------------------------------------"
		print "doing LSI.."
		lsi = gensim.models.lsimodel.LsiModel(corpus=corpus, id2word=dictionary, num_topics=400)

		# load/save
		lsi.save('model/model.lsi')
		dictionary.save('model/dict.lda')
		tfidf.save('model/tfidf.lda')
		# lsi = models.LsiModel.load('model/model.lsi')
		# dictionary = corpora.Dictionary.load('model/dict.lda')
		# tfidf = models.TfidfModel.load('model/tfidf.lda')

		topics = lsi.print_topics(10)
		for topic in topics:
			print dictionary.id2token[topic[0]], topic[1]

		print "LSI done."

	else:
		raise ValueError('choose proper mode: lda/lsi')

if __name__ == '__main__' : main()