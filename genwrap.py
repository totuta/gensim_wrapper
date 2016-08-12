#!usr/bin/python
# -*- coding: utf-8 -*-

'''
filename: genwrap.py
description: a python wrapper for gensim APIs
version : 0.1 (basic funciton. NO support for generator yet)
author: Wonchang Chung
date: 08/12/2016
'''

import os
import sys
import math
import time

import gensim
from gensim import corpora

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
    parser.add_argument('--mode', type=str, default='lda',
                        help='choose mode : lda/lsi')
    
    args = parser.parse_args()
    print "arguments parsing done."

    # printout all arguments
    print "----------------------------------------"
    print "LIST OF ARGUMENTS"
    for arg in vars(args):
        print arg, ":", getattr(args, arg)
    print "----------------------------------------"

    return args


def make_dictionary(raw_file):

	stoplist = stopwords.words('english')

	try:
		dictionary = corpora.Dictionary(line.lower().split() for line in read_large_file(open(raw_file)))
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

		return corpus

	except (IOError, OSError):
		print("Error opening / processing file")
	

def main():

	args = get_arguments()

	INPUT_FILE = 'data/stimuli_384sentences_dereferencedpronouns.txt'

	dictionary = make_dictionary(INPUT_FILE)
	corpus     = make_corpus(INPUT_FILE, dictionary)

	# TODO : add any other possible options

	if args.mode == 'lda':
		lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=100, update_every=1, chunksize=10000, passes=1)
		print lda.print_topics(5)
	elif args.mode == 'lsi':
		lsi = gensim.models.lsimodel.LsiModel(corpus=corpus, id2word=dictionary, num_topics=400)
		print lsi.print_topics(5)
	else:
		raise ValueError('choose proper mode: lda/lsi')

	


if __name__ == '__main__' : main()