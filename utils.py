#!usr/bin/python
# -*- coding: utf-8 -*-

'''
filename: utils.py
description: utils for the wrapper for gensim APIs
version : 0.1 read_large_file
author: Wonchang Chung
date: 08/12/2016
'''

# import os
# import sys
# import math
# import time

# import gensim
# from gensim import corpora

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# import string
# import argparse

# import numpy as np

def read_large_file(file_object):
	# generator for reading a big file
	while True:
		data = file_object.readline()
		if not data:
			break
		yield data
