from __future__ import print_function
from nltk.chat.util import Chat, reflections

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names

from bottle import Bottle, run, route, static_file, request, response
from pymongo import MongoClient
from bson.json_util import dumps
from string import Template
import json
import pymongo
import requests
import datetime
import time
import math
import hashlib
import os



app = Bottle(__name__)

#################### NLTK Chat #######################3

pairs = [
	[
		r'get me tickets for (.*)',
		['{"intent":"event_search_by_name", "entities": [{"name": "event_name", "value": "%1"}]}']
	],
	[
		r'register me for (.*)',
		['{"intent":"event_search_by_name", "entities": [{"name": "event_name", "value": "%1"}]}']
	],
	[
		r'hi',
		['hello', 'hey',]
	],
	[
		r'hey',
		['hi', 'hello']
	],
	[
		r'bye',
		['goodbye', 'see you soon :)']
	],
	[
		r'(.*)',
		["%1"]
	]
]

chat = Chat(pairs, reflections)

###################### Sentiment Analysis ###################33333

def word_feats(words):
    return dict([(word, True) for word in words])

def get_sentiment(msg):
 
	positive_vocab = [ 'awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice', 'great', ':)', ':-)', ':-D' ]
	negative_vocab = [ 'bad', 'terrible','useless', 'hate', ':(' ]
	neutral_vocab = [ 'movie','the','sound','was','is','actors','did','know','words','not' ]
	 
	positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
	negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]
	neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]
	 
	train_set = negative_features + positive_features + neutral_features
	 
	classifier = NaiveBayesClassifier.train(train_set) 

	neg = 0
	pos = 0

	sentence = msg
	sentence = sentence.lower()
	words = sentence.split(' ')

	for word in words:
	    classResult = classifier.classify( word_feats(word))

	    if classResult == 'neg':
	        neg = neg + 1
	    if classResult == 'pos':
	        pos = pos + 1

	return {'pos': str(float(pos)/len(words)), 'neg': str(float(neg)/len(words))}

@app.route('/')
def root():
	return 'Testing chat server'

@app.route('/chat/<msg>')
def chat_user(msg):

	sentiment_data = get_sentiment(msg)

	return {'response': chat.respond(msg), 'sentiment': sentiment_data}