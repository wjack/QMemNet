# pylint: disable=unused-import,g-bad-import-order
"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
import sys
import time
import numpy as np
import tensorflow as tf
import nltk
from sklearn.feature_extraction.text import CountVectorizer

def build_datasets(train_path, dev_path, test_path, N):
    #build datasets considering N previous words
    train = open(train_path).read()
    dev= open(dev_path).read()
    test = open(test_path).read()

    all_text = train+dev+test + '<pre>'
    vectorizer = CountVectorizer(min_df = 1, analyzer = 'word', tokenizer = split)

    vectorizer.fit_transform(all_text.split())

    word_to_index = vectorizer.vocabulary_
    index_to_word = vectorizer.get_feature_names()
    print('Parsing train data...')
    x_train_ptb, label_train_ptb = file_to_dataset(train_path, N, vectorizer)
    print('Parsing dev data...')
    x_dev_ptb, label_dev_ptb = file_to_dataset(dev_path, N, vectorizer)
    print('Parsing test data...')
    x_test_ptb, label_test_ptb = file_to_dataset(test_path, N, vectorizer)

    return x_train_ptb, x_dev_ptb, x_test_ptb, label_train_ptb, label_dev_ptb, label_test_ptb, word_to_index, index_to_word


def split(input_string):
    return input_string.split()


def file_to_dataset(filepath, N, vectorizer):
    #convert file to dataset, returns X and Y tensors of integer indexes describing the N words (X) leading up to (Y)
    f = open(filepath)
    lines = f.readlines()
    X = []
    Y = []
    line_index = 0
    for line in lines:
        if line_index % 1000 == 0:
            print(str(line_index) +'/' len(lines) + ' lines read...')
        line_index +=1
        words =  line.split()
        padding = (N-1)*['<pre>']

        #remove /n at the end, add padding to front
        words = padding + words[:-1]
        for i in range(0, len(words) - N):
            x = []
            y = vectorizer.transform(words[i+N]).toarray()
            x_words = words[i:i+N]
            for word in x_words:
                x.append(vectorizer.transform(word).toarray())
            X.append(x)
            Y.append(y)

    return tf.convert_to_tensor(X), tf.convert_to_tensor(Y)
x_train_ptb, x_dev_ptb, x_test_ptb, label_train_ptb, label_dev_ptb, label_test_ptb, word_to_index, index_to_word = build_datasets('ptb_data/ptb.train.txt', 'ptb_data/ptb.valid.txt', 'ptb_data/ptb.test.txt', 2)
