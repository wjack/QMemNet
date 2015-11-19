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
    x_train_ptb, label_train_ptb = file_to_dataset(train_path, N, word_to_index)
    x_dev_ptb, label_dev_ptb = file_to_dataset(dev_path, N, word_to_index)
    x_test_ptb, label_test_ptb = file_to_dataset(test_path, N, word_to_index)

    return x_train_ptb, x_dev_ptb, x_test_ptb, label_train_ptb, label_dev_ptb, label_test_ptb, word_to_index, index_to_word


def split(input_string):
    return input_string.split()


def file_to_dataset(filepath, N, word_to_index):
    #convert file to dataset, returns X and Y tensors of integer indexes describing the N words (X) leading up to (Y)
    f = open(filepath)
    lines = f.readlines()
    X = []
    Y = []
    for line in lines:
        words =  line.split()
        padding = (N-1)*['<pre>']

        #remove /n at the end, add padding to front
        words = padding + words[:-1]
        for i in range(0, len(words) - N):
            x = []
            y = word_to_index[words[i+N]]
            x_words = words[i:i+N]
            for word in x_words:
                x.append(word_to_index[word])
            X.append(x)
            Y.append(y)

    return tf.convert_to_tensor(X), tf.convert_to_tensor(Y)
#x_train_ptb, x_dev_ptb, x_test_ptb, label_train_ptb, label_dev_ptb, label_test_ptb, word_to_index, index_to_word = build_datasets('parser_test/train.txt', 'parser_test/dev.txt', 'parser_test/test.txt', 2)
