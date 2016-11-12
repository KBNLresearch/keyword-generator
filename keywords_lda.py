#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#
# Keyword Generator
#
# Copyright (C) 2015 Juliette Lonij, Koninklijke Bibliotheek -
# National Library of the Netherlands
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


import argparse
import codecs
import corpus as cp
import csv
import gensim
import math
import operator
import os
import sys
import time

from builtins import input


# Exclude topics
def exclude_topics(topics):
    print_topics(topics)
    inp = input('Enter topics to exclude, separated by commas: ')
    if inp == '':
        return topics
    excl_topics = [int(i.strip()) for i in inp.split(',')]
    excl_topics.sort(reverse=True)
    for i in excl_topics:
        del topics[i-1]
    return topics


# Print topics
def print_topics(topics):
    print('Topics generated:')
    for i, topic in enumerate(topics):
        print('(' + str(i + 1) + ') ' + ', '.join([t[1] for t in topic]))


# Generate keywords
def generate_keywords(corpus, dictionary, topics, num_keywords):
    print('Generating keywords...')
    keywords = {}

    # Sum of probabilities for token in all topics
    for topic in topics:
        for t in topic:
            token = t[1]
            pr = t[0]
            if token in keywords:
                keywords[token] += pr
            else:
                keywords[token] = pr

    # Probability for each token multiplied by token frequency
    matrix = gensim.matutils.corpus2csc(corpus)
    for token, pr in keywords.items():
        for d in dictionary.items():
            if d[1] == token:
                token_index = d[0]
                break
        token_row = matrix.getrow(token_index)
        token_freq = token_row.sum(1).item()
        keywords[token] = pr * math.log(token_freq)

    # Sort keywords by highest score
    sorted_keywords = sorted(keywords.items(), key=operator.itemgetter(1),
            reverse=True)

    return sorted_keywords[:num_keywords]


def print_keywords(keywords):
    print('Keywords generated:')
    for i, k in enumerate(keywords):
        print('(%i) %s [%s]' % (i + 1, k[0], k[1]))


def save_keywords(keywords):
    timestamp = int(time.time())
    with open('data' + os.sep + 'results' + os.sep + str(timestamp) +
            '_keywords' + '.csv', 'w+') as f:
        csv_writer = csv.writer(f, delimiter='\t')
        for k in keywords:
            csv_writer.writerow([k[0].encode('utf-8'), str(k[1])])


def save_topics(topics):
    timestamp = int(time.time())
    with open('data' + os.sep + 'results' + os.sep + str(timestamp) +
            '_topics' + '.csv', 'w+') as f:
        csv_writer = csv.writer(f, delimiter='\t')
        for topic in topics:
            csv_writer.writerow([t[1].encode('utf-8') for t in topic])
            csv_writer.writerow([str(t[0]) for t in topic])


def save_distributions(distributions):
    timestamp = int(time.time())
    with open('data' + os.sep + 'results' + os.sep + str(timestamp) +
            '_distributions' + '.csv', 'w+') as f:
        csv_writer = csv.writer(f, delimiter='\t')
        csv_writer.writerow(['Document'] + ['Topic ' + str(i + 1) for i in
                range(len(distributions[0]))])
        for i, dist in enumerate(distributions):
            csv_writer.writerow([str(i)] + ['{0:.5f}'.format(t[1]) for t in
                    dist])


if __name__ == '__main__':
    if sys.stdout.encoding != 'UTF-8':
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout, 'strict')

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', required=False, help='number of topics')
    parser.add_argument('-w', required=False, help='number of words per topic')
    parser.add_argument('-k', required=False, help='number of keywords')
    parser.add_argument('-d', required=False, help='document length')
    parser.add_argument('-m', required=False, help='Mallet path')
    args = parser.parse_args()

    num_topics = int(vars(args)['t']) if vars(args)['t'] else 10
    num_words = int(vars(args)['w']) if vars(args)['w'] else 10
    num_keywords = int(vars(args)['k']) if vars(args)['k'] else 10
    doc_length = int(vars(args)['d']) if vars(args)['d'] else 0
    mallet_path = vars(args)['m']

    doc_folder = 'data' + os.sep + 'documents'
    stop_folder = 'data' + os.sep + 'stop_words'

    corpus, dictionary = cp.MyCorpus(doc_folder, stop_folder, doc_length).load()

    if mallet_path:
        print('Generating model with Mallet LDA ...')
        lda = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus,
                id2word=dictionary, num_topics=num_topics)
        topics = lda.show_topics(num_topics=num_topics, num_words=num_words,
                formatted=False)
        distributions = [dist for dist in lda.load_document_topics()]
    else:
        print('Generating model with Gensim LDA ...')
        lda = gensim.models.LdaModel(corpus, id2word=dictionary,
                num_topics=num_topics, alpha='auto', chunksize=1, eval_every=1)
        gensim_topics = [t[1] for t in lda.show_topics(num_topics=num_topics,
                num_words=num_words, formatted=False)]
        topics = [[(i[1], i[0]) for i in t] for t in gensim_topics]
        distributions = []
        matrix = gensim.matutils.corpus2csc(corpus)
        for i in range(matrix.get_shape()[1]):
            bow = gensim.matutils.scipy2sparse(matrix.getcol(i).transpose())
            distributions.append(lda.get_document_topics(bow, 0))

    topics = exclude_topics(topics)
    keywords = generate_keywords(corpus, dictionary, topics, num_keywords)

    print_keywords(keywords)
    save_keywords(keywords)
    save_topics(topics)
    save_distributions(distributions)
