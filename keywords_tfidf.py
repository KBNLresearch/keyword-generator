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


# Generate keywords
def generate_keywords(tfidf_scores, num_keywords):
    print('Generating keywords...')
    keywords = {}

    # Sum of scores for token in all documents
    for doc in tfidf_scores:
        for t in doc:
            key = t[0]
            score = t[1]
            if key in keywords:
                keywords[key] += score
            else:
                keywords[key] = score

    # Sort keywords by highest score
    sorted_keywords = sorted(keywords.items(), key=operator.itemgetter(1),
            reverse=True)

    return sorted_keywords[:num_keywords]


def print_keywords(keywords, dictionary):
    print('Keywords generated:')
    for i, k in enumerate(keywords):
        print('(%i) %s [%s]' % (i + 1, dictionary.get(k[0]), k[1]))


def save_keywords(keywords, dictionary):
    timestamp = int(time.time())
    with open('data' + os.sep + 'results' + os.sep + str(timestamp) +
            '_keywords' + '.csv', 'wb') as f:
        csv_writer = csv.writer(f, delimiter='\t')
        for k in keywords:
            csv_writer.writerow([dictionary.get(k[0]).encode('utf-8'),
                    str(k[1])])


if __name__ == '__main__':
    if sys.stdout.encoding != 'UTF-8':
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout, 'strict')

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', required=False, type=int, default=10,
            help='number of keywords')
    parser.add_argument('-d', required=False, type=int, default=0,
            help='document length')
    args = parser.parse_args()

    num_keywords, doc_length = vars(args)['k'], vars(args)['d']

    doc_folder = 'data' + os.sep + 'documents'
    stop_folder = 'data' + os.sep + 'stop_words'

    corpus, dictionary = cp.MyCorpus(doc_folder, stop_folder, doc_length).load()
    tfidf = gensim.models.TfidfModel(corpus)
    tfidf_scores = tfidf[corpus]
    keywords = generate_keywords(tfidf_scores, num_keywords)

    print_keywords(keywords, dictionary)
    save_keywords(keywords, dictionary)
