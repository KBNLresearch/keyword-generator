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
import gensim
import math
import operator
import os
import pprint
import sys
import time

from builtins import input


# Exclude topics
def exclude_topics(topics):
    print("Topics generated:")
    print_topics(topics)
    inp = input("Enter topics to exclude, separated by commas: ")
    inp = inp.replace(" ", "")
    if inp == "":
        return topics
    excl_topics = inp.split(",")
    for i in range(len(excl_topics)):
        excl_topics[i] = int(excl_topics[i])
    excl_topics.sort(reverse=True)
    for i in excl_topics:
        del topics[i-1]
    return topics


# Print topics
def print_topics(topics):
    i = 1
    for topic in topics:
        sys.stdout.write("(" + str(i) + ")")
        for token_tuple in topic:
            sys.stdout.write(" " + token_tuple[1])
        sys.stdout.write("\n")
        i += 1


# Generate keywords
def generate_keywords(corpus, dictionary, topics, num_keywords):
    print("Generating keywords...")
    keywords = {}

    # Sum of probabilities for token in all topics
    for topic in topics:
        for token_tuple in topic:
            token = token_tuple[1]
            pr = token_tuple[0]
            if token in keywords:
                keywords[token] += pr
            else:
                keywords[token] = pr

    # Probability for each token multiplied by token frequency
    matrix = gensim.matutils.corpus2csc(corpus)
    for token, pr in keywords.items():
        for dict_tuple in dictionary.iteritems():
            if dict_tuple[1] == token:
                token_index = dict_tuple[0]
                break
        token_row = matrix.getrow(token_index)
        token_freq = token_row.sum(1).item()
        keywords[token] = pr * math.log(token_freq)

    # Sort keywords by highest score
    sorted_keywords = sorted(keywords.items(), key=operator.itemgetter(1), reverse=True)

    # Return only requested number of keywords
    sorted_keywords = sorted_keywords[:num_keywords]
    return sorted_keywords


def print_keywords(keywords):
    i = 1
    for k in keywords:
        print("(%i) %s [%s]" % (i, k[0], k[1]))
        i += 1


def save_keywords(keywords):
    timestamp = int(time.time())
    with open("data" + os.sep + "results" + os.sep + str(timestamp) + "_keywords" + ".csv", "w+") as f:
        for k in keywords:
            f.write(k[0] + "\t")
            f.write(str(k[1]) + "\n")


def save_topics(topics):
    timestamp = int(time.time())
    with open("data" + os.sep + "results" + os.sep + str(timestamp) + "_topics" + ".csv", "w+") as f:
        for i, topic in enumerate(topics):
            f.write(str(i))
            for t in topic:
                f.write("\t" + t[1])
            f.write("\n")
            for t in topic:
                f.write("\t" + str(t[0]))
            f.write("\n")


def save_distributions(distributions):
    timestamp = int(time.time())
    with open("data" + os.sep + "results" + os.sep + str(timestamp) + "_distributions" + ".csv", "w+") as f:
        f.write("Document")
        for i in range(len(distributions[0])):
            f.write("\t" + "Topic " + str(i + 1))
        f.write("\n")
        for i, dist in enumerate(distributions):
            f.write(str(i))
            for topic in dist:
                f.write("\t" + "{0:.5f}".format(topic[1]))
            f.write("\n")


def main():
    if sys.stdout.encoding != 'UTF-8':
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout, 'strict')

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", required=False, help="the number of topics")
    parser.add_argument("-w", required=False, help="the number of words per topic")
    parser.add_argument("-k", required=False, help="the number of keywords")
    parser.add_argument("-d", required=False, help="document length")
    parser.add_argument("-m", required=False, help="Mallet path")
    args = parser.parse_args()

    num_topics = vars(args)["t"]
    if not num_topics:
        num_topics = 10
    else:
        num_topics = int(num_topics)

    num_words = vars(args)["w"]
    if not num_words:
        num_words = 10
    else:
        num_words = int(num_words)

    num_keywords = vars(args)["k"]
    if not num_keywords:
        num_keywords = 10
    else:
        num_keywords = int(num_keywords)

    doc_length = vars(args)["d"]
    if not doc_length:
        doc_length = 0
    else:
        doc_length = int(doc_length)

    mallet_path = vars(args)["m"]

    doc_folder = "data" + os.sep + "documents"
    stop_folder = "data" + os.sep + "stop_words"

    c = cp.MyCorpus(doc_folder, stop_folder, doc_length)
    corpus, dictionary = c.load()

    if mallet_path:
        print("Generating model with Mallet LDA ...")
        lda = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, id2word=dictionary, num_topics=num_topics)
        topics = lda.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
        distributions = [dist for dist in lda.load_document_topics()]
    else:
        print("Generating model with Gensim LDA ...")
        lda = gensim.models.LdaModel(corpus, id2word=dictionary, num_topics=num_topics, alpha='auto', chunksize=1, eval_every=1)
        gensim_topics = [t[1] for t in lda.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)]
        topics = [[(i[1], i[0]) for i in t] for t in gensim_topics]
        distributions = []
        matrix = gensim.matutils.corpus2csc(corpus)
        for i in range(matrix.get_shape()[1]):
            bow = gensim.matutils.scipy2sparse(matrix.getcol(i).transpose())
            distributions.append(lda.get_document_topics(bow, 0))

    topics = exclude_topics(topics)

    keywords = generate_keywords(corpus, dictionary, topics, num_keywords)
    print("Keywords generated:")
    print_keywords(keywords)

    save_keywords(keywords)
    save_topics(topics)
    save_distributions(distributions)


if __name__ == "__main__":
    main()
