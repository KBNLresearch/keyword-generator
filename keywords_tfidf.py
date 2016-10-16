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


# Generate keywords
def generate_keywords(tfidf_scores, num_keywords):
    print("Generating keywords...")
    keywords = {}

    # Sum of scores for token in all documents
    for doc in tfidf_scores:
        for tuple in doc:
            key = tuple[0]
            score = tuple[1]
            if key in keywords:
                keywords[key] += score
            else:
                keywords[key] = score

    # Sort keywords by highest score
    sorted_keywords = sorted(keywords.items(), key=operator.itemgetter(1), reverse=True)

    # Return only requested number of keywords
    sorted_keywords = sorted_keywords[:num_keywords]
    return sorted_keywords


def print_keywords(keywords, dictionary):
    i = 1
    for k in keywords:
        print "(%i) %s [%s]" % (i, dictionary.get(k[0]), k[1])
        i += 1


def export_keywords(keywords, dictionary):
    filename = int(time.time())
    f = open("data" + os.sep + "keywords" + os.sep + str(filename) + ".txt", "w+")
    keywords = [dictionary.get(k[0]) for k in keywords]
    f.write("\n".join(keywords) + "\n\n")
    f.write(" ".join(keywords) + "\n\n")
    f.write(" OR ".join(keywords) + "\n")
    f.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", required=False, help="the number of keywords")
    parser.add_argument("-d", required=False, help="document length")
    args = parser.parse_args()

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

    doc_folder = "data" + os.sep + "documents"
    stop_folder = "data" + os.sep + "stop_words"

    c = cp.MyCorpus(doc_folder, stop_folder, doc_length)
    corpus, dictionary = c.load()

    tfidf = gensim.models.TfidfModel(corpus)
    tfidf_scores = tfidf[corpus]

    keywords = generate_keywords(tfidf_scores, num_keywords)
    print("Keywords generated:")
    print_keywords(keywords, dictionary)
    export_keywords(keywords, dictionary)


if __name__ == "__main__":
    main()
