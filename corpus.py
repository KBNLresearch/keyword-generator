#!/usr/bin/env python
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


import gensim
import os
import warnings


# Pre-process input documents
def get_documents(path, doc_length):
    print("Processing documents ...")
    docs = []
    for filename in os.listdir(path):
        if not filename.startswith("."):
            file = open(path + "/" + filename)
            string = file.read()
            file.close()
            if doc_length > 0:
                for piece in splitter(doc_length, string):
                    docs.append(piece)
            else:
                docs.append(string)
    num_docs = len(docs)
    print("Number of documents: %i" % num_docs)
    return docs


# Split large documents into smaller parts
def splitter(n, s):
    pieces = s.split()
    return (" ".join(pieces[i:i+n]) for i in xrange(0, len(pieces), n))


# Set stop word list
def get_stop_words(path):
    print("Getting stop words ...")
    stop_words = []
    for filename in os.listdir(path):
        if not filename.startswith("."):
            file = open(path + os.sep + filename)
            string = file.read()
            string = string.replace("\r\n", "\n")
            string = string.replace("\r", "\n")
            words = string.split("\n")
            for w in words:
                stop_words.append(w)
    num_words = len(stop_words)
    print("Number of stop words: %i" % num_words)
    return stop_words


# Read documents, tokenize
def iter_docs(doclist, stoplist):
    for doc in doclist:
        yield (x for x in gensim.utils.tokenize(doc, lowercase=True, deacc=True, errors="replace") if x not in stoplist)


# Corpus class
class MyCorpus(object):
    warnings.filterwarnings('ignore')
    model_folder = "data/models"

    def __init__(self, topdir, stopdir, doc_length):
        self.doclist = get_documents(topdir, doc_length)
        self.stoplist = get_stop_words(stopdir)

        print("Generating dictionary ...")
        self.dictionary = gensim.corpora.Dictionary(iter_docs(self.doclist, self.stoplist))
        no_above = 0.95	
        if len(self.doclist) < 5:
            no_above = 1
        self.dictionary.filter_extremes(no_below=2, no_above=no_above, keep_n=100000)
        num_tokens = len(self.dictionary.items())
        print("Number of unique tokens in dictionary: %s" % num_tokens)
        self.dictionary.save(os.path.join(self.model_folder, "kwg.dict"))

        print("Generating corpus ...")
        gensim.corpora.MmCorpus.serialize(os.path.join(self.model_folder, "kwg.mm"), self)

    def __iter__(self):
        for tokens in iter_docs(self.doclist, self.stoplist):
            yield self.dictionary.doc2bow(tokens)

    def load(self):
        dictionary = gensim.corpora.Dictionary.load(os.path.join(self.model_folder, "kwg.dict"))
        corpus = gensim.corpora.MmCorpus(os.path.join(self.model_folder, "kwg.mm"))
        return corpus, dictionary
