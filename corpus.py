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


import gensim
import os
import warnings


# Pre-process input documents
def get_documents(path, doc_length):
    print('Processing documents ...')
    docs = []
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            with open(path + '/' + filename) as f:
                s = decode(f.read())
                if not s:
                    print('Could not decode file contents, skipping.')
                    continue
                if doc_length > 0:
                    for piece in splitter(doc_length, s):
                        docs.append(piece)
                else:
                    docs.append(s)
    num_docs = len(docs)
    print('Number of documents: %i' % num_docs)
    return docs


# Split large documents into smaller parts
def splitter(n, s):
    pieces = s.split()
    return (' '.join(pieces[i:i+n]) for i in xrange(0, len(pieces), n))


# Set stop word list
def get_stop_words(path):
    print('Getting stop words ...')
    stop_words = []
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
            with open(path + os.sep + filename) as f:
                s = decode(f.read())
                if not s:
                    print('Could not decode file contents, skipping.')
                    continue
                stop_words += [sw.lower() for sw in s.split() if sw]
    num_words = len(stop_words)
    print('Number of stop words: %i' % num_words)
    return stop_words


# Try to decode various possible encodings
def decode(s):
    encodings = ['utf-8', 'iso-8859-1']
    decoded = None
    for e in encodings:
        try:
            decoded = s.decode(e)
            break
        except UnicodeDecodeError:
            continue
    return decoded


# Read documents, tokenize
def iter_docs(doc_list, stop_list):
    for doc in doc_list:
        yield (x for x in gensim.utils.tokenize(doc, lowercase=True, deacc=True,
                errors='replace') if x not in stop_list)


# Corpus class
class MyCorpus(object):
    warnings.filterwarnings('ignore')
    model_folder = 'data/models'

    def __init__(self, doc_dir, stop_dir, doc_length):
        self.doc_list = get_documents(doc_dir, doc_length)
        self.stop_list = get_stop_words(stop_dir)

        print('Generating dictionary ...')
        self.dictionary = gensim.corpora.Dictionary(iter_docs(self.doc_list,
                self.stop_list))
        no_below = 1 if len(self.doc_list) <= 10 else 2
        no_above = 1 if len(self.doc_list) <= 10 else 0.95
        self.dictionary.filter_extremes(no_below=no_below, no_above=no_above,
                keep_n=100000)
        num_tokens = len(self.dictionary.items())
        print('Number of unique tokens in dictionary: %s' % num_tokens)
        self.dictionary.save(os.path.join(self.model_folder, 'kwg.dict'))

        print('Generating corpus ...')
        gensim.corpora.MmCorpus.serialize(os.path.join(self.model_folder,
                'kwg.mm'), self)

    def __iter__(self):
        for tokens in iter_docs(self.doc_list, self.stop_list):
            yield self.dictionary.doc2bow(tokens)

    def load(self):
        dictionary = gensim.corpora.Dictionary.load(os.path.join(self.model_folder,
                'kwg.dict'))
        corpus = gensim.corpora.MmCorpus(os.path.join(self.model_folder,
                'kwg.mm'))
        return corpus, dictionary
