Keyword Generator
=================

The keyword generator offers two methods to extract relevant keywords from a collection of sample texts provided by the user:

1) keywords_tfidf.py, extracting keywords based on tf-idf scores. Options:
 
- k : number of keywords to be generated (default 10)
- d : document length (the documents provided by the user will be split into parts containing the specified number of words; by default the documents will not be split.)
 
2) keywords_lda.py, extracting keywords based on either [Gensim](https://radimrehurek.com/gensim/)'s or [Mallet](http://mallet.cs.umass.edu)'s implementation of lda topic modeling. Options:
 
- t : number of topics (default 10)
- w : number of words per topic (default 10)
- k : number of keywords (default 10)
- d : document length (the documents provided by the user will be split into parts containing the specified number of words; by default the documents will not be split.)
- m : mallet path (full path to the [Mallet](http://mallet.cs.umass.edu) executable; if not provided, [Gensim](https://radimrehurek.com/gensim/)'s lda implementation will be used to generate topics.)

Documents have to be placed in the data/documents folder, stop word lists in the data/stop_words folder. The keyword lists generated will be saved to a plain text file in the data/keywords folder. When using one of the topic modeling-based approaches, topic distributions per document will be saved in data/topic_distributions. 

[Gensim](https://radimrehurek.com/gensim/) and [Mallet](http://mallet.cs.umass.edu) need to be installed locally.

Some examples of commands:

- python keywords_tfidf.py
- python keywords_tfidf.py -k 20 -d 100
- python keywords_lda.py -k 10 -d 100 -t 5 -w 20
- python keywords_lda.py -k 10 -d 100 -t 5 -w 20 -m /usr/local/bin/mallet-2.0.7/bin/mallet

