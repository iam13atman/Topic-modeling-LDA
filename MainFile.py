import nltk
import gensim
from gensim import corpora, models, similarities
from nltk.corpus import stopwords 
from nltk.corpus import PlaintextCorpusReader
from nltk import FreqDist
import os

#Define the folder where all news articles reside
corpus_root = '/path/to/text/files/directory'
file = open("terrier-stop.txt")
stopwords = file.read().splitlines()
wordlists = PlaintextCorpusReader(corpus_root, '.*')

texts = []
for item in wordlists.fileids():
    try:
        temp = [word.lower() for word in wordlists.words(item) if word.lower() not in stopwords and word.isalpha()]
        texts.append(temp)
    except UnicodeError:
        print(word, item)
        

flat_list = [item for sublist in texts for item in sublist]
fdist = FreqDist(flat_list)


dictionary = corpora.Dictionary(texts)
dictionary.save('/path/to/saving/dictionary.dict')

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('/path/to/save/corpus/new_corpus.mm', corpus)


mm = gensim.corpora.MmCorpus('new_corpuss.mm')


lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=dictionary, num_topics= 5, update_every=2, chunksize=10, passes=1)



