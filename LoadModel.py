import nltk
import gensim
from gensim import corpora, models, similarities
from nltk.corpus import stopwords 
from nltk.corpus import PlaintextCorpusReader
from nltk import FreqDist
import os

stop_file = open("terrier-stop.txt")
stopwords = stop_file.read().splitlines()

lda =  models.LdaModel.load('mymodel.lda')
document_root = '/path/to/text/files/'
file_list = os.listdir(document_root)

for file in file_list:

	new_wordlists = PlaintextCorpusReader(document_root, file)
	new_texts = []
	
	for item in new_wordlists.fileids():
		try:
			new_texts = [word.lower() for word in new_wordlists.words(item) if word.lower() not in stopwords and word.isalpha()]
			
		except UnicodeError:
			print(word, item)
	
	print("New File")
	bow = lda.id2word.doc2bow(new_texts)
	print("Topic distribution for " + file)
	print(lda[bow])