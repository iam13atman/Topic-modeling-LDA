import nltk
import gensim
from gensim import corpora, models, similarities



lda =  models.LdaModel.load('mymodel.lda')

topic = ["Business [0]", "Entertainment [1]", "Sports [2]", "Technology [3]", "Politics [4]"]
i = 0
while(i < 5):
	print(topic[i])
	print(lda.show_topic(i, topn=10))
	i = i + 1
	print("**********************************")
