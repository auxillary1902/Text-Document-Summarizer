from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
def read_article(filename):
	file =  open(filename,'r')
	filedata = file.readlines()
	article = filedata[0].split(". ")
	sentences = []

	for sentence in article:
		print(sentence)
		sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
		sentences.pop()

	return sentences

def build_similarity_matrix(sentences,stop_words):
	similarity_matrix = np.zeroes(len(sentences),len(sentences))
	for i1 in range(len(sentences)):
		for i2 in range(len(sentences)):
			if i1 == i2:
				continue
			similarity_matrix[i1][i2] = sentence_similarity(sentences[i1],sentences[i2],stop_words)
    
    return similarity_matrix