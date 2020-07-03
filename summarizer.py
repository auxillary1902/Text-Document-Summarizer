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


