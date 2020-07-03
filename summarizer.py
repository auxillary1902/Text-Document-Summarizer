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

def sentence_similarity(sent1,sent2, stopwords = None):
	if stopwords is None:
		stopwords = []

	sent1 = [w.lower() for w in sent1]
	sent2 = [w.lower() for w in sent2]

	all_words = list(set(sent1 + sent2))

	vector1 = [0] * len(all_words)
	vector2 = [0] * len(all_words)

	#vector for first sentence
	for w in sent1:
		if w in stopwords:
			continue
		vector1[all_words.index(w)] += 1
    for w in sent2:
		if w in stopwords:
			continue
		vector2[all_words.index(w)] += 1

	return 1 - cosine_distance(vector1,vector2)

def build_similarity_matrix(sentences,stop_words):
	similarity_matrix = np.zeroes(len(sentences),len(sentences))
	for i1 in range(len(sentences)):
		for i2 in range(len(sentences)):
			if i1 == i2:
				continue
			similarity_matrix[i1][i2] = sentence_similarity(sentences[i1],sentences[i2],stop_words)
    
    return similarity_matrix


def generate_summary(filename,top_n = 5):
	stop_words = stopwords.words('english')
	summarize_text = []

	sentences = read_article(filename)

	sentences_similarity_matrix = build_similarity_matrix(sentences,stop_words)

	sentence_similarity_graph = nx.from_numpy_array(sentences_similarity_matrix)

	scores = nx.pagerank(sentence_similarity_graph)

	ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    
    for i in range(top_n):
    	summarize_text.append(" ".join(ranked_sentences[i][1]))

    print("Summarized text: \n ,".join(summarize_text))
