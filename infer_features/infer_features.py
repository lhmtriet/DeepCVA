# Functions to infer the NLP features of a dataset
import numpy as np
import pandas as pd
import re
import pickle
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
# Local imports
from tokenizer import get_tokenizer

embedding_sz = 300
w2v_model = None

# Remove changes separation between files
def clean_changes(changes):
	return re.sub("##<END>##[\r\n]", "", changes)

# Convert a sentence to a feature vector using text embeddings
#  Vector averaging of words in sentence
def sen_to_vec(sen):
	sen_vec = np.array([0.0] * embedding_sz)
	cnt = 0

	for w in sen:
		try:
			sen_vec = sen_vec + w2v_model[w]
			cnt += 1
		except:
			pass
	if cnt == 0:
		return np.array([0.0] * embedding_sz)

	return sen_vec / (cnt * 1.0)

# method = 'w2v' or 'bow'
def infer_nlp_features(data, scenario, method):
	global w2v_model
	# Clean the data
	data = data.map(clean_changes)
	# Create a tokenizer
	tokenizer = get_tokenizer()
	# Tokenize the features
	data = data.map(tokenizer)
	# Load NLP feature models
	if method == 'w2v':
		w2v_model = Word2Vec.load("feature_models/code_w2v_"+scenario+".model")
		features = np.asarray(data.map(sen_to_vec).values.tolist())
	elif method == 'bow':
		bow_model = pickle.load(open("feature_models/code_bow_"+scenario+".model", "rb"))
		features = bow_model.transform(data.astype(str).values)
	return features

if __name__ == '__main__':
	exit()
