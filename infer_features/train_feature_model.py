# Train feature models for NLP code features. Save the inferred features.
import pandas as pd
import numpy as np
import pickle
import sys
import re
from glob import glob
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import hstack
from pathlib import Path
# Local Import
from tokenizer import get_tokenizer
from infer_features import infer_nlp_features

num_folds = 10
time_partition = True	# Use time-based folds
# W2V Settings
embedding_sz = 300
num_workers = 3

# Make directory if not present
def makedir(directory):
	Path(directory).mkdir(parents=True, exist_ok=True)

# Train a w2v embedding model
def train_w2v(sentences, dataName, partition):
	# Tokenize files
	tokenizer = get_tokenizer()
	sentences = list(map(tokenizer, sentences))
	# Train and save model
	w2v_model = Word2Vec(sentences, size=embedding_sz, window=5, min_count=0, max_vocab_size=10000, workers=num_workers, sg=1, seed=42)
	w2v_model.save("feature_models/code_w2v_"+dataName+"_"+partition+".model")

# Train and save a BoW feature model
def train_BoW(sentences, dataName, partition):
	vectorizer = get_tokenizer(vectorize=True)
	vectorizer.fit(sentences)
	pickle.dump(vectorizer, open("feature_models/code_bow_"+dataName+"_"+partition+".model", "wb"))

# Print the vocab of a 'BoW' or a 'w2v' model for the 'multiclass' or 'binary' scenario
def examine_vocab(model, scenario, partition):
	makedir("feature_models/vocab/")
	if model == 'BoW':
		# Load the model
		feature_model = pickle.load(open("feature_models/code_bow_"+scenario+"_"+partition+".model", "rb"))
		# Print the Vocab
		f = open("feature_models/vocab/code_bow_"+scenario+"_"+partition+".txt", 'w')
		for i in feature_model.vocabulary_:
			f.write(str(i) + '\n')
		f.close()
	elif model == 'w2v':
		# Load the model
		if model == 'w2v':
			feature_model = Word2Vec.load("feature_models/code_w2v_"+scenario+"_"+partition+".model")
		# Record the Vocab
		vocab_freqs = {}
		for i in feature_model.wv.vocab:
			vocab_freqs[i] = feature_model.wv.vocab[i].count
		# Sort the Vocab
		vocab_sorted = {k: v for k, v in sorted(vocab_freqs.items(), key=lambda item: item[1], reverse=True)}
		# Print the Vocab
		f = open("feature_models/vocab/code_w2v_"+scenario+"_"+partition+".txt", 'w')
		for i in vocab_sorted:
			f.write(str(i) + ' ' + str(vocab_sorted[i]) + '\n')
		f.close()

# Infer and save the features
def save_features(problem, fold, feature_set):
	# Load the data
	data = pd.read_parquet('data/code_'+feature_set+'.parquet')
	# Infer NLP features
	for method in ['w2v', 'bow']:
		# Get features
		prev_features = infer_nlp_features(data['prev_data'], problem+"_"+fold, method)
		cur_features = infer_nlp_features(data['cur_data'], problem+"_"+fold, method)
		if method == 'w2v':
			prev_features, cur_features = prev_features.tolist(), cur_features.tolist()
		if feature_set == 'hc':
			prev_context = infer_nlp_features(data['prev_context'], problem+"_"+fold, method)
			cur_context = infer_nlp_features(data['cur_context'], problem+"_"+fold, method)
			if method == 'w2v':
				prev_context, cur_context = prev_context.tolist(), cur_context.tolist()
		# Convert to structured format
		features = pd.DataFrame()
		features['commit'] = data['commit']
		features['prev_features'] = [x for x in prev_features]
		features['cur_features'] = [x for x in cur_features]
		if feature_set == 'hc':
			features['prev_context'] = [x for x in prev_context]
			features['cur_context'] = [x for x in cur_context]
		# Save as .pkl
		features.to_pickle("inferred_features/code_"+feature_set+"_"+method+"_"+problem+"_"+fold+".pkl")

def infer_features():
	makedir('inferred_features/')
	for problem in ['multiclass', 'binary']:
		for fold in ['holdout', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
			for feature_set in ['hunk', 'ss', 'hc']:
				save_features(problem, fold, feature_set)

def train_feature_model():
	makedir('feature_models/')
	# Load the data
	raw_file_data = pd.read_parquet("data/code_file.parquet")

	### HOLDOUT ###
	# Consider each scenario and partition type
	# for problem_type in ['multiclass', 'binary']:
	for problem_type in ['multiclass']:
		if problem_type == 'multiclass':
			commit_data = pd.read_csv("mc_map.csv")
		elif problem_type == 'binary':
			commit_data = pd.read_csv('binary_map.csv')
		file_data = pd.merge(raw_file_data, commit_data, how='left', on='commit')
		# Prepare the data
		data = file_data[file_data['set'] == 'train']
		if problem_type == 'multiclass':
			data = data[data['cwe'] != '-']
		print(data)
		data_train = np.r_[data['prev_file'].values, data['cur_file'].values]
		# Train the models
		train_w2v(data_train, problem_type, 'holdout')
		train_BoW(data_train, problem_type, 'holdout')
		# Examine vocabulary
		for i in ['w2v', 'BoW']:
			examine_vocab(i, problem_type, "holdout")

	### FOLDS ###
	# Consider each scenario and partition type
	# for problem_type in ['multiclass', 'binary']:
	for problem_type in ['multiclass']:
		if problem_type == 'multiclass':
			commit_data = pd.read_csv("mc_map.csv")
		elif problem_type == 'binary':
			commit_data = pd.read_csv('binary_map.csv')
		file_data = pd.merge(raw_file_data, commit_data, how='left', on='commit')
		for k in range(0, num_folds):
			# Prepare the data
			if time_partition:
				data = pd.DataFrame()
				for i in range(0, k+1):
					data = pd.concat([data, file_data[file_data['time_partition'] == k]])
			else:
				data = file_data[file_data['partition'] != k]
			if problem_type == 'multiclass':
				data = data[data['cwe'] != '-']
			print(data)
			data_train = np.r_[data['prev_file'].values, data['cur_file'].values]
			# Train the models
			train_w2v(data_train, problem_type, str(k))
			train_BoW(data_train, problem_type, str(k))
			# Examine vocabulary
			for i in ['w2v', 'BoW']:
				examine_vocab(i, problem_type, str(k))

if __name__ == '__main__':
	if sys.argv[1] == 'train':
		train_feature_model()
	else:
		save_features(sys.argv[1], sys.argv[2], sys.argv[3])
	# infer_features()
	exit()
