import time
import pickle
import pickle5 as pk5
import math
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from gensim.models import Word2Vec

from scipy.sparse import coo_matrix, hstack, issparse
from scipy.stats import gmean
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.preprocessing import LabelEncoder, LabelBinarizer

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc, roc_auc_score, \
	matthews_corrcoef
from sklearn.neural_network import MLPClassifier

from keras.utils import np_utils, multi_gpu_model, plot_model
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Layer

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from keras.preprocessing import sequence
from keras.models import Sequential, Model, load_model, clone_model
from keras.layers import Dense, Dropout, Activation, Input, Flatten, Reshape, Dot
from keras.layers import LSTM, Bidirectional, GRU, RNN
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, Concatenate, BatchNormalization, MaxPooling1D
from keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

import keras.backend as K
from collections import Counter

from numpy.random import seed

seed(1)

import tensorflow as tf
import tensorflow.keras.backend as K

tf.random.set_seed(2)

################## Configurations#################

print("Arguments:", sys.argv)
print(tf.config.list_physical_devices("GPU"))

val_partitions = [str(i) for i in range(10)]

binary = False

# code vs. ast
feature_type = sys.argv[1]

# hunk, ss (Smallest Scope) and hc (Hunk Context)
feature_scope = sys.argv[2]

# word vs. char
token = sys.argv[3]

# Path where you store the data
data_path = '../'

commit_column = 'commit'

labels = ['cvss2_conf', 'cvss2_integrity', 'cvss2_avail', 'cvss2_accessvect',
			  'cvss2_accesscomp', 'cvss2_auth', 'cvss2_severity']

weights = {'cvss2_conf': 1.0, 'cvss2_integrity': 0.8, 'cvss2_avail': 0.8, 'cvss2_accessvect': 5.7,
			  'cvss2_accesscomp': 0.8, 'cvss2_auth': 2, 'cvss2_severity': 0.8}

weights = {'cvss2_conf': 1.0, 'cvss2_integrity': 1, 'cvss2_avail': 1, 'cvss2_accessvect': 1,
			  'cvss2_accesscomp': 1, 'cvss2_auth': 1, 'cvss2_severity': 1}

# DL configs
n_gpus = 1
epochs = 50
batch_size = 32

# Number of runs
runs = 10


###################################################

# Make directory if not present
def makedir(directory):
	Path(directory).mkdir(parents=True, exist_ok=True)


def gen_tok_pattern():
	single_toks = ['<=', '>=', '<', '>', '\\?', '\\/=', '\\+=', '\\-=', '\\+\\+', '--', '\\*=', '\\+', '-', '\\*',
				   '\\/', '!=', '==', '=', '!', '&=', '&', '\\%', '\\|\\|', '\\|=', '\\|', '\\$', '\\:']

	single_toks = '(?:' + '|'.join(single_toks) + ')'

	word_toks = '(?:[a-zA-Z0-9]+)'

	return single_toks + '|' + word_toks


# Extract features
def extract_features(config, start_n_gram, end_n_gram, token_pattern=None, vocabulary=None):
	if config == 1:
		return TfidfVectorizer(stop_words=None, ngram_range=(1, 1), use_idf=False, min_df=0.01,
							   norm=None, smooth_idf=False, lowercase=False, token_pattern=token_pattern,
							   vocabulary=vocabulary)
	elif config == 2:
		return TfidfVectorizer(stop_words=['aka'], ngram_range=(1, 1), use_idf=True, min_df=0.001,
							   norm='l2', token_pattern=r'\S*[A-Za-z]\S+', vocabulary=vocabulary)
	elif config < 6:
		return TfidfVectorizer(stop_words=['aka'], ngram_range=(start_n_gram, end_n_gram), use_idf=False,
							   min_df=0.001, norm=None, smooth_idf=False, token_pattern=r'\S*[A-Za-z]\S+',
							   vocabulary=vocabulary)

	return TfidfVectorizer(stop_words=['aka'], ngram_range=(start_n_gram, end_n_gram), use_idf=True,
						   min_df=0.001, norm='l2', token_pattern=r'\S*[A-Za-z]\S+', vocabulary=vocabulary)


def sen_to_vec(sen, embedding_sz, model):
	sen_vec = np.array([0.0] * embedding_sz)
	cnt = 0

	for w in sen:
		try:
			sen_vec = sen_vec + model[w]
			cnt += 1
		except:
			pass
	if cnt == 0:
		return np.array([0.0] * embedding_sz)

	return sen_vec / (cnt * 1.0)


def get_tokenizer(vectorize=False):
	code_token_pattern = gen_tok_pattern()
	vectorizer = extract_features(config=1, start_n_gram=1, end_n_gram=1, token_pattern=code_token_pattern)
	if vectorize:
		return vectorizer
	return vectorizer.build_analyzer()


def multi_to_binary(row):
	if row == "-":
		return 0
	return 1


def most_common(l):
	data = Counter(l)
	return data.most_common(1)[0][0]


def extract_label_results(y_true, y_pred):
	global labels
	global n_classes
	global lb_dict

	perf_str = []

	for index, label in enumerate(labels):

		label_output = label + "_output"

		if n_classes[label] == 2:
			cur_y_pred = np.round(y_pred[index]).ravel()
		else:
			cur_y_pred = np.argmax(y_pred[index], axis=1)

		cur_y_pred = np.array([lb_dict[label].classes_[int(item)] for item in cur_y_pred])
		cur_y_true = y_true[label_output].copy()
        
		precision = precision_score(cur_y_true, cur_y_pred, average='macro')
		recall = recall_score(cur_y_true, cur_y_pred, average='macro')
		gmean = math.sqrt(recall * precision)

		perf_str.append([accuracy_score(cur_y_true, cur_y_pred), precision, recall,
						 f1_score(cur_y_true, cur_y_pred, average='macro'), gmean,
						 matthews_corrcoef(cur_y_true, cur_y_pred)])      

	return perf_str


def validate_batch_sequential(val_prev, val_cur, y_val, model, fold, binary, combined=True):

	global labels

	start_time = time.time()

	if combined:
		val_input = np.c_[val_prev, val_cur]

	if combined:
		y_pred = model.predict(val_input, batch_size=256)
	else:

		# Check for feature type='hc'
		if len(val_prev.shape) == 3:
			y_pred = model.predict([val_prev[:, 0, :], val_prev[:, 1, :], val_cur[:, 0, :], val_cur[:, 1, :]],
								   batch_size=256)
		else:
			y_pred = model.predict([val_prev, val_cur], batch_size=256)
	
	perf_str_raw = np.asarray(extract_label_results(y_val, y_pred))
    
	perf_str = np.mean(perf_str_raw, axis=0)

	mcc_labels = ""

	for perf_str_tmp in perf_str_raw:
		for item_str in perf_str_tmp:
			mcc_labels += "," + "{:.3f}".format(item_str)

	_val_metric = perf_str[-1]

	val_time = time.time() - start_time

	# Evaluate
	perf_str = fold + "," + "{:.3f}".format(perf_str[0]) + "," + "{:.3f}".format(perf_str[1]) + "," + \
			   "{:.3f}".format(perf_str[2]) + "," + "{:.3f}".format(perf_str[3]) + "," + \
			   "{:.3f}".format(perf_str[4]) + "," + "{:.3f}".format(perf_str[5]) + "," + \
			   "{:.3f}".format(val_time)

	perf_str += mcc_labels + "\n"

	print(perf_str)
	print("Validation time:", val_time, "s.\n")

	return _val_metric, perf_str


class MyMetricsSequential(Callback):
	def __init__(self, val_prev, val_cur, y_val, fold, binary=False, patience_lr=3, patience_stopping=5,
				 decrease_ratio=0.2, best_model_name='best_model.h5',
				 **kwargs):
		super(Callback, self).__init__(**kwargs)

		self.val_prev = val_prev
		self.val_cur = val_cur

		self.y_val = y_val

		self.fold = fold

		self.binary = binary

		self.patience_lr = patience_lr
		self.patience_stopping = patience_stopping

		self.best_val = -1
		self.best_epoch_val = 0

		self.best_train = 100
		self.best_epoch_train = 0

		self.decrease_ratio = decrease_ratio
		self.best_model_name = best_model_name

	def on_train_begin(self, logs={}):
		self.val_f1s = []

	def on_epoch_end(self, epoch, logs={}):

		_val_metric, _ = validate_batch_sequential(self.val_prev, self.val_cur, self.y_val, self.model, self.fold,
												   self.binary, combined=False)

		cur_train_loss = float(logs.get('loss'))

		if cur_train_loss < self.best_train:
			self.best_train = cur_train_loss
			self.best_epoch_train = epoch

		self.val_f1s.append(_val_metric)

		if _val_metric > self.best_val:
			print("Val metric increases from", self.best_val, "to", _val_metric)
			print("Saving best model at epoch", epoch + 1)
			self.best_epoch_val = epoch
			self.best_val = _val_metric
			self.model.save(self.best_model_name)
		else:
			print("Val metric did not increase from the last epoch.")

		if epoch - self.best_epoch_train == self.patience_lr:
			self.model.optimizer.lr.assign(self.model.optimizer.lr * self.decrease_ratio)
			print("Train loss did not decrease in the last", self.patience_lr, "epochs, thus reducing learning rate to",
				  K.eval(self.model.optimizer.lr), ".")

		if epoch - self.best_epoch_val == self.patience_stopping:
			print("Val metric did not increase in the last", self.patience_stopping,
				  "epochs, thus stopping training.")
			self.model.stop_training = True

		return


def create_conv1d_layer(input_layer, rnn_units, attention_units, filters, kernel_size, name):
	conv1d = Conv1D(filters,
					kernel_size,
					padding='valid',
					activation=None,
					strides=1, name=name)(input_layer)
	conv1d = BatchNormalization()(conv1d)
	conv1d = Activation('relu')(conv1d)
	conv1d = GRU(rnn_units, dropout=0.2, recurrent_activation='sigmoid', reset_after=True, return_sequences=True)(conv1d)
	conv1d = Attention(attention_units)(conv1d)

	return conv1d


def generate_output(df, label, fit=True, lb=None):

	if fit:
		lb = LabelBinarizer()
		y = lb.fit_transform(df[label].values)

		return y, lb
	
	y = df[label].values

	return y


def generate_output_layer(previous_layer, n_class, name, hidden_units=128):

	output_layer = Dense(hidden_units, activation='relu')(previous_layer)

	if n_class == 2:
	
		output_layer = Dense(1, activation='sigmoid', name=name)(output_layer)
	elif n_class > 2:

		output_layer = Dense(n_class, activation='softmax', name=name)(output_layer)

	if n_class > 2:
		loss = 'categorical_crossentropy'
	else:
		loss = 'binary_crossentropy'

	return output_layer, loss


class Attention(Layer):
	def __init__(self, units=32, return_attention=False, **kwargs):
		super(Attention, self).__init__(**kwargs)
		self.units = units
		self.return_attention = return_attention

	def build(self, input_shape):
		self.Wh = self.add_weight(name="att_weight_1", shape=(input_shape[-1], self.units), initializer="normal")
		self.Ws = self.add_weight(name="att_weight_2", shape=(self.units, 1), initializer="normal")
		self.bh = self.add_weight(name="att_bias", shape=(self.units,), initializer="zeros")
		super(Attention, self).build(input_shape)

	def call(self, x, **kwargs):

		et = K.tanh(K.dot(x, self.Wh) + self.bh)  # maxlen x self.units
		et = K.squeeze(K.dot(et, self.Ws), axis=-1)  # maxlen x self.units x self.units x 1 = maxlen x 1 + self.bs

		at = K.softmax(et)
		at = K.expand_dims(at, axis=-1)
		output = x * at

		if self.return_attention:
			return [K.sum(output, axis=1), K.squeeze(at, axis=-1)]

		return K.sum(output, axis=1)

	def compute_output_shape(self, input_shape):

		output_shape = (input_shape[0], input_shape[-1])

		if self.return_attention:
			attention_shape = (input_shape[0], input_shape[1])
			return [output_shape, attention_shape]

		return output_shape

	def get_config(self):
		config = {
			'units': self.units,
		}
		base_config = super(Attention, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	@staticmethod
	def get_custom_objects():
		return {'Attention': Attention}


class AddPoolingLayer(Layer):
	def __init__(self, axis=1, **kwargs):
		super(AddPoolingLayer, self).__init__(**kwargs)
		self.axis = axis

	def build(self, input_shape):
		super(AddPoolingLayer, self).build(input_shape)

	def call(self, x, **kwargs):
		print(x.shape)

		return K.sum(x, axis=self.axis)

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[-1])

	def get_config(self):
		config = {
			'axis': self.axis,
		}
		base_config = super(AddPoolingLayer, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	@staticmethod
	def get_custom_objects():
		return {'AddPoolingLayer': AddPoolingLayer}


def feature_extractor_cnn(max_features, embedding_dims, filters, maxlen, rnn_units, attention_units):
	input_layer = Input(shape=(maxlen,))

	embedding_layer = Embedding(max_features, embedding_dims, name='Embedding_Layer')(input_layer)
	embedding_layer = Dropout(0.2)(embedding_layer)

	conv1d_1 = create_conv1d_layer(embedding_layer, rnn_units, attention_units, filters=filters, kernel_size=1, name='Conv1d_1')
	conv1d_3 = create_conv1d_layer(embedding_layer, rnn_units, attention_units, filters=filters, kernel_size=3, name='Conv1d_3')
	conv1d_5 = create_conv1d_layer(embedding_layer, rnn_units, attention_units, filters=filters, kernel_size=5, name='Conv1d_5')

	output = Concatenate(name='Concatenate_Layer')([conv1d_1, conv1d_3, conv1d_5])

	model = Model(inputs=[input_layer], outputs=output, name='Feature_Extractor')

	return model


def feature_extractor(max_features, embedding_dims, rnn_units, attention_units):
	model = Sequential(name='Feature_Extractor')
	model.add(Embedding(max_features, embedding_dims, name='Embedding_Layer'))
	model.add(Dropout(0.2))
	model.add(GRU(rnn_units, dropout=0.2, recurrent_activation='sigmoid', reset_after=True, return_sequences=True))
	model.add(Attention(attention_units, name='Attention_Layer'))

	return model


def baseline_dl_model(input_dim, max_features, embedding_dims, filters, rnn_units, attention_units, labels, n_classes, hc=False,
					  sparse=False, n_gpus=1):
	# shared_feature_extractor = feature_extractor(max_features, embedding_dims, rnn_units, attention_units)
	shared_feature_extractor = feature_extractor_cnn(max_features, embedding_dims, filters, input_dim, rnn_units, attention_units)

	if hc:
		prev_code_input = Input(shape=(input_dim,), name='Prev_Code', sparse=sparse)
		cur_code_input = Input(shape=(input_dim,), name='Cur_Code', sparse=sparse)

		prev_context_input = Input(shape=(input_dim,), name='Prev_Context', sparse=sparse)
		cur_context_input = Input(shape=(input_dim,), name='Cur_Context', sparse=sparse)

		prev_code = shared_feature_extractor(prev_code_input)
		prev_context = shared_feature_extractor(prev_context_input)

		cur_code = shared_feature_extractor(cur_code_input)
		cur_context = shared_feature_extractor(cur_context_input)

		prev = Concatenate(name='Prev')([prev_code, prev_context])
		cur = Concatenate(name='Cur')([cur_code, cur_context])

	else:
		prev_input = Input(shape=(input_dim,), name='Prev', sparse=sparse)
		cur_input = Input(shape=(input_dim,), name='Cur', sparse=sparse)

		prev = shared_feature_extractor(prev_input)
		cur = shared_feature_extractor(cur_input)
	
	commit = Concatenate(name='Changes')([prev, cur])
	
	output_layers = []
	losses = {}
	loss_weights = {}

	max_weight = np.max([val for val in list(weights.values())])

	norm_weights = {}
	for label in labels:
		norm_weights[label] = weights[label] * 1.0 / max_weight

	for label in labels:
		output_layer_name = label + "_output"
		output_layer = generate_output_layer(commit, n_class=n_classes[label], name=output_layer_name, hidden_units=rnn_units)
		output_layers.append(output_layer[0])

		cur_loss_name = label + "_output"

		losses[cur_loss_name] = output_layer[1]

		loss_weights[cur_loss_name] = norm_weights[label]

	if hc:
		model = Model(inputs=[prev_code_input, prev_context_input, cur_code_input, cur_context_input],
					  outputs=output_layers, name='DL_Multitask_Sequential_Model')
	else:
		model = Model(inputs=[prev_input, cur_input], outputs=output_layers, name='DL_Multitask_Sequential_Model')

	if n_gpus > 1:
		model = multi_gpu_model(model, gpus=n_gpus)
	
	opt = Adam(lr=0.001)

	model.compile(loss=losses, metrics=['acc'], optimizer=opt, loss_weights=loss_weights)

	print(model.summary())

	return model


def series_to_dense(row):
	return row.toarray()[0]


################## Starting main code ###################################
print("\n########################")
print("Loading mapping files")

print(val_partitions)

for val_partition in val_partitions:

	print("\n########################")
	print("Current partition:", val_partition)

	mapping_file = data_path + 'cvss_map.csv'

	map_df = pd.read_csv(mapping_file)
	map_df.columns = map_df.columns.str.lower()

	raw_data_path = data_path + 'inferred_features_time/seq_' + feature_type + '_' + feature_scope + '_' + token + '_' + \
					'multiclass' + '_' + val_partition + '.pkl'

	raw_data = pd.read_pickle(raw_data_path, compression='gzip')

	raw_data.columns = raw_data.columns.str.lower()

	data_cols = ['prev_features', 'cur_features']

	prev_col, cur_col = data_cols

	print("\n########################")
	print("Loading data")

	selected_cols = [commit_column] + labels

	map_column = 'time_partition'
	map_df[map_column] = map_df[map_column].astype('int32')

	fold_indices = np.unique(map_df[map_column])

	train_indices = [num for num in fold_indices if num <= int(val_partition)]
	val_index = int(val_partition) + 1

	train_commits = map_df.loc[map_df[map_column].isin(train_indices)][selected_cols].set_index(commit_column)
	val_commits = map_df.loc[map_df[map_column] == val_index][selected_cols].set_index(commit_column)

	train_commits = raw_data.join(train_commits, how='inner', on=commit_column)
	val_commits = raw_data.join(val_commits, how='inner', on=commit_column)

	train_commits['index_col'] = train_commits.index
	val_commits['index_col'] = val_commits.index

	test_index = int(val_partition) + 2
	test_commits = map_df.loc[map_df[map_column] == test_index][selected_cols].set_index(commit_column)
	test_commits = raw_data.join(test_commits, how='inner', on=commit_column)
	test_commits['index_col'] = test_commits.index
    
	print(train_commits.values.shape, val_commits.values.shape)

	# Extracting features
	print("\n########################")
	print("Extracting features")

	start_time = time.time()

	train_prev_input = np.array(train_commits[prev_col].values.tolist())
	train_cur_input = np.array(train_commits[cur_col].values.tolist())

	val_prev_input = np.array(val_commits[prev_col].values.tolist())
	val_cur_input = np.array(val_commits[cur_col].values.tolist())

	train_input = np.c_[train_prev_input, train_cur_input]
	val_input = np.c_[val_prev_input, val_cur_input]

	test_prev_input = np.array(test_commits[prev_col].values.tolist())
	test_cur_input = np.array(test_commits[cur_col].values.tolist())

	test_input = np.c_[test_prev_input, test_cur_input]

	print("\n########################")
	print("Extracting labels")

	y_train = {}
	y_val = {}
	lb_dict = {}

	n_classes = {label: len(train_commits[label].unique()) for label in labels}

	for label in labels:
		cur_y_train, lb_dict[label] = generate_output(train_commits, label, fit=True)

		cur_output = label + "_output"
		y_train[cur_output] = cur_y_train

	for label in labels:
		cur_y_val = generate_output(val_commits, label, fit=False, lb=lb_dict[label])

		cur_output = label + "_output"
		y_val[cur_output] = cur_y_val

	y_test = {}

	for label in labels:
		cur_y_test = generate_output(test_commits, label, fit=False, lb=lb_dict[label])

		cur_output = label + "_output"
		y_test[cur_output] = cur_y_test

	print("\n########################")
	print("Training the model")

	model_folder = 'best_models_sequential_multitask_crnn/'
	makedir(model_folder)

	results = []

	for run in range(runs):

		print("\n##############################")
		print("Run #", run + 1)

		best_model_name = model_folder + feature_type + '_' + feature_scope + '_' + token + '_' + val_partition + '_best_model.h5'

		my_metrics = MyMetricsSequential(val_prev_input, val_cur_input, y_val, fold=val_partition, binary=binary, patience_stopping=5, best_model_name=best_model_name)
		
		hc = True if feature_scope == 'hc' else False

		clf = baseline_dl_model(input_dim=train_prev_input.shape[-1], max_features=10001, embedding_dims=300,
								filters=128, rnn_units=128, attention_units=128, labels=labels, n_classes=n_classes,
								hc=hc, sparse=False, n_gpus=n_gpus)
		
		print("Training models")

		if hc:
			train_prev_code, train_prev_context = train_prev_input[:, 0, :], train_prev_input[:, 1, :]
			train_cur_code, train_cur_context = train_cur_input[:, 0, :], train_cur_input[:, 1, :]
			clf.fit([train_prev_code, train_prev_context, train_cur_code, train_cur_context], y_train,
					batch_size=batch_size * n_gpus, epochs=epochs,
					validation_data=None, verbose=2, callbacks=[my_metrics])
		else:
			clf.fit([train_prev_input, train_cur_input], y_train, batch_size=batch_size * n_gpus, epochs=epochs,
					validation_data=None, verbose=2, callbacks=[my_metrics])

		print("Evaluating the best model on the validation set")
		try:
			best_clf = load_model(best_model_name, compile=False)
		except Exception:
			best_clf = load_model(best_model_name, custom_objects={'Attention': Attention}, compile=False)

		tmp_results = str(run) + ',' + 'val' + ',' + \
					  validate_batch_sequential(val_prev_input, val_cur_input, y_val, model=best_clf,
												fold=my_metrics.fold, binary=my_metrics.binary, combined=False)[1]
			
		print("Evaluating the best model on the testing set")

		test_metric, perf_str = validate_batch_sequential(test_prev_input, test_cur_input, y_test, model=best_clf,
															fold=my_metrics.fold, binary=my_metrics.binary,
															combined=False)

		tmp_results += str(run) + ',' + 'test' + ',' + perf_str
		
		results.extend([item.split(',') for item in tmp_results.splitlines()])

		# Cleaning up the model
		del clf
		del best_clf

		K.clear_session()        

            
	print("\n########################")
	print("Aggregating the results")

	result_folder = 'sequential_results_multitask_crnn/'
	makedir(result_folder)

	result_file = result_folder + feature_type + '_' + feature_scope + '_' + token + '_' + val_partition + '_results.csv'
    
	label_cols = []
	for label in labels:
		for metric in ['acc', 'prec', 'rec', 'f1', 'gmean', 'mcc']:
			label_cols.append(label + "_" + metric)

	columns = ['run', 'set', 'fold', 'accuracy', 'precision', 'recall', 'f1-score', 'gmean', 'mcc',
			   'pred_time'] + label_cols

	result_df = pd.DataFrame(results, columns=columns)
	result_df.iloc[:, 3:] = result_df.iloc[:, 3:].astype(float)
	result_df.to_csv(result_file, index=False)

	val_res = result_df.loc[result_df['set'] == 'val']['mcc'].values

	print("Average validation result:", np.mean(val_res))

	test_res = result_df.loc[result_df['set'] == 'test']['mcc']
	print("Average testing result:", np.mean(test_res))
