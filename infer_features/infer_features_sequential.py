from text_processing import TextProcessor
from helpers import *
from multiprocessing import Pool, cpu_count, freeze_support

data_path = '../'
feature_path = '../'

commit_column = 'commit'
cwe_column = 'cwe'
index_col = 'index_col'

total_len = []


def divide_work(f, args):
	return f(*args)


tp = None
token = None
tokenizer = None
max_len = None
total_len = []


def transform_features(col, tp, token, tokenizer, max_len):
	# def transform_features(col):
	global total_len
	# global tp
	# global token
	# global global_tokenizer
	# global max_len

	code_token_pattern = gen_tok_pattern()
	vectorizer = extract_features(config=1, start_n_gram=1, end_n_gram=1, token_pattern=code_token_pattern)
	tokenizer = vectorizer.build_analyzer()

	# print(global_max_len)

	# Change to [col] instead of col when using together with apply function in pandas dataframe

	if col is None:
		# res, lens = tp.transform_text(col, type=token, tokenizer=tokenizer, maxlen=max_len)
		res, lens = tp.transform_text([''], type=token, tokenizer=tokenizer, maxlen=max_len)
	else:
		res, lens = tp.transform_text([col], type=token, tokenizer=tokenizer, maxlen=max_len)
	# print(res)
	if lens[0] > 0:
		total_len.append(lens[0])
	return res


def transform_features_combined(col):
	# global total_len
	global tp
	global token
	# global global_tokenizer
	global max_len

	code_token_pattern = gen_tok_pattern()
	vectorizer = extract_features(config=1, start_n_gram=1, end_n_gram=1, token_pattern=code_token_pattern)
	tokenizer = vectorizer.build_analyzer()

	# Change to [col] instead of col when using together with apply function in pandas dataframe

	if len(col) == 2:

		if col[0] is None:
			prev_res, _ = tp.transform_text([''], type=token, tokenizer=tokenizer, maxlen=max_len)
		else:
			prev_res, _ = tp.transform_text([col[0]], type=token, tokenizer=tokenizer, maxlen=max_len)

		if col[1] is None:
			cur_res, _ = tp.transform_text([''], type=token, tokenizer=tokenizer, maxlen=max_len)
		else:
			cur_res, _ = tp.transform_text([col[1]], type=token, tokenizer=tokenizer, maxlen=max_len)

		return [prev_res[0], cur_res[0]]

	if col[0] is None:
		prev_data, _ = tp.transform_text([''], type=token, tokenizer=tokenizer, maxlen=max_len)
	else:
		prev_data, _ = tp.transform_text([col[0]], type=token, tokenizer=tokenizer, maxlen=max_len)

	if col[1] is None:
		prev_context, _ = tp.transform_text([''], type=token, tokenizer=tokenizer, maxlen=max_len)
	else:
		prev_context, _ = tp.transform_text([col[1]], type=token, tokenizer=tokenizer, maxlen=max_len)

	if col[2] is None:
		cur_data, _ = tp.transform_text([''], type=token, tokenizer=tokenizer, maxlen=max_len)
	else:
		cur_data, _ = tp.transform_text([col[2]], type=token, tokenizer=tokenizer, maxlen=max_len)

	if col[3] is None:
		cur_context, _ = tp.transform_text([''], type=token, tokenizer=tokenizer, maxlen=max_len)
	else:
		cur_context, _ = tp.transform_text([col[3]], type=token, tokenizer=tokenizer, maxlen=max_len)

	return [[prev_data[0], prev_context[0]], [cur_data[0], cur_context[0]]]


def init_pool(local_tp, local_token, local_max_len):
	global tp
	global token
	global max_len

	tp = local_tp
	token = local_token
	max_len = local_max_len


def generate_data(val_partitions, argv=None):
	print(argv)

	# Binary vs multiclass
	binary = True
	binary = True if argv[1] == 'binary' else False

	# binary vs. multiclass
	scenario = 'binary' if binary else 'multiclass'

	# code, manual vs. ast
	feature_type = 'code'
	# feature_type = 'ast'
	# feature_type = 'manual'
	feature_type = argv[2]

	# hunk, ss (Smallest Scope) and hc (Hunk Context)
	# feature_scope = 'hunk'
	# feature_scope = 'ss'
	feature_scope = 'hc'
	feature_scope = argv[3]

	data_file = data_path + 'data/' + feature_type + '_file.parquet'
	file_data = pd.read_parquet(data_file)
	file_data.columns = file_data.columns.str.lower()

	if binary:
		mapping_file = data_path + 'binary_map.csv'
	else:
		mapping_file = data_path + 'cvss_map.csv'

	map_df = pd.read_csv(mapping_file)
	map_df.columns = map_df.columns.str.lower()

	raw_data = data_path + 'data/' + feature_type + '_' + feature_scope + '.parquet'
	raw_data = pd.read_parquet(raw_data)
	raw_data.columns = raw_data.columns.str.lower()

	# print(raw_data.columns)

	# file_data = file_data.iloc[:10]
	# raw_data = raw_data.iloc[:10]

	code_token_pattern = gen_tok_pattern()
	vectorizer = extract_features(config=1, start_n_gram=1, end_n_gram=1, token_pattern=code_token_pattern)
	analyzer = vectorizer.build_analyzer()

	# file_cols = ['prev_file', 'cur_file']

	# prev_data_col, cur_data_col = ['prev_data', 'cur_data']
	prev_file_col, cur_file_col = ['prev_file', 'cur_file']

	# if 'prev_file' in raw_data.columns:
	# 	raw_data.rename(columns={'prev_file': 'prev_data'}, inplace=True)
	# 	raw_data.rename(columns={'cur_file': 'cur_data'}, inplace=True)

	data_cols = raw_data.columns
	data_cols = data_cols[data_cols != 'commit']

	prev_feature_col, cur_feature_col = 'prev_features', 'cur_features'

	token = 'word'
	token = argv[4]

	if token == 'word':
		max_len = 512
	else:
		max_len = 1024

	# code/ast_hunk.parquet

	for val_partition in val_partitions:

		max_features = 10000

		print("Current partition:", val_partition)

		# Loading data

		# data_cols = ['prev_data', 'cur_data']

		print("Loading data")

		if val_partition == 'holdout':
			map_column = 'set'
			train_commits = map_df.loc[map_df[map_column] == 'train'][[commit_column, cwe_column]].set_index(commit_column)
		elif val_partition == 'train_holdout':
			map_column = 'set'
			train_commits = map_df.loc[(map_df[map_column] == 'train') | (map_df[map_column] == 'val')][[commit_column, cwe_column]].set_index(commit_column)  
		else:
			print(map_df.columns, val_partition)
			map_column = 'time_partition'
			map_df[map_column] = map_df[map_column].astype('int32')

			fold_indices = np.unique(map_df[map_column])

			train_indices = [num for num in fold_indices if num <= int(val_partition)]

			train_commits = map_df.loc[map_df[map_column].isin(train_indices)][[commit_column, cwe_column]].set_index(
				commit_column)

		train_files = file_data.join(train_commits, how='inner', on=commit_column)
		train_files = np.r_[train_files[prev_file_col], train_files[cur_file_col]]

		start_time = time.time()

		# Train feature model
		print("Training feature model")
		tp = TextProcessor(oov_token=None)

		if token == 'word':
			tp.build_vocab_word(train_files, min_ngram=1, max_ngram=1, min_df=0.001, max_df=1.0, tokenizer=analyzer,
								max_vocab=max_features,
								append_dict=False)
		else:
			tp.build_vocab_char(train_files, min_ngram=1, max_ngram=1, word_bound=True, whole_word=False, min_df=0.001,
								max_df=0.7, max_vocab=max_features, append_dict=False)
		
		if len(tp.word_dict) > 0:
			# print(tp.word_dict)
			max_features = max(tp.word_dict.values()) + 1
		else:
			# print(tp.char_dict)
			max_features = max(tp.char_dict.values()) + 1

		# print(tokenizer.word_dict)
		print("Number of features:", max_features)

		print("Training feature model time:", time.time() - start_time, "s.")

		# Transform features
		print("Transforming features")

		start_time = time.time()

		features_df = raw_data.copy()

		print(features_df.columns)
        
		global total_len
        
		total_len = []

		combined_data = list(zip(*[features_df[data_col].values.tolist() for data_col in data_cols]))
		
		no_processors = cpu_count()
		chunkSz = len(features_df) // no_processors
		
		mp = Pool(processes=no_processors, initializer=init_pool, initargs=(tp, token, max_len))
		
		combined_out = mp.map(transform_features_combined, combined_data, chunksize=chunkSz)
		
		combined_out = np.asarray(combined_out)
		print(combined_out.shape, combined_out[:, 0, :].shape)
		
		mp.close()
		mp.join()
		
		if feature_scope == 'hc':
			features_df[prev_feature_col] = combined_out[:, 0, :, :].tolist()
			features_df[cur_feature_col] = combined_out[:, 1, :, :].tolist()
		else:
			features_df[prev_feature_col] = combined_out[:, 0, :].tolist()
			features_df[cur_feature_col] = combined_out[:, 1, :].tolist()
		
		features_df.drop(columns=data_cols, inplace=True)
		
		print("Transforming features time:", time.time() - start_time, "s.")
		
		feature_path = data_path + 'inferred_features_time/' + 'seq_' + feature_type + '_' + feature_scope + '_' + token + \
					   '_' + scenario + '_' + val_partition + '.pkl'
		
		features_df.to_pickle(feature_path, compression='gzip', protocol=4)
		
		del features_df
		del combined_out

	del raw_data
	del file_data


def main():
	val_partitions = [str(i) for i in range(10)]
	generate_data(val_partitions, argv=sys.argv)


if __name__ == '__main__':
	freeze_support()
	main()
