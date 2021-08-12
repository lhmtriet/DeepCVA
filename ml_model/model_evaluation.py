# Use grid search to optimise the hyper-parameters for 6 ML Models
import warnings
warnings.filterwarnings('ignore')	# Ignore warnings
import pandas as pd
import numpy as np
import pickle
import math
import time
import sys
import os
import itertools
from collections import Counter
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc, matthews_corrcoef, roc_auc_score
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from scipy.sparse import hstack, vstack, coo_matrix
from imblearn.over_sampling import RandomOverSampler, SMOTE
# Local imports
from extra_evaluate import get_best_classifier, transfer_folds, optimal_classifiers, evaluate_one_task

# Hyper-parameters.
regularization_lr = ['0.01', '0.1', '1', '10', '100']	# Regularization Coefficient for LR
regularization_svm = ['0.1', '1', '10', '100', '1000', '10000'] 	# Regularization Coefficient for SVM
neighbours = ['11', '31', '51']					# Number of Neighbours for KNN
weights = ['uniform', 'distance']		# Distance Weight for KNN
norms = ['1', '2']								# Distance Norm for KNN
estimators = ['100', '300', '500']				# Number of estimators for RF, XGB, LGBM
leaf_nodes = ['100', '200', '300']			# Number of leaf nodes for RF, XGB, LGBM

# Settings
partition_type = True	# Use time-based folds
workers = 1

if len(sys.argv) < 6:
	print("Usage: python3 prediction_model.py <problem_type> <partition_type> <feature_type> <feature_scope> <token> <alg> <sampling>")
# Either a 'binary' or 'multiclass' problem
problem = sys.argv[2]

# Manual features
man_features = ['STARS', 'COMMITS', 'NS', 'LA', 'LD', 'LT', 'NDEV', 'AGE', 'NUC', 'EXP', 'NRF', 'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', 'continue', 'const', 'default', 'do', 'double', 'else', 'enum', 'exports', 'extends', 'false', 'final', 'finally', 'float', 'for', 'goto', 'if', 'implements', 'import', 'instanceof', 'int', 'interface', 'long', 'module', 'native', 'new', 'null', 'package', 'private', 'protected', 'public', 'requires', 'return', 'short', 'static', 'strictfp', 'super', 'switch', 'synchronized', 'this', 'throw', 'throws', 'transient', 'true', 'try', 'var', 'void', 'volatile', 'while']
# Open an output file
Path("prediction_models/").mkdir(exist_ok=True)
Path("ml_results/").mkdir(exist_ok=True)

def most_common(l):
	return Counter(l).most_common(1)[0][0]

# Train, Evaluate and Save a Classifier
def evaluate(clf, x_train, y_train, x_test, y_test, clf_settings, outpath, resampling, write=True):
	# Open the results file
	if not os.path.exists(outpath):
		outfile = open(outpath, 'w')
		outfile.write("problem,partition,feature_type,feature_scope,token,classifier,parameters,resampling,accuracy,precision,recall,gmean,auc,f1,mcc,train_time,pred_time\n")
	else:
		outfile = open(outpath, 'a')
	# Train
	t_start = time.time()
	clf.fit(x_train, y_train)
	train_time = time.time() - t_start
	# Predict
	p_start = time.time()
	y_pred = clf.predict(x_test)
	pred_time = time.time() - p_start

	# Evaluate
	if len(np.unique(np.r_[y_test, y_pred])) == 2:
		pos_label = most_common(y_test)
		precision = precision_score(y_test, y_pred, average='macro', pos_label=pos_label)
		recall = recall_score(y_test, y_pred, average='macro', pos_label=pos_label)
		f1 = f1_score(y_test, y_pred, average='macro', pos_label=pos_label)
	else:
		precision = precision_score(y_test, y_pred, average='macro')
		recall = recall_score(y_test, y_pred, average='macro')
		f1 = f1_score(y_test, y_pred, average='macro')
	gmean = math.sqrt(recall*precision)
	# Calculate AUC
	try:
		if len(np.unique(y_test)) == 2:
			auc = roc_auc_score(y_test, clf.predict_proba(x_test)[:,1])
		else:
			auc = roc_auc_score(y_test, clf.predict_proba(x_test), average='macro', multi_class='ovr')
	except:
		auc = 0

	output = f"{clf_settings},{resampling},{round(accuracy_score(y_test, y_pred),3)},{round(precision,3)},{round(recall,3)}," + \
			f"{round(gmean,3)},{round(auc,3)},{round(f1,3)},{round(matthews_corrcoef(y_test, y_pred),3)}," + \
			f"{round(train_time,3)},{round(pred_time,3)}\n"
	# Save results
	if write:
		outfile.write(output)
	# Save model
	model_path = clf_settings.replace(',','_')
	if resampling != 'none': model_path += resampling
	if 'validate' in outpath:
		pickle.dump(clf, open(f"prediction_models/{model_path}.model", "wb"))
	elif 'test' in outpath:
		pickle.dump(clf, open(f"prediction_models/test/{model_path}.model", "wb"))

def get_classifier(alg, multiclass, *parameters):
	# Logistic Regression.
	if alg == 'lr':
		if multiclass: problem_type = 'multinomial'
		else: problem_type = 'ovr'
		return LogisticRegression(C=float(parameters[0]), multi_class=problem_type, n_jobs=workers, solver='lbfgs', tol=0.001, max_iter=1000, random_state=42)
	# Support Vector Machine
	elif alg == 'svm':
		return SVC(random_state=42, C=float(parameters[0]), kernel='rbf', max_iter=-1, probability=True)
	# K-Nearest Neighbours
	elif alg == 'knn':
		return KNeighborsClassifier(n_neighbors=int(parameters[0]), weights=parameters[1], p=int(parameters[2]), n_jobs=workers)
	# Random Forest
	elif alg == 'rf':
		return RandomForestClassifier(n_estimators=int(parameters[0]), max_depth=None, max_leaf_nodes=int(parameters[1]), random_state=42, n_jobs=workers)
	# Extreme Gradient Boosting
	elif alg == 'xgb':
		if multiclass: problem_type = 'reg:squarederror'
		else: problem_type = 'binary:logistic'
		return XGBClassifier(objective=problem_type, max_depth=0, n_estimators=int(parameters[0]), max_leaves=int(parameters[1]), grow_policy='lossguide', n_jobs=workers, random_state=42, tree_method='hist')
	# Light Gradient Boosting Machine
	elif alg == 'lgbm':
		if multiclass: problem_type = 'multiclass'
		else: problem_type = 'binary'
		return LGBMClassifier(n_estimators=int(parameters[0]), num_leaves=int(parameters[1]), max_depth=-1, objective=problem_type, n_jobs=workers, random_state=42)

# Average folds results for K-Folds Cross-Testing
def average_folds(problem):
	# Define paths
	if problem in ['multiclass', 'binary']:
		resultpath = "ml_results/validate.csv"
		outpath = "ml_results/test.csv"
	elif problem == 'combined':
		resultpath = "ml_results/combined_validate.csv"
		outpath = "ml_results/combined_test.csv"
	elif 'cvss' in problem:
		resultpath = "ml_results/cvss_test.csv"
		outpath = "ml_results/cvss_test_average.csv"

	# Open the results file
	if not os.path.exists(outpath):
		outfile = open(outpath, 'w')
		outfile.write("problem,partition,feature_type,feature_scope,token,resampling,classifier,accuracy,precision,recall,gmean,f1,mcc,train_time,pred_time\n")
	else:
		outfile = open(outpath, 'a')
	# Load the validation results
	all_results = pd.read_csv(resultpath)
	print(all_results)

	for feature_type in ['code', 'ast', 'manual']:
		for feature_scope in ['hunk', 'ss', 'hc']:
			for token in ['bow', 'w2v']:
				for sampling in ['none', 'over', 'over+', 'over+_1', 'over+_5', 'over+_10', 'over+_15', 'over+_20']:
					for alg in ['lr', 'svm', 'rf', 'knn', 'xgb', 'lgbm', '-']:
						if feature_type == 'manual':
								feature_scope, token = '-', '-'
						if sampling != 'over+':
							results = all_results[(all_results['problem'] == problem) & (all_results['feature_type'] == feature_type) & (all_results['feature_scope'] == feature_scope) & (all_results['token'] == token) & (all_results['classifier'] == alg) & (all_results['resampling'] == sampling) & (all_results['partition'] != 'holdout')]
						else:
							results = all_results[(all_results['problem'] == problem) & (all_results['feature_type'] == feature_type) & (all_results['feature_scope'] == feature_scope) & (all_results['token'] == token) & (all_results['classifier'] == alg) & (all_results['resampling'].str.contains("over\+")) & (all_results['partition'] != 'holdout')]
						# Check empty
						if results.empty:
							print("Empty")
							continue
						print(results)
						if sampling == 'over+':
							results['resampling'] = 'over+'
						results.drop(columns=['parameters'], inplace=True)	# Drop the parameters column as not averagable.
						group_features = ['problem', 'feature_type', 'feature_scope', 'token', 'resampling', 'classifier']	# Features to average the folds by
						for x in [i for i in results.columns if i not in group_features]:
							results[x] = results[x].astype(float)	# Average
						results = results.groupby(group_features).agg({i: 'mean' for i in results.columns if i not in group_features}).reset_index()
						# Get best fold average
						test_result = results.iloc[results['mcc'].idxmax()]
						# Re-order and save
						test_result = test_result[['problem', 'partition', 'feature_type', 'feature_scope', 'token', 'resampling', 'classifier', 'accuracy', 'precision', 'recall', 'gmean', 'f1', 'mcc', 'train_time', 'pred_time']]
						output = [t if type(t) is str else str(round(t, 3)) for t in test_result.tolist()]
						outfile.write(','.join(output)+'\n')
				if feature_type == 'manual':
					return

def main(process, problem, fold, feature_type, feature_scope, token, alg, resampling='none'):
	"""
	Validate a given model for a given set of features
	problem = 'multiclass' / 'binary' / 'cvss_x' / 'combined' (cvss prediction and binary prediction)
	fold = 'holdout' / ['0' - '9']
	feature_type = 'code' / 'ast' / 'manual'
	feature_scope = 'hunk' / 'ss' / 'hc'
	token = 'bow' / 'w2v'
	alg = 'lr' / 'svm' / 'rf' / 'knn' / 'xgb' / 'lgbm'
	parameters = [...]
	"""

	# If folds test set
	if 'average' in process and fold != 'holdout':
		average_folds(problem)
		return
	# If folds transfer set
	if process == 'transfer' and fold != 'holdout':
		transfer_folds()
		optimal_classifiers()
		return
	# If compare one-task to binary and multiclass problem
	if process == 'compare_ot' and fold != 'holdout':
		evaluate_one_task(feature_type, feature_scope, token)
		return

	# Define result file
	if problem in ['multiclass', 'binary']: result_file = f"ml_results/{process}.csv"
	elif problem == 'combined': result_file = f"ml_results/combined_{process}.csv"
	elif 'cvss' in problem: result_file = f"ml_results/cvss_{process}.csv"

	# Load the commit map
	if problem == 'multiclass':
		commits = pd.read_csv("mc_map.csv")
	elif problem in ['binary', 'combined']:
		commits = pd.read_csv("binary_map.csv")
	elif 'cvss' in problem:
		commits = pd.read_csv('cvss_map.csv')
	# Get train and validation commits
	if fold == 'holdout':
		train_commits = commits[commits['set'] == 'train']
		val_commits = commits[commits['set'] == 'val']
		test_commits = commits[commits['set'] == 'test']
	else:
		if time_partition:
			train_commits = pd.DataFrame()
			for k in range(0, int(fold)+1):
				train_commits = pd.concat([train_commits, commits[commits['time_partition'] == k]])
			val_commits = commits[commits['time_partition'] == int(fold)+1]
			test_commits = commits[commits['time_partition'] == int(fold)+2]
		else:
			train_commits = commits[commits['partition'] != int(fold)]
			val_commits = commits[commits['partition'] == int(fold)]
			test_commits = commits[commits['partition'] == int(fold)]
	# Load the inferred features
	if feature_type == 'manual':
		data = pd.read_parquet("inferred_features/manual.parquet")
		data = pd.merge(data, commits, how='left', on='commit')
		train = data[data['commit'].isin(train_commits['commit'])]
		x_train = train[man_features].astype(float).values
		validate = data[data['commit'].isin(val_commits['commit'])]
		x_validate = validate[man_features].astype(float).values
		test = data[data['commit'].isin(test_commits['commit'])]
		x_test = test[man_features].astype(float).values
	else:
		if 'cvss' in problem:
			data = pd.read_pickle(f"inferred_features/{feature_type}_{feature_scope}_{token}_multiclass_{fold}.pkl")
		elif 'combined' in problem:
			data = pd.read_pickle(f"inferred_features/{feature_type}_{feature_scope}_{token}_binary_{fold}.pkl")
		else:
			data = pd.read_pickle(f"inferred_features/{feature_type}_{feature_scope}_{token}_{problem}_{fold}.pkl")
		data = pd.merge(data, commits, how='left', on='commit')
		# Fill blank ast values
		if feature_type == 'ast' and token == 'bow':
			data['prev_features'] = data['prev_features'].apply(lambda x: coo_matrix((1, 10000), dtype=np.float64) if x is None else x)
			data['cur_features'] = data['cur_features'].apply(lambda x: coo_matrix((1, 10000), dtype=np.float64) if x is None else x)
			if feature_scope == 'hc':
				data['prev_context'] = data['prev_context'].apply(lambda x: coo_matrix((1, 10000), dtype=np.float64) if x is None else x)
				data['cur_context'] = data['cur_context'].apply(lambda x: coo_matrix((1, 10000), dtype=np.float64) if x is None else x)
		train = data[data['commit'].isin(train_commits['commit'])]
		validate = data[data['commit'].isin(val_commits['commit'])]
		test = data[data['commit'].isin(test_commits['commit'])]
		if token == 'bow':
			if feature_scope != 'hc':
				x_train = hstack([vstack(train['prev_features'].values), vstack(train['cur_features'].values)])
				x_validate = hstack([vstack(validate['prev_features'].values), vstack(validate['cur_features'].values)])
				x_test = hstack([vstack(test['prev_features'].values), vstack(test['cur_features'].values)])
			else:
				x_train = hstack([vstack(train['prev_features'].values), vstack(train['cur_features'].values), vstack(train['prev_context'].values), vstack(train['cur_context'].values)])
				x_validate = hstack([vstack(validate['prev_features'].values), vstack(validate['cur_features'].values), vstack(validate['prev_context'].values), vstack(validate['cur_context'].values)])
				x_test = hstack([vstack(test['prev_features'].values), vstack(test['cur_features'].values), vstack(test['prev_context'].values), vstack(test['cur_context'].values)])
			# XGB cannot handle coo_matrix
			x_train = x_train.tocsr()
			x_validate = x_validate.tocsr()
			x_test = x_test.tocsr()
		elif token == 'w2v':
			if feature_scope != 'hc':
				x_train = np.hstack([np.asarray(train['prev_features'].values.tolist()), np.asarray(train['cur_features'].values.tolist())])
				x_validate = np.hstack([np.asarray(validate['prev_features'].values.tolist()), np.asarray(validate['cur_features'].values.tolist())])
				x_test = np.hstack([np.asarray(test['prev_features'].values.tolist()), np.asarray(test['cur_features'].values.tolist())])
			else:
				x_train = np.hstack([np.asarray(train['prev_features'].values.tolist()), np.asarray(train['cur_features'].values.tolist()), np.asarray(train['prev_context'].values.tolist()), np.asarray(train['cur_context'].values.tolist())])
				x_validate = np.hstack([np.asarray(validate['prev_features'].values.tolist()), np.asarray(validate['cur_features'].values.tolist()), np.asarray(validate['prev_context'].values.tolist()), np.asarray(validate['cur_context'].values.tolist())])
				x_test = np.hstack([np.asarray(test['prev_features'].values.tolist()), np.asarray(test['cur_features'].values.tolist()), np.asarray(test['prev_context'].values.tolist()), np.asarray(test['cur_context'].values.tolist())])

	# Get the labels
	if problem == 'binary':
		train['cwe'] = np.where(train['cwe']=='-', 0, 1)
		validate['cwe'] = np.where(validate['cwe']=='-', 0, 1)
		test['cwe'] = np.where(test['cwe']=='-', 0, 1)
	if problem in ['binary', 'multiclass', 'combined']:
		y_train, y_validate, y_test = train['cwe'], validate['cwe'], test['cwe']
	elif 'cvss' in problem:
		y_train, y_validate, y_test = train[problem], validate[problem], test[problem]

	# Determine resampling
	if resampling == 'over+':
		# K_neighbours must be smaller than number of samples for smallest class. Skip when not true.
		smallest_sample = len(y_train[y_train == y_train.value_counts().index[-1]])
		orig_x_train, orig_y_train = x_train, y_train
		if smallest_sample <= 1: return
		elif smallest_sample <= 5: resample = ['over+_1']
		elif smallest_sample <= 10: resample = ['over+_1', 'over+_5']
		elif smallest_sample <= 15: resample = ['over+_1', 'over+_5', 'over+_10']
		elif smallest_sample <= 20: resample = ['over+_1', 'over+_5', 'over+_10', 'over+_15']
		else: resample = ['over+_1', 'over+_5', 'over+_10', 'over+_15', 'over+_20']
	else:
		resample = [resampling]

	# Apply resampling
	for sampling in resample:
		if sampling == 'over':
			sampler = RandomOverSampler(random_state=42)
			x_train, y_train = sampler.fit_resample(x_train, y_train)
		elif 'over+' in sampling:
			sampler = SMOTE(k_neighbors=int(sampling.split('_')[1]), random_state=42)
			x_train, y_train = sampler.fit_resample(orig_x_train, orig_y_train)

		# Run for each algorithm
		for alg in ['lr', 'svm', 'rf', 'knn', 'xgb', 'lgbm']:
			# Validation Step
			if 'validate' in process:
				# Define parameters to test
				if alg == 'lr':
					param_set = list(itertools.product(*[regularization_lr]))
				elif alg == 'svm':
					param_set = list(itertools.product(*[regularization_svm]))
				elif alg == 'knn':
					param_set = list(itertools.product(*[neighbours, weights, norms]))
				elif alg == 'rf' or alg == 'xgb' or alg == 'lgbm':
					param_set = list(itertools.product(*[estimators, leaf_nodes]))
				# Run for each parameter configuration
				for parameters in param_set:
					clf_settings = f"{problem},{fold},{feature_type},{feature_scope},{token},{alg},{'-'.join(parameters)}"
					multiclass = False if len(np.unique(y_train)) else True
					# Get and evaluate the classifier
					clf = get_classifier(alg, multiclass, *parameters)
					evaluate(clf, x_train, y_train, x_validate, y_validate, clf_settings, result_file, sampling, write=True)
			# Test step
			elif 'test' in process:
				parameters = get_best_classifier(problem,feature_type,feature_scope,token,alg,fold, sampling)
				if not parameters: continue		# Skip if no best model
				clf_settings = f"{problem},{fold},{feature_type},{feature_scope},{token},{alg},{parameters}"
				print(clf_settings, sampling)
				# # Concatenate train and validate set.
				# if token == 'bow':
				# 	x_train = vstack([x_train, x_validate], format='csr')
				# else:
				# 	x_train = np.concatenate((x_train, x_validate), axis=0)
				# y_train = np.concatenate((y_train, y_validate), axis=0)
				# Get and evaluate the classifier
				multiclass = False if len(np.unique(y_train)) else True
				clf = get_classifier(alg, multiclass, *parameters.split('-'))
				evaluate(clf, x_train, y_train, x_test, y_test, clf_settings, result_file, sampling, write=True)

if __name__ == '__main__':
	print(sys.argv)
	main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])
	print("Completed")
	exit()
