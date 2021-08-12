# Extra evaluation functions for additional ML tasks
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from cvss import CVSS2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc, matthews_corrcoef
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from scipy.sparse import hstack, vstack, coo_matrix

# Settings
time_partition = True	# Use time-based folds

# Manual features
man_features = ['STARS', 'COMMITS', 'NS', 'LA', 'LD', 'LT', 'NDEV', 'AGE', 'NUC', 'EXP', 'NRF', 'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', 'continue', 'const', 'default', 'do', 'double', 'else', 'enum', 'exports', 'extends', 'false', 'final', 'finally', 'float', 'for', 'goto', 'if', 'implements', 'import', 'instanceof', 'int', 'interface', 'long', 'module', 'native', 'new', 'null', 'package', 'private', 'protected', 'public', 'requires', 'return', 'short', 'static', 'strictfp', 'super', 'switch', 'synchronized', 'this', 'throw', 'throws', 'transient', 'true', 'try', 'var', 'void', 'volatile', 'while']

# Classift CVSS score severity
def severity(score):
	if 0 <= score < 4: return('LOW')
	elif 4 <= score < 7: return('Medium')
	elif 7 <= score <= 10: return('High')
	else: return 'Eh?'

# Get the optimal classifier from the holdout validation process. Return the hyperparameters
def get_best_classifier(problem, feature_type, feature_scope, token, alg, fold, sampling):
	# Load the validation results
	if problem in ['multiclass', 'binary']: results = pd.read_csv("ml_results/validate.csv")
	elif problem == 'combined': results = pd.read_csv("ml_results/combined_validate.csv")
	elif 'cvss' in problem: results = pd.read_csv("ml_results/cvss_validate.csv")
	results = results[(results['problem'] == problem) & (results['feature_type'] == feature_type) & (results['feature_scope'] == feature_scope) & (results['token'] == token) & (results['classifier'] == alg) & (results['partition'] == int(fold)) & (results['resampling'] == sampling)].reset_index()
	# Check empty
	if results.empty:
		return False
	# Get the best model
	test_result = results.iloc[results['mcc'].idxmax()]
	return test_result['parameters']

# Get results for the multiclass case using the best configuration for the binary case.
def transfer_folds():
	# Open the results file
	if not os.path.exists("ml_results/transfer.csv"):
		outfile = open("ml_results/transfer.csv", 'w')
		outfile.write("problem,partition,feature_type,feature_scope,token,classifier,parameters,accuracy,precision,recall,gmean,f1,mcc,f1_difference,mcc_difference,mc_mcc_variance,mc_f1_variance,binary_mcc_variance,binary_f1_variance\n")
	else:
		outfile = open("ml_results/transfer.csv", 'a')
	# Load the validation results
	all_results = pd.read_csv("ml_results/validate.csv")

	for feature_type in ['code', 'ast', 'manual']:
		for feature_scope in ['hunk', 'ss', 'hc']:
				for token in ['bow', 'w2v']:
					for alg in ['lr', 'knn', 'svm', 'rf', 'lgbm', 'xgb', 'all']:
						if feature_type == 'manual':
							feature_scope, token = '-', '-'
						if alg != 'all':
							results = all_results[(all_results['feature_type'] == feature_type) & (all_results['feature_scope'] == feature_scope) & (all_results['token'] == token) & (all_results['classifier'] == alg) & (all_results['partition'] != 'holdout')]
						else:
							results = all_results[(all_results['feature_type'] == feature_type) & (all_results['feature_scope'] == feature_scope) & (all_results['token'] == token) & (all_results['partition'] != 'holdout')]
						results['partition'] = results['partition'].astype(int)	# Average partitions
						group_features = ['problem', 'parameters', 'feature_type', 'feature_scope', 'token', 'classifier']	# Features to average the folds by
						results = results.groupby(group_features).agg({i: 'mean' for i in results.columns if i not in group_features}).reset_index()
						# Get best fold averages
						binary_result = results.iloc[results[results['problem'] == 'binary']['mcc'].idxmax()]
						mc_result = results.iloc[results[results['problem'] == 'multiclass']['mcc'].idxmax()]
						# Get worst fold averages, for variance
						binary_result_worst = results.iloc[results[results['problem'] == 'binary']['mcc'].idxmin()]
						mc_result_worst = results.iloc[results[results['problem'] == 'multiclass']['mcc'].idxmin()]
						# Get transfer result
						if alg != 'all':
							transfer_result = results[(results['problem'] == 'multiclass') & (results['parameters'] == binary_result['parameters'])]
						else:
							transfer_result = results[(results['problem'] == 'multiclass') & (results['classifier'] == binary_result['classifier']) & (results['parameters'] == binary_result['parameters'])]
						# Calculate transfer difference
						transfer_result['f1_diff'] = transfer_result['f1'] - mc_result['f1']
						transfer_result['mcc_diff'] = transfer_result['mcc'] - mc_result['mcc']
						# Calculate hyperparameter variance
						transfer_result['binary_f1_variance'] = binary_result['f1'] - binary_result_worst['f1']
						transfer_result['binary_mcc_variance'] = binary_result['mcc'] - binary_result_worst['mcc']
						transfer_result['mc_f1_variance'] = mc_result['f1'] - mc_result_worst['f1']
						transfer_result['mc_mcc_variance'] = mc_result['mcc'] - mc_result_worst['mcc']
						# Fix column names
						if alg == 'all': transfer_result['classifier'] = 'all_(' + transfer_result['classifier'] + ')'
						transfer_result = transfer_result.iloc[0]
						# Re-order and save
						transfer_result = transfer_result[['problem', 'partition', 'feature_type', 'feature_scope', 'token', 'classifier', 'parameters', 'accuracy', 'precision', 'recall', 'gmean', 'f1', 'mcc', 'f1_diff', 'mcc_diff', 'mc_mcc_variance', 'mc_f1_variance', 'binary_mcc_variance', 'binary_f1_variance']]
						outfile.write(','.join([str(round(x, 4)) if isinstance(x, float) else x for x in transfer_result.tolist()])+'\n')
					# Break early for manual
					if feature_type == 'manual':
						return

# Compare the optimal classifier for each task combination (considering ML, DL and all)
def optimal_classifiers():
	all_results = pd.read_csv("ml_results/vcc_results_all.csv")
	# Remove 'best' columns
	all_results = all_results[[c for c in all_results.columns if 'best' not in c.lower()]]
	# Convert to long format
	columns = all_results.columns.tolist()
	id_cols = ['Task', 'Feature type', 'Feature scope', 'Token']
	var_cols = [x for x in columns if x not in id_cols]
	all_results = pd.melt(all_results, id_vars=id_cols, value_vars=var_cols, var_name='Classifier', value_name='MCC')
	# Only keep real values
	all_results = all_results[all_results['MCC'] != '-']
	all_results['MCC'] = all_results['MCC'].astype(float)
	# Get ML and DL models
	ml_classifiers = ['LR', 'KNN', 'SVM', 'RF', 'LGBM', 'XGB']
	ml_results = all_results[all_results['Classifier'].isin(ml_classifiers)].reset_index(drop=True)
	# dl_classifiers = ['MLP (best)', 'MLP (David)', 'MLP (Triet)', 'Sequential CNN (best)', 'Sequential CNN', 'Sequential RNN (best)', 'Sequential RNN', 'Siamese CNN (best)', 'Siamese CNN', 'Siamese (MLP-Best)', 'Siamese (MLP)']
	dl_classifiers = ['MLP (Triet)', 'Sequential CNN', 'Sequential RNN', 'Siamese CNN', 'Siamese (MLP)']
	dl_results = all_results[all_results['Classifier'].isin(dl_classifiers)].reset_index(drop=True)
	# Group task combinations
	ml = ml_results.loc[ml_results.groupby(id_cols)['MCC'].idxmax()]
	dl = dl_results.loc[dl_results.groupby(id_cols)['MCC'].idxmax()]
	all = all_results.loc[all_results.groupby(id_cols)['MCC'].idxmax()].rename(columns={'Classifier': 'Classifier_all', 'MCC': 'MCC_all'})
	merged = pd.merge(ml, dl, how='left', on=id_cols, suffixes=('_ml', '_dl'))
	merged = merged.merge(all, how='left', on=id_cols)
	# Compare optimal classifiers
	merged['ml_same'] = 'No'
	merged['dl_same'] = 'No'
	merged['all_same'] = 'No'
	for index, row in merged.iterrows():
		split = 13 if index < len(merged)/2 else -13
		if merged.at[index, 'Classifier_ml'] == merged.at[index+split, 'Classifier_ml']: merged.at[index, 'ml_same'] = 'Yes'
		if merged.at[index, 'Classifier_dl'] == merged.at[index+split, 'Classifier_dl']: merged.at[index, 'dl_same'] = 'Yes'
		if merged.at[index, 'Classifier_all'] == merged.at[index+split, 'Classifier_all']: merged.at[index, 'all_same'] = 'Yes'
	merged.to_csv('ml_results/best_classifiers.csv', index=False)

	# Get the transfer and variance in performance for MC, DL and ALL classifiers
	result_transfer = pd.DataFrame(columns=['Feature type', 'Feature scope', 'Token', 'Classifier', 'ML Max', 'ML Transfer', 'DL Max', 'DL Transfer', 'All Max', 'All Transfer'])
	for feature_type in ['Code', 'AST', 'Manual']:
		for feature_scope in ['Hunk only', 'Smallest scope', 'Hunk Context']:
				for token in ['BoW', 'W2V']:
					if feature_type == 'Manual':
						feature_scope, token = '-', '-'
					# ML
					ml_bin = ml_results.loc[ml_results[(ml_results['Task'] == 'Binary') & (ml_results['Feature type'] == feature_type) & (ml_results['Feature scope'] == feature_scope) & (ml_results['Token'] == token)]['MCC'].idxmax()]
					ml_max = ml_results.loc[ml_results[(ml_results['Task'] == 'Multiclass') & (ml_results['Feature type'] == feature_type) & (ml_results['Feature scope'] == feature_scope) & (ml_results['Token'] == token)]['MCC'].idxmax()]
					ml_transfer = ml_results[(ml_results['Task'] == 'Multiclass') & (ml_results['Feature type'] == ml_bin['Feature type']) & (ml_results['Feature scope'] == ml_bin['Feature scope']) & (ml_results['Token'] == ml_bin['Token']) & (ml_results['Classifier'] == ml_bin['Classifier'])]
					# DL
					dl_bin = dl_results.loc[dl_results[(dl_results['Task'] == 'Binary') & (dl_results['Feature type'] == feature_type) & (dl_results['Feature scope'] == feature_scope) & (dl_results['Token'] == token)]['MCC'].idxmax()]
					dl_max = dl_results.loc[dl_results[(dl_results['Task'] == 'Multiclass') & (dl_results['Feature type'] == feature_type) & (dl_results['Feature scope'] == feature_scope) & (dl_results['Token'] == token)]['MCC'].idxmax()]
					dl_transfer = dl_results[(dl_results['Task'] == 'Multiclass') & (dl_results['Feature type'] == dl_bin['Feature type']) & (dl_results['Feature scope'] == dl_bin['Feature scope']) & (dl_results['Token'] == dl_bin['Token']) & (dl_results['Classifier'] == dl_bin['Classifier'])]
					# ALL
					all_bin = all_results.loc[all_results[(all_results['Task'] == 'Binary') & (all_results['Feature type'] == feature_type) & (all_results['Feature scope'] == feature_scope) & (all_results['Token'] == token)]['MCC'].idxmax()]
					all_max = all_results.loc[all_results[(all_results['Task'] == 'Multiclass') & (all_results['Feature type'] == feature_type) & (all_results['Feature scope'] == feature_scope) & (all_results['Token'] == token)]['MCC'].idxmax()]
					all_transfer = all_results[(all_results['Task'] == 'Multiclass') & (all_results['Feature type'] == all_bin['Feature type']) & (all_results['Feature scope'] == all_bin['Feature scope']) & (all_results['Token'] == all_bin['Token']) & (all_results['Classifier'] == all_bin['Classifier'])]
					# Append
					result = [feature_type, feature_scope, token, ml_bin['Classifier'], ml_max['MCC'], ml_transfer['MCC'].values[0], dl_max['MCC'], dl_transfer['MCC'].values[0], all_max['MCC'], all_transfer['MCC'].values[0]]
					result_transfer.loc[len(result_transfer)] = result
					if feature_type == 'Manual':
						break
				if feature_type == 'Manual':
					break
	result_transfer.to_csv('ml_results/transfer_all.csv', index=False)

# Re-evaluate the one-task classifier (cvss and binary vulnerability prediction) as either a binary or multiclass classifier.
def evaluate_one_task(feature_type, feature_scope, token):
	# Load the one-task test results
	onetask_results = pd.read_csv("ml_results/combined_test.csv")
	# Load the normal test results
	norm_results = pd.read_csv("ml_results/test.csv")
	compare_results = pd.DataFrame(columns=['comparison', 'partition', 'feature_type', 'feature_scope', 'token', 'classifier', 'parameters', 'f1', 'f1_diff', 'mcc', 'mcc_diff'])

	# Compare to binary and multiclass scenario
	for comparison in ['binary', 'multiclass']:
		# Load the commit map
		commits = pd.read_csv("binary_map.csv")
		# Get labels
		if comparison == 'binary':
			commits['cwe'] = np.where(commits['cwe']=='-', 0, 1)

		for fold in [str(x) for x in range(10)]:
			train_commits = commits[commits['partition'] != int(fold)]
			test_commits = commits[commits['partition'] == int(fold)]
			if comparison == 'multiclass':
				train_commits = train_commits[train_commits['cwe'] != '-']
				test_commits = test_commits[test_commits['cwe'] != '-']

			# Load the inferred features
			if feature_type == 'manual':
				data = pd.read_parquet("inferred_features/manual.parquet")
				data = pd.merge(data, commits, how='left', on='commit')
				train = data[data['commit'].isin(train_commits['commit'])]
				x_train = train[man_features].astype(float).values
				test = data[data['commit'].isin(test_commits['commit'])]
				x_test = test[man_features].astype(float).values
			else:
				data = pd.read_pickle(f"inferred_features/{feature_type}_{feature_scope}_{token}_binary_{fold}.pkl")
				data = pd.merge(data, commits, how='left', on='commit')
				# Fill blank ast values
				if feature_type == 'ast' and token == 'bow':
					data['prev_features'] = data['prev_features'].apply(lambda x: coo_matrix((1, 10000), dtype=np.float64) if x is None else x)
					data['cur_features'] = data['cur_features'].apply(lambda x: coo_matrix((1, 10000), dtype=np.float64) if x is None else x)
					if feature_scope == 'hc':
						data['prev_context'] = data['prev_context'].apply(lambda x: coo_matrix((1, 10000), dtype=np.float64) if x is None else x)
						data['cur_context'] = data['cur_context'].apply(lambda x: coo_matrix((1, 10000), dtype=np.float64) if x is None else x)
				train = data[data['commit'].isin(train_commits['commit'])]
				test = data[data['commit'].isin(test_commits['commit'])]
				if token == 'bow':
					if feature_scope != 'hc':
						x_train = hstack([vstack(train['prev_features'].values), vstack(train['cur_features'].values)])
						x_test = hstack([vstack(test['prev_features'].values), vstack(test['cur_features'].values)])
					else:
						x_train = hstack([vstack(train['prev_features'].values), vstack(train['cur_features'].values), vstack(train['prev_context'].values), vstack(train['cur_context'].values)])
						x_test = hstack([vstack(test['prev_features'].values), vstack(test['cur_features'].values), vstack(test['prev_context'].values), vstack(test['cur_context'].values)])
					# XGB cannot handle coo_matrix
					x_train = x_train.tocsr()
					x_test = x_test.tocsr()
				elif token == 'w2v':
					if feature_scope != 'hc':
						x_train = np.hstack([np.asarray(train['prev_features'].values.tolist()), np.asarray(train['cur_features'].values.tolist())])
						x_test = np.hstack([np.asarray(test['prev_features'].values.tolist()), np.asarray(test['cur_features'].values.tolist())])
					else:
						x_train = np.hstack([np.asarray(train['prev_features'].values.tolist()), np.asarray(train['cur_features'].values.tolist()), np.asarray(train['prev_context'].values.tolist()), np.asarray(train['cur_context'].values.tolist())])
						x_test = np.hstack([np.asarray(test['prev_features'].values.tolist()), np.asarray(test['cur_features'].values.tolist()), np.asarray(test['prev_context'].values.tolist()), np.asarray(test['cur_context'].values.tolist())])
			y_train = train['cwe']
			y_test = test['cwe']	# Get labels

			for alg in ['lr', 'knn', 'svm', 'rf', 'lgbm', 'xgb']:
				if feature_type == 'manual':
					feature_scope, token = '-', '-'

				# Get the optimal model settings
				parameters = onetask_results[(onetask_results['feature_type'] == feature_type) & (onetask_results['feature_scope'] == feature_scope) & (onetask_results['token'] == token) & (onetask_results['classifier'] == alg)][['parameters']].values[0][0]
				# Load the prediction model
				clf_settings = f"combined,{fold},{feature_type},{feature_scope},{token},{alg},{parameters}"
				try:
					clf = pickle.load(open(f"prediction_models/{clf_settings.replace(',','_')}.model", "rb"))
				except:
					continue
				y_pred = clf.predict(x_test)	# Predict
				if comparison == 'binary':	y_pred = [0 if x == '-' else 1 for x in y_pred]
				# Evaluate
				if comparison == 'binary':
					f1 = f1_score(y_test, y_pred, average='binary')
				elif comparison == 'multiclass':
					f1 = f1_score(y_test, y_pred, average='weighted')
				mcc = matthews_corrcoef(y_test, y_pred)
				# Get the original values for comparison
				f1_orig = norm_results[(norm_results['problem'] == comparison) & (norm_results['feature_type'] == feature_type) & (norm_results['feature_scope'] == feature_scope) & (norm_results['token'] == token) & (norm_results['classifier'] == alg)][['f1']].values[0][0]
				mcc_orig = norm_results[(norm_results['problem'] == comparison) & (norm_results['feature_type'] == feature_type) & (norm_results['feature_scope'] == feature_scope) & (norm_results['token'] == token) & (norm_results['classifier'] == alg)][['mcc']].values[0][0]
				print(clf_settings)
				# Re-run for multiclass on the binary folds
				if comparison == 'multiclass':
					# Load the multiclass prediction model
					mc_parameters = norm_results[(norm_results['problem'] == comparison) & (norm_results['feature_type'] == feature_type) & (norm_results['feature_scope'] == feature_scope) & (norm_results['token'] == token) & (norm_results['classifier'] == alg)][['parameters']].values[0][0]
					clf_settings = f"multiclass,{fold},{feature_type},{feature_scope},{token},{alg},{parameters}"
					# mc_clf = pickle.load(open(f"prediction_models/{clf_settings.replace(',','_')}.model", "rb"))
					# Retrain the model
					mc_parameters = mc_parameters.split('-')
					workers = 3
					if alg == 'lr':
						mc_clf = LogisticRegression(C=float(mc_parameters[0]), multi_class='multinomial', n_jobs=workers, solver='lbfgs', tol=0.001, max_iter=1000, random_state=42)
					elif alg == 'svm':
						mc_clf = SVC(random_state=42, C=float(mc_parameters[0]), kernel='rbf', max_iter=-1)
					elif alg == 'knn':
						mc_clf = KNeighborsClassifier(n_neighbors=int(mc_parameters[0]), weights=mc_parameters[1], p=int(mc_parameters[2]), n_jobs=workers)
					elif alg == 'rf':
						mc_clf = RandomForestClassifier(n_estimators=int(mc_parameters[0]), max_depth=None, max_leaf_nodes=int(mc_parameters[1]), random_state=42, n_jobs=workers)
					elif alg == 'xgb':
						mc_clf = XGBClassifier(objective='reg:squarederror', max_depth=0, n_estimators=int(mc_parameters[0]), max_leaves=int(mc_parameters[1]), grow_policy='lossguide', n_jobs=workers, random_state=42, tree_method='hist')
					elif alg == 'lgbm':
						mc_clf = LGBMClassifier(n_estimators=int(mc_parameters[0]), num_leaves=int(mc_parameters[1]), max_depth=-1, objective='multiclass', n_jobs=workers, random_state=42)
					mc_clf.fit(x_train, y_train)

					y_pred = mc_clf.predict(x_test)	# Predict
					f1_orig = f1_score(y_test, y_pred, average='weighted')
					mcc_orig = matthews_corrcoef(y_test, y_pred)
				f1_diff = f1 - f1_orig
				mcc_diff = mcc - mcc_orig
				compare = [comparison, fold, feature_type, feature_scope, token, alg, parameters, f1, f1_diff, mcc, mcc_diff]
				compare_results.loc[len(compare_results)] = compare

	# Average the folds
	compare_results['partition'] = compare_results['partition'].astype(int)	# Average partitions
	group_features = ['comparison', 'parameters', 'feature_type', 'feature_scope', 'token', 'classifier']	# Features to average the folds by
	compare_results = compare_results.groupby(group_features).agg({i: 'mean' for i in compare_results.columns if i not in group_features}).reset_index()
	print(compare_results)
	# Re-order
	compare_results_mc = compare_results[compare_results['comparison'] == 'multiclass'].rename(columns={'f1': 'mc_f1', 'f1_diff': 'mc_f1_diff', 'mcc': 'mc_mcc', 'mcc_diff': 'mc_mcc_diff'})
	compare_results_binary = compare_results[compare_results['comparison'] == 'binary'].rename(columns={'f1': 'binary_f1', 'f1_diff': 'binary_f1_diff', 'mcc': 'binary_mcc', 'mcc_diff': 'binary_mcc_diff'})
	onetask_results = pd.merge(onetask_results, compare_results_mc, how='right', on=['parameters', 'feature_type', 'feature_scope', 'token', 'classifier'])
	onetask_results = pd.merge(onetask_results, compare_results_binary, how='right', on=['parameters', 'feature_type', 'feature_scope', 'token', 'classifier'])
	onetask_results = onetask_results[["partition", "feature_type", "feature_scope", "token", "classifier", "parameters", "f1", "mcc", "train_time", "binary_f1", "binary_mcc", "binary_f1_diff", "binary_mcc_diff", "mc_f1", "mc_mcc", "mc_f1_diff", "mc_mcc_diff"]]
	onetask_results[["f1", "mcc", "train_time", "binary_f1", "binary_mcc", "binary_f1_diff", "binary_mcc_diff", "mc_f1", "mc_mcc", "mc_f1_diff", "mc_mcc_diff"]] = onetask_results[["f1", "mcc", "train_time", "binary_f1", "binary_mcc", "binary_f1_diff", "binary_mcc_diff", "mc_f1", "mc_mcc", "mc_f1_diff", "mc_mcc_diff"]].round(4)
	# Open the results file
	if not os.path.exists("ml_results/onetask.csv"):
		onetask_results.to_csv("ml_results/onetask.csv", index=False)
	else:
		onetask_results.to_csv("ml_results/onetask.csv", mode='a', index=False, header=False)


# Compare the individual cvss tasks to the extreme multiclass (all cvss values in one) vector prediction
# Partition = 'holdout' or 'folds'
def compare_cvss_vector(partition):
	# Load the CVSS results
	norm_results = pd.read_csv("ml_results/cvss_test.csv")
	# Overall f1/mcc, score of the individual vector evaluation ('individual'), score of the individual multiclass task ('task'). Individual is value we want.
	compare_results = pd.DataFrame(columns=['comparison', 'partition', 'feature_type', 'feature_scope', 'token', 'resampling', 'classifier', 'f1', 'f1_individual', 'f1_task', 'mcc', 'mcc_individual', 'mcc_task'])
	commits = pd.read_csv("cvss_map.csv")

	# Setting to iterate through
	feature_types = ['code', 'manual']
	feature_scopes = ['hc']
	tokens = ['bow', 'w2v']

	for feature_type in feature_types:
		for feature_scope in feature_scopes:
			for token in tokens:
				if feature_type == 'manual':
					feature_scope, token = '-', '-'
				print(feature_type, feature_scope, token)
				partitions = ['holdout'] if partition == 'holdout' else range(10)
				for fold in [str(x) for x in partitions]:
					# Get the test set
					if partition == 'holdout':
						test_commits = commits[commits['set'] == 'test']
					else:
						if time_partition:
							test_commits = commits[commits['time_partition'] == int(fold)+2]
						else:
							test_commits = commits[commits['partition'] == int(fold)]
					# Load the inferred features
					if feature_type == 'manual':
						data = pd.read_parquet("inferred_features/manual.parquet")
						data = pd.merge(data, commits, how='left', on='commit')
						test = data[data['commit'].isin(test_commits['commit'])]
						x_test = test[man_features].astype(float).values
					else:
						data = pd.read_pickle(f"inferred_features/{feature_type}_{feature_scope}_{token}_multiclass_{fold}.pkl")
						data = pd.merge(data, commits, how='left', on='commit')
						# Fill blank ast values
						if feature_type == 'ast' and token == 'bow':
							data['prev_features'] = data['prev_features'].apply(lambda x: coo_matrix((1, 10000), dtype=np.float64) if x is None else x)
							data['cur_features'] = data['cur_features'].apply(lambda x: coo_matrix((1, 10000), dtype=np.float64) if x is None else x)
							if feature_scope == 'hc':
								data['prev_context'] = data['prev_context'].apply(lambda x: coo_matrix((1, 10000), dtype=np.float64) if x is None else x)
								data['cur_context'] = data['cur_context'].apply(lambda x: coo_matrix((1, 10000), dtype=np.float64) if x is None else x)
						test = data[data['commit'].isin(test_commits['commit'])]
						if token == 'bow':
							if feature_scope != 'hc':
								x_test = hstack([vstack(test['prev_features'].values), vstack(test['cur_features'].values)])
							else:
								x_test = hstack([vstack(test['prev_features'].values), vstack(test['cur_features'].values), vstack(test['prev_context'].values), vstack(test['cur_context'].values)])
							# XGB cannot handle coo_matrix
							x_test = x_test.tocsr()
						elif token == 'w2v':
							if feature_scope != 'hc':
								x_test = np.hstack([np.asarray(test['prev_features'].values.tolist()), np.asarray(test['cur_features'].values.tolist())])
							else:
								x_test = np.hstack([np.asarray(test['prev_features'].values.tolist()), np.asarray(test['cur_features'].values.tolist()), np.asarray(test['prev_context'].values.tolist()), np.asarray(test['cur_context'].values.tolist())])
					# For each sampling method
					for sampling in ['none', 'over']:
						# For each classifier algorithm
						for alg in ['lr', 'svm', 'rf', 'knn', 'xgb', 'lgbm']:
							# Get the labels
							y_test = test['cvss2_vector']
							# Get the optimal classifier parameters
							vector_results = norm_results[(norm_results['problem'] == 'cvss2_vector') & (norm_results['feature_type'] == feature_type) & (norm_results['feature_scope'] == feature_scope) & (norm_results['token'] == token) & (norm_results['classifier'] == alg) & (norm_results['resampling'] == sampling) & (norm_results['partition'] == int(fold))]
							print(vector_results)
							parameters = vector_results[['parameters']].values[0][0]
							print(parameters)
							# Load the classifier
							clf_settings = f"cvss2_vector,{fold},{feature_type},{feature_scope},{token},{alg},{parameters}"
							if sampling == 'none':
								clf = pickle.load(open(f"prediction_models/test/{clf_settings.replace(',','_')}.model", "rb"))
							else:
								clf = pickle.load(open(f"prediction_models/test/{clf_settings.replace(',','_')}{sampling}.model", "rb"))
							# Make vector predictions
							y_pred = clf.predict(x_test)
							# Add severity to the vector
							y_test = [i + '/' + severity(CVSS2(i).scores()[0]) for i in y_test]
							y_pred = [i + '/' + severity(CVSS2(i).scores()[0]) for i in y_pred]
							# Consider each individual task
							for count, problem in enumerate(['cvss2_' + i for i in ['accessvect', 'accesscomp', 'auth', 'conf', 'integrity', 'avail', 'severity']]):
								# Get the optimal results for the individual task
								task_results = norm_results[(norm_results['problem'] == problem) & (norm_results['feature_type'] == feature_type) & (norm_results['feature_scope'] == feature_scope) & (norm_results['token'] == token) & (norm_results['classifier'] == alg) & (norm_results['resampling'] == sampling) & (norm_results['partition'] == int(fold))]
								if task_results.empty:
									task_results = pd.DataFrame(data={'f1': [0], 'mcc': [0]})
								print(task_results)
								# Only considering the relevant part of the vector prediction.
								task_test = [i.split('/')[count] for i in y_test]
								task_pred = [i.split('/')[count] for i in y_pred]
								# Individual evaluation
								f1_ind = f1_score(task_test, task_pred, average='macro')
								mcc_ind = matthews_corrcoef(task_test, task_pred)
								# Save results for comparison
								result_values = [problem, fold, feature_type, feature_scope, token, sampling, alg, round(vector_results['f1'].values[0],3), round(f1_ind,3), round(task_results['f1'].values[0],3), round(vector_results['mcc'].values[0],3), round(mcc_ind,3), round(task_results['mcc'].values[0],3)]
								compare_results.loc[len(compare_results)] = result_values
				if feature_type == 'manual':
					break
			if feature_type == 'manual':
				break
	if partition == 'holdout':
		compare_results.to_csv("ml_results/cvss_extreme_holdout.csv", index=False)
	if partition == 'folds':
		# Average the folds
		print(compare_results)
		compare_results.to_csv("ml_results/cvss_extreme_folds_all.csv", index=False)
		group_features = ['comparison', 'feature_type', 'feature_scope', 'token', 'resampling', 'classifier']	# Features to average the folds by
		for x in [i for i in compare_results.columns if i not in group_features]:
			compare_results[x] = compare_results[x].astype(float)	# Average
		compare_results = compare_results.groupby(group_features).agg({i: 'mean' for i in compare_results.columns if i not in group_features}).reset_index()
		compare_results = compare_results.round(3)
		compare_results.to_csv("ml_results/cvss_extreme_folds.csv", index=False)

if __name__ == '__main__':
	compare_cvss_vector('folds')
	exit()
