# Partition the data into time based holdout, and k-folds splits
import pandas as pd
import numpy as np
from glob import glob
from pathlib import Path

path = 'infer_features/'
outpath = ''

# Read and concatenate all dataframes in a folder
def read_data(path, p, csv=True):
	if csv:
		return pd.concat([pd.read_csv(i) for i in glob(path + p)]).drop_duplicates().dropna(how='all')
	return pd.concat([pd.read_parquet(i) for i in glob(path + p)]).drop_duplicates().dropna(how='all')

# Make directory if not present
def makedir(directory):
	Path(directory).mkdir(parents=True, exist_ok=True)

# Partition the parsed data into time-based train/val/test split
# Problem = 'mc' or 'binary'
def partition_data(problem, k=10):
	if problem == 'mc':
		data = pd.read_csv(path+"../extract_vcc/VCC/filtered/java_vccs.csv")
	elif problem == 'binary':
		data = pd.concat([pd.read_csv(path+"../extract_vcc/VCC/filtered/java_vccs.csv"), pd.read_csv(path+"../extract_vcc/VCC/filtered/java_non_vccs.csv")])
	elif problem == 'cvss':
		data = pd.read_csv(path+"../extract_vcc/VCC/filtered/java_vccs.csv")
		# Append CVSS data
		nvd = pd.read_csv(path+"../extract_data/data/nvd_data.csv")
		nvd = nvd.rename(columns={'CVSS2_Vectors': 'CVSS2_Vector', 'CVSS2_BScore': 'CVSS2_BaseScore'})[['ID', 'CVSS2_Vector', 'CVSS2_AccessVect', 'CVSS2_AccessComp', 'CVSS2_Auth', 'CVSS2_Conf', 'CVSS2_Integrity', 'CVSS2_Avail', 'CVSS2_BaseScore', 'CVSS2_Severity']]
		data = pd.merge(data, nvd, how='left', on='ID')

	### HOLDOUT ###
	# Sort the data by time
	data['Date'] = pd.to_datetime(data.Date, utc=True, errors='ignore')
	data = data.sort_values(by=['Date'])
	# Determine split point for 80/10/10 based on commits (VCCs)
	unique_commits = pd.unique(data['Commit'].values)
	split_point_1 = int(0.8 * len(unique_commits))
	split_point_2 = int(0.9 * len(unique_commits))
	train_commits, val_commits, test_commits = unique_commits[:split_point_1], unique_commits[split_point_1:split_point_2], unique_commits[split_point_2:]
	# Split the data
	def split_data(row):
		if row['Commit'] in train_commits:
			return 'train'
		elif row['Commit'] in val_commits:
			return 'val'
		else:
			return 'test'
	data['Set'] = data.apply(split_data, axis=1)

	### TIME FOLDS ###
	# Split the data
	time_split_data = pd.DataFrame()
	step_sz = 1/12
	for i in range(12):
		# Determine split points
		split_point_1, split_point_2 = int((i*step_sz) * len(unique_commits)), int(((i+1)*step_sz) * len(unique_commits))
		split_commits = unique_commits[split_point_1:split_point_2]
		# Split
		split = data.loc[data['Commit'].isin(split_commits)]
		split['Time_Partition'] = i
		time_split_data = pd.concat([time_split_data, split])
	time_split_data.columns = time_split_data.columns.str.lower()

	### FOLDS ###
	# Random Shuffle
	np.random.seed(42)
	np.random.shuffle(unique_commits)
	# Split the data
	split_data = pd.DataFrame()
	step_sz = 1/k
	for i in range(k):
		# Determine split points
		split_point_1, split_point_2 = int((i*step_sz) * len(unique_commits)), int(((i+1)*step_sz) * len(unique_commits))
		split_commits = unique_commits[split_point_1:split_point_2]
		# Split
		split = data.loc[data['Commit'].isin(split_commits)]
		split['Partition'] = i
		split_data = pd.concat([split_data, split])
	split_data.columns = split_data.columns.str.lower()
	print(split_data)

	# Save partitions
	if problem in ['binary', 'mc']:
		split_data = time_split_data.merge(split_data, how='left', on=['commit', 'date', 'cwe', 'set'])
		split_data = split_data[['commit', 'date', 'cwe', 'set', 'partition', 'time_partition']]
	elif problem == 'cvss':
		split_data = time_split_data.merge(split_data, how='left', on=['commit', 'date', 'cvss2_vector', 'cvss2_accessvect', 'cvss2_accesscomp', 'cvss2_auth', 'cvss2_conf', 'cvss2_integrity', 'cvss2_avail', 'cvss2_basescore', 'cvss2_severity', 'set'])
		split_data = split_data[['commit', 'date', 'cvss2_vector', 'cvss2_accessvect', 'cvss2_accesscomp', 'cvss2_auth', 'cvss2_conf', 'cvss2_integrity', 'cvss2_avail', 'cvss2_basescore', 'cvss2_severity', 'set', 'partition', 'time_partition']]
	split_data.to_csv(outpath+problem+"_map.csv", index=False)

# Collect the different raw feature sets ('files', 'changes', 'scopes', 'contexts')
def get_raw_features():
	# Get commit list
	commits = pd.read_csv(outpath+"binary_map.csv")
	# For each feature set
	for feature_set in ['files', 'changes', 'scopes', 'contexts']:
		# Read the feature data
		commit_features = read_data(path+'../feature_extraction/'+feature_set+'/', '*')
		if feature_set == 'files':
			commit_features['Prev_Data'] = commit_features['Prev_Data'].fillna('').astype('str')
			commit_features['Cur_Data'] = commit_features['Cur_Data'].fillna('').astype('str')
			commit_features = commit_features[['Commit', 'Prev_Data', 'Cur_Data']]
			commit_features = commit_features.rename(columns={'Commit': 'commit', 'Prev_Data': 'Prev_File', 'Cur_Data': 'Cur_File'})
		else:
			commit_features['Prev_Changes'] = commit_features['Prev_Changes'].fillna('').astype('str')
			commit_features['Cur_Changes'] = commit_features['Cur_Changes'].fillna('').astype('str')
			if feature_set == 'contexts':
				commit_features['Prev_Context'] = commit_features['Prev_Context'].fillna('').astype('str')
				commit_features['Cur_Context'] = commit_features['Cur_Context'].fillna('').astype('str')
				commit_features = commit_features[['Commit', 'Prev_Changes', 'Cur_Changes', 'Prev_Context', 'Cur_Context']]
			else:
				commit_features = commit_features[['Commit', 'Prev_Changes', 'Cur_Changes']]
			commit_features = commit_features.rename(columns={'Commit': 'commit', 'Prev_Changes': 'Prev_Data', 'Cur_Changes': 'Cur_Data'})
			commit_features = commit_features.drop_duplicates(subset=['commit'])
			print(commit_features[(commit_features['Prev_Data']=='') & (commit_features['Cur_Data']=='')])
		print(commit_features)
		print(commits[~commits['commit'].isin(commit_features['commit'])])
		if feature_set == 'files':
			filepath = 'data/code_file.parquet'
		elif feature_set == 'changes':
			filepath = 'data/code_hunk.parquet'
		elif feature_set == 'scopes':
			filepath = 'data/code_ss.parquet'
		elif feature_set == 'contexts':
			filepath = 'data/code_hc.parquet'
		commit_features.columns = commit_features.columns.str.lower()
		makedir(outpath+'data/')
		commit_features.to_parquet(outpath+filepath, index=False, compression='gzip')

def main():
	# Partition the data
	partition_data('cvss')
	# Get the raw feature sets
	get_raw_features()

if __name__ == '__main__':
	main()
	exit()
