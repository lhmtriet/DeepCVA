import json
import pickle as pkl
import re
import sys
from collections import Counter
from glob import glob
from pathlib import Path
from random import sample
from time import perf_counter

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from scipy.sparse import hstack
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm as tqdm

path = "../../"
Path("_more_cvss_baseline_results").mkdir(exist_ok=True)
savedir = "_more_cvss_baseline_results/"
cvss_df = pd.read_csv(path + "cvss_map_time.csv")

# Get dims as argument and split
input_string = "code_hc_bow_multiclass"
input_string = sys.argv[1]
dims = ["{}_{}".format(input_string, i) for i in range(10)]
feature_file_paths = [
    path + "inferred_features_time/{}.pkl".format(dim) for dim in dims
]
if "manual" in pd.Series(dims).any():
    feature_file_paths = [path + "data/manual_{}.parquet".format(i) for i in range(10)]

# Get all cvss metrics from dataframe
cvss2_mets = cvss_df.columns[cvss_df.columns.str.contains("cvss2_")]
cvss2_mets = [i for i in cvss2_mets if i not in ["cvss2_basescore"]]

# %% -------------------------------------------------------------------------------- #
#                                   Helper functions                                  #
# ----------------------------------------------------------------------------------- #


def get_train_test(df, partition):
    """
    Helper function to get train/test partitions from dataframe

    df: set (train/test), partition (0-9)
    """
    if str(partition) == "holdout":
        return df[df.set == "train"], df[df.set == "test"]
    return df[df.partition != int(partition)], df[df.partition == int(partition)]


def get_train_test_val(df, partition):
    df.time_partition = df.time_partition.astype(int)
    return (
        df[df.time_partition < int(partition)],
        df[df.time_partition == int(partition)],
        df[df.time_partition == int(partition) + 1],
    )


def evaluate(
    classifier,
    feature_extractor,
    mapping_file,
    label_col,
    eval_func,
    partition,
    sampling_method,
    sampling,
):
    """
    Helper function to perform cross-fold validation given a generic classifier,
    feature extraction function, and relevant data

    classifier (class instance): sklearn estimator API
    feature_extractor (function): given mapping df, return features in row order
    mapping file (pandas dataframe): cvss_map.csv, binary_map.csv, mc_map.csv
    label_col (string): column name for labels
    eval_func: evaluation metric to be used (api: sklearn)
    partition: input value to get_train_test (i.e. "holdout", 0-9)
    """
    mapping_file["X"] = feature_extractor(mapping_file)
    mapping_file["y"] = mapping_file[label_col]

    train, val, test = get_train_test_val(mapping_file, int(partition) + 1)

    train_x, train_y = np.array(train.X.tolist()), np.array(train.y.tolist())
    if sampling == -1:
        sampler = RandomOverSampler(random_state=42)
        train_x, train_y = sampler.fit_resample(train_x, train_y)
    if sampling > 0:
        smallest_sample = np.array(list(Counter(train_y).values())).min()
        try:
            sampler = SMOTE(k_neighbors=sampling, random_state=42)
            train_x, train_y = sampler.fit_resample(train_x, train_y)
        except ValueError:
            print("Skip smote: SS:{} KN:{}".format(smallest_sample, sampling))
            return

    t_start = perf_counter()
    classifier.fit(train_x, train_y)
    t_end = perf_counter()
    train_time = t_end - t_start
    t_start = perf_counter()
    y_val = classifier.predict(np.array(val.X.tolist()))
    t_end = perf_counter()
    val_time = t_end - t_start
    t_start = perf_counter()
    y_test = classifier.predict(np.array(test.X.tolist()))
    t_end = perf_counter()
    test_time = t_end - t_start

    all_results = {
        "eval_val": eval_func(val.y.tolist(), y_val),
        "eval_test": eval_func(test.y.tolist(), y_test),
        "train_time": train_time,
        "test_time": test_time,
        "val_time": val_time,
        "sampling": sampling_method,
        "smote": sampling,
    }
    return json.dumps(all_results)


def feature_extractor_generator(file):
    """
    Helper function that generates combinations of feature_extractor functions
    to be passed to evaluate function.
    The feature extraction function drops the commit column in a given dataframe
    and hstacks the rest of the columns
    (i.e. prev_features, cur_features, prev_context, cur_context)

    feature_file (strings): filepath to inferred feature file
    """
    assert "pkl" in file or "parquet" in file
    if "pkl" in file:
        load_func = lambda x: pkl.load(open(x, "rb"))
    if "parquet" in file:
        load_func = pd.read_parquet
    if "manual" in file:
        file = re.sub("_[\d]+", "", file)

    def feature_extractor(mapping_file):
        inferred_features = load_func(file)
        inferred_features = (
            inferred_features.set_index("commit").loc[mapping_file.commit].reset_index()
        )
        assert inferred_features.commit.eq(mapping_file.commit).all()
        return inferred_features.drop(columns=["commit"]).apply(
            lambda x: hstack(x).toarray()[0], axis=1
        )

    return feature_extractor


def result_format(task, clf, ftype, fscope, token, partition, res):
    """
    Helper function to produce result dictionary (for use in dataframe)
    """
    return {
        "task": task,
        "classifier": clf,
        "feature_type": ftype,
        "feature_scope": fscope,
        "token": token,
        "partition": str(int(partition) + 1),
        "results": res,
    }


def filepath_result_format(classifier_name, task, res, filepath):
    """
    Helper function to produce result dictionary using filepath
    filepath (str) : (e.g. '../../inferred_features/code_ss_w2v_multiclass_7.pkl'
    """
    filepath = filepath.split("/")[-1]
    dims = filepath.split(".")[0].split("_")
    if dims[0] == "manual":
        return result_format(
            task, classifier_name, dims[0], dims[0], dims[0], dims[1], res
        )
    return result_format(task, classifier_name, dims[0], dims[1], dims[2], dims[4], res)


def result_completed(str):
    return str in [
        i.split("/")[-1].split(".")[0] for i in glob("_more_cvss_baseline_results/*")
    ]


def concat_eval(x, y):
    """
    Helper function to calculate multiple evaluation metrics at once
    """
    return {
        "recall": recall_score(x, y, average="macro", zero_division=0),
        "precision": precision_score(x, y, average="macro", zero_division=0),
        "f1_score": f1_score(x, y, average="macro", zero_division=0),
        "mcc": mcc(x, y),
    }


# %% -------------------------------------------------------------------------------- #
#                             Clustering + Nearest Centroid                           #
# ----------------------------------------------------------------------------------- #
t1_start = perf_counter()


class MajorityKMeansEstimator:
    """
    Perform clustering via kmeans
    On prediction, assign CVSS string based on centroid
    """

    def __init__(self, n_clusters=2):
        self.kmeans = KMeans(n_clusters, random_state=0)
        self.cluster_to_label = {}

    def fit(self, X, y):
        self.kmeans.fit_transform(X)
        # Count all pairs of cluster_num, label and keep highest
        # cluster_num, label pair for each cluster
        labelled_clusters = zip(self.kmeans.predict(X), y, np.ones(len(y)))
        self.cluster_to_label = dict(
            pd.DataFrame(labelled_clusters, columns=["cluster", "label", "count"])
            .groupby(["cluster", "label"])
            .count()
            .sort_values("count", ascending=0)
            .groupby("cluster")
            .head(1)
            .reset_index()[["cluster", "label"]]
            .values
        )

    def predict(self, X):
        return [self.cluster_to_label[i] for i in self.kmeans.predict(X)]


n_clusters = int(sys.argv[2])  # :TODO: Refactor
clf_name = "clustercentroid-n{}".format(n_clusters)

all_results = []

for sampling in [-2, -1, 1, 5, 10, 15, 20]:
    if sampling == -2:
        sampling_method = "none"
    elif sampling == -1:
        sampling_method = "over"
    else:
        sampling_method = "smote"

    for cvss2_met in tqdm(cvss2_mets):
        # if result_completed("{}_".format(clf_name) + input_string):
        #     break
        for feature_file in feature_file_paths:
            result = evaluate(
                classifier=MajorityKMeansEstimator(n_clusters=n_clusters),
                feature_extractor=feature_extractor_generator(feature_file),
                mapping_file=cvss_df,
                label_col=cvss2_met,
                eval_func=concat_eval,
                partition=feature_file.split("_")[-1].split(".")[0],
                sampling=sampling,
                sampling_method=sampling_method,
            )
            all_results.append(
                filepath_result_format(clf_name, cvss2_met, result, feature_file)
            )

    if len(all_results) > 0:
        save_file = "{}_{}_{}_{}.csv".format(
            clf_name, input_string, sampling_method, sampling
        )
        pd.DataFrame(all_results).to_csv(savedir + save_file, index=0)
        t1_stop = perf_counter()
        print("TOTAL_ELAPSED_TIME ({}):".format(save_file), t1_stop - t1_start)

# %% -------------------------------------------------------------------------------- #
#                            Most Frequent  estimator (naive)                         #
# ----------------------------------------------------------------------------------- #

all_results = []
for run in range(10):
    for sampling in tqdm([-2, -1, 1, 5, 10, 15, 20]):
        if sampling == -2:
            sampling_method = "none"
        elif sampling == -1:
            sampling_method = "over"
        else:
            sampling_method = "smote"

        for cvss2_met in tqdm(cvss2_mets):
            for feature_file in feature_file_paths:
                result = evaluate(
                    classifier=DummyClassifier(strategy="uniform"),
                    feature_extractor=feature_extractor_generator(feature_file),
                    mapping_file=cvss_df,
                    label_col=cvss2_met,
                    eval_func=concat_eval,
                    partition=feature_file.split("_")[-1].split(".")[0],
                    sampling=sampling,
                    sampling_method=sampling_method,
                )
                final_res = filepath_result_format(
                    "most_frequent", cvss2_met, result, feature_file
                )
                final_res["run"] = run
                all_results.append(final_res)

pd.DataFrame(all_results).to_csv(savedir + "cvss_baseline_uniform.csv", index=0)

