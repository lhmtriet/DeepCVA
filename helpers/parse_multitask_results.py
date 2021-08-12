from glob import glob

import pandas as pd


def transform_multitask_csv(df, avg=True):
    """
     Helper function to transform the format of the multitask_csv outputs
     to standard long form
     multitask_csv outputs have columns like below (snippet):

     Index(['run', 'set', 'fold', 'accuracy', 'precision', 'recall', 'f1-score',
    'gmean', 'mcc', 'pred_time', 'cvss2_conf_acc', 'cvss2_conf_prec',
    'cvss2_conf_rec', 'cvss2_conf_f1', 'cvss2_conf_gmean', 'cvss2_conf_mcc',
    'cvss2_integrity_acc', 'cvss2_integrity_prec', 'cvss2_integrity_rec',
    'cvss2_integrity_f1', 'cvss2_integrity_gmean', 'cvss2_integrity_mcc',...])
    """
    df = df.rename(
        columns={
            "accuracy": "multitask_acc",
            "precision": "multitask_prec",
            "recall": "multitask_rec",
            "f1-score": "multitask_f1",
            "gmean": "multitask_gmean",
            "mcc": "multitask_mcc",
            "auc": "multitask_auc",
        }
    )

    def reverse_col_name(colname):
        if "_" not in colname:
            return colname
        splits = colname.split("_")
        assert splits[-1] in [
            "acc",
            "rec",
            "prec",
            "f1",
            "gmean",
            "mcc",
            "auc",
            "time",
        ]
        return "_".join([splits[-1]] + splits[:-1])

    df.columns = [reverse_col_name(i) for i in df.columns]

    df = pd.wide_to_long(
        df,
        stubnames=["acc", "rec", "prec", "f1", "gmean", "mcc", "auc"],
        i=["run", "set", "fold"],
        j="scenario",
        sep="_",
        suffix="\\w+",
    ).reset_index()

    if not avg:
        return df

    df = df.drop(columns=["run", "fold"])
    df = df.groupby("scenario").mean()
    return df.reset_index()


def read_files(path, avg=False, verbose=0):
    """
    Helper function to read all multittask sequential files.
    """

    files = glob(path + "/*")

    all_dfs = []
    for file in files:
        filename = file.split("/")[-1].split("_")
        if len(filename) != 5:
            if verbose > 0:
                print("Skipped ", filename)
            continue
        df = transform_multitask_csv(pd.read_csv(file), avg=avg)
        df["feature_type"] = filename[0]
        df["feature_scope"] = filename[1]
        df["token"] = filename[2]
        df["fold"] = filename[3]
        # df["set"] = "folds"
        df = df.set_index(
            ["set", "scenario", "feature_type", "feature_scope", "token", "fold"]
        ).reset_index()
        all_dfs.append(df)
    all_results_df = pd.concat(all_dfs)
    return all_results_df


def parse_sequential_results(path, verbose=0):
    all_dfs = []
    for file in glob(path + "/*"):
        filename = file.split("/")[-1].split("_")
        if len(filename) != 7:
            if verbose > 0:
                print("Skipped", file)
            continue
        assert filename[3] == "cvss2"
        df = pd.read_csv(file)
        df["feature_type"] = filename[0]
        df["feature_scope"] = filename[1]
        df["token"] = filename[2]
        df["scenario"] = "_".join(filename[3:5])
        df["fold"] = filename[5]
        df = df.set_index(
            ["scenario", "feature_type", "feature_scope", "token", "fold"]
        ).reset_index()
        all_dfs.append(df)
    return pd.concat(all_dfs)
