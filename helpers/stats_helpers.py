import random
from math import sqrt

import numpy as np
import pandas as pd
from numpy import mean, var
from scipy.stats import mannwhitneyu, wilcoxon


def mwyu(g1, g2, override_alt=None):
    """Calculate mannwhitneyu test."""
    g1_mean = np.mean(g1)
    g2_mean = np.mean(g2)
    if g1_mean < g2_mean:
        altmethod = "greater"
    else:
        altmethod = "less"
    if override_alt:
        altmethod = override_alt
    score = mannwhitneyu(g1, g2, alternative=altmethod)
    return score


def wilc(g1, g2):
    """Make groups equal in size before wilcoxon."""
    random.seed(42)
    if len(g1) < len(g2):
        g2 = random.sample(list(g2), len(g1))
    if len(g2) < len(g1):
        g1 = random.sample(list(g1), len(g2))
    return wilcoxon(g1, g2).pvalue


def cohend(d1, d2):
    """Calculate Cohen's d for independent samples."""
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = mean(d1), mean(d2)
    # calculate the effect size
    return abs(u1 - u2) / s


def compute_stats(
    df1, df2, a="A", b="B", evals=["mcc", "f1"], extra_metrics=[], override_alt=None,
):
    """
    Helper function for computing stats
    df1 and df2 in form:

         set          scenario feature_type feature_scope token fold  run    mcc
    0  folds         multitask         code            hc  word    6    0  0.488
    1  folds        cvss2_conf         code            hc  word    6    0  0.428
    2  folds   cvss2_integrity         code            hc  word    6    0  0.478
    3  folds       cvss2_avail         code            hc  word    6    0  0.546
    4  folds  cvss2_accessvect         code            hc  word    6    0  0.573

    """
    cvss_metrics = [
        "cvss2_auth",
        "cvss2_conf",
        "cvss2_accessvect",
        "cvss2_avail",
        "cvss2_accesscomp",
        "cvss2_severity",
        "cvss2_integrity",
    ]

    cvss_metrics = cvss_metrics + extra_metrics

    stats_final = []
    for eval in evals:
        stats_eval_df = []
        for metric in cvss_metrics + ["overall"]:
            if metric == "overall":
                df1_f = df1.copy()
                df2_f = df2.copy()
            else:
                df1_f = df1[df1.scenario == metric]
                df2_f = df2[df2.scenario == metric]

            df1_v = df1_f[eval].values
            df2_v = df2_f[eval].values
            stats_eval_df.append(
                {
                    "scenario": metric,
                    "A": a,
                    "B": b,
                    "mean_A_{}".format(eval): np.mean(df1_v),
                    "mean_B_{}".format(eval): np.mean(df2_v),
                    "mwyu_{}".format(eval): mwyu(df1_v, df2_v, override_alt).pvalue,
                    "cohend_{}".format(eval): cohend(df1_v, df2_v),
                    "wilcoxon_{}".format(eval): wilc(df1_v, df2_v),
                    "raw_A_{}".format(eval): df1_v,
                    "raw_B_{}".format(eval): df2_v,
                    "lengths": [len(df1_v), len(df2_v)],
                }
            )
        stats_final.append(
            pd.DataFrame.from_dict(stats_eval_df).set_index(["scenario", "A", "B"])
        )
    return pd.concat(stats_final, axis=1).reset_index()


def calc_cvss_score(df, extra_metrics=[]):
    """
    Calculates score given dataframe of format like:

    columns: ['commit', 'cvss2_accesscomp', 'cvss2_accesscomp-pred',
       'cvss2_accessvect', 'cvss2_accessvect-pred', 'cvss2_auth',
       'cvss2_auth-pred', 'cvss2_avail', 'cvss2_avail-pred', 'cvss2_conf',
       'cvss2_conf-pred', 'cvss2_integrity', 'cvss2_integrity-pred',
       'cvss2_severity', 'cvss2_severity-pred']

    Example:

             cvss2_accesscomp cvss2_accesscomp-pred cvss2_accessvect  \
    26             MEDIUM                   LOW          NETWORK
    29                LOW                   LOW          NETWORK
    77                LOW                   LOW          NETWORK
    93             MEDIUM                   LOW          NETWORK
    100               LOW                   LOW          NETWORK
    ...               ...                   ...              ...
    2420              LOW                   LOW          NETWORK
    2430           MEDIUM                MEDIUM          NETWORK
    2439              LOW                   LOW          NETWORK
    2442              LOW                   LOW          NETWORK
    2446              LOW                   LOW          NETWORK

         cvss2_accessvect-pred
    26                 NETWORK
    29                 NETWORK
    77                 NETWORK
    93                 NETWORK
    100                NETWORK
    ...                    ...
    2420               NETWORK
    2430               NETWORK
    2439               NETWORK
    2442               NETWORK
    2446               NETWORK

    """

    df = df.copy()

    def partial_score(i, j, l=["LOW", "MEDIUM", "HIGH"]):
        if i == j:
            return 0
        if i == "-" or j == "-":
            return 0
        if i == l[0] and j == l[1] or i == l[1] and j == l[0]:
            return 0.5
        if i == l[1] and j == l[2] or i == l[2] and j == l[1]:
            return 0.5
        if i == l[0] and j == l[2] or i == l[2] and j == l[0]:
            return 1

    def calc_score(row, score_type="abs"):
        score = 0
        for i in [
            "cvss2_conf",
            "cvss2_integrity",
            "cvss2_avail",
            "cvss2_accessvect",
            "cvss2_accesscomp",
            "cvss2_auth",
            "cvss2_severity",
        ] + extra_metrics:
            if score_type == "abs" and row[i] != row[i + "-pred"]:
                score += 1
            if score_type == "partial":
                true_val = row[i]
                pred_val = row[i + "-pred"]
                if true_val == pred_val:
                    continue
                if i in ["cvss2_accesscomp", "cvss2_severity"]:
                    score += partial_score(true_val, pred_val)
                elif i in ["cvss2_conf", "cvss2_avail", "cvss2_integrity"]:
                    score += partial_score(
                        true_val, pred_val, ["NONE", "PARTIAL", "COMPLETE"]
                    )
                else:
                    score += 1

        return score

    df["score_abs"] = df.apply(calc_score, args=["abs"], axis=1)
    df["score_partial"] = df.apply(calc_score, args=["partial"], axis=1)
    return df.set_index(["commit", "score_abs", "score_partial"]).reset_index()
