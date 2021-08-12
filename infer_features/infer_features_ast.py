import sys

sys.path.append("../")
import helpers.feature_model_helpers as fmh
import pandas as pd
import pickle
from gensim.models import Word2Vec
from importlib import reload
from pathlib import Path
import pickle as pkl
import numpy as np
from scipy.sparse import coo_matrix

reload(fmh)

# %% Get ID from arg
try:
    modelType = sys.argv[1]  # bow, w2v
    splitType = sys.argv[2]  # holdout, 0-9
    classType = sys.argv[3]  # multiclass, binary
    scopeType = sys.argv[4]  # hunk, ss, hc
except:
    modelType = "w2v"
    splitType = "holdout"
    classType = "binary"
    scopeType = "hunk"
print(modelType, splitType, classType, scopeType)

# %% Load Data
data = pd.read_parquet("../data/ast_{}.parquet".format(scopeType)).fillna("")

if scopeType == "hunk" or scopeType == "ss":
    inferred_data = fmh.infer_ast_nlp_features(data, classType, splitType, modelType)
    inferred_data = inferred_data.rename(
        columns={"prev_data": "prev_features", "cur_data": "cur_features"}
    )

if scopeType == "hc":
    data[["prev_features", "prev_context"]] = data.prev_data.str.split(",", expand=True)
    data[["cur_features", "cur_context"]] = data.cur_data.str.split(",", expand=True)
    data = data.fillna("")
    data[["prev_data", "cur_data"]] = data[["prev_features", "prev_context"]]
    data = fmh.infer_ast_nlp_features(data, classType, splitType, modelType)
    data = data.drop(columns=["prev_features", "prev_context"])
    data = data.rename(
        columns={"prev_data": "prev_features", "cur_data": "prev_context"}
    )
    data[["prev_data", "cur_data"]] = data[["cur_features", "cur_context"]]
    data = fmh.infer_ast_nlp_features(data, classType, splitType, modelType)
    data = data.drop(columns=["cur_features", "cur_context"])
    data = data.rename(columns={"prev_data": "cur_features", "cur_data": "cur_context"})
    inferred_data = data

inferred_data = inferred_data.reset_index(drop=True)
Path("../inferred_features").mkdir(parents=True, exist_ok=True)

# %% Fill in with sparse
if modelType == "bow":
    feat_len = inferred_data.iloc[0][inferred_data.columns[1]][0].shape[1]
    for col in inferred_data.columns:
        if col == "commit":
            continue
        inferred_data[col] = inferred_data[col].apply(
            lambda x: coo_matrix((1, feat_len), dtype=np.float64) if x is None else x
        )

# %% Save File
pkl.dump(
    inferred_data,
    open(
        "../inferred_features/ast_{}_{}_{}_{}.pkl".format(
            scopeType, modelType, classType, splitType
        ),
        "wb",
    ),
)

# %%
