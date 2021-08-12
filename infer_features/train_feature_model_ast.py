# %%
import sys

sys.path.append("../")
import pandas as pd
from pathlib import Path
import logging
from glob import glob
import helpers.feature_model_helpers as fmh
from progressbar import progressbar as pb
from pathlib import Path

Path("../feature_models").mkdir(parents=True, exist_ok=True)
Path("_logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(format="%(asctime)s: %(levelname)s: %(message)s")
logging.root.setLevel(level=logging.INFO)

# %% Get ID from arg
try:
    model = sys.argv[1]  # bow, w2v
    splitType = sys.argv[2]  # holdout, 0-9
    classType = sys.argv[3]  # multiclass, binary
except:
    model = "w2v"
    splitType = "holdout"
    classType = "binary"
print(model, splitType, classType)

# %% Load Data
file_data = pd.read_parquet("../data/ast_file.parquet")
mapname = "mc" if classType == "multiclass" else "binary"
set_map = pd.read_csv("../{}_map.csv".format(mapname))
set_map.columns = map(str.lower, set_map.columns)

# %% Extract Training Data
if splitType == "holdout":
    set_map = set_map[set_map.set == "train"]
else:
    set_map = set_map[set_map.partition != int(splitType)]
file_data = file_data[file_data.commit.isin(set_map.commit)]

# %% Train model
if model == "bow":
    sentences = file_data["prev_file"].to_list() + file_data["cur_file"].to_list()
    model_name = "../feature_models/ast_bow_{}_{}.model".format(classType, splitType)
    fmh.train_BoW(sentences, dataName=model_name)

if model == "w2v":
    sentences = file_data["prev_file"].to_list() + file_data["cur_file"].to_list()
    model_name = "../feature_models/ast_w2v_{}_{}.model".format(classType, splitType)
    fmh.train_w2v(sentences, dataName=model_name)
