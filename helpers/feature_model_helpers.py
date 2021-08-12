import pandas as pd
import numpy as np
import pickle
import sys
import re
import time
import math

from scipy.sparse import hstack
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    auc,
    matthews_corrcoef,
)
import tensorflow.keras.models
from tqdm import tqdm

# Local Import
sys.path.append("../helpers")
from tokenizer import get_tokenizer

tqdm.pandas()

embedding_sz = 300
num_workers = 3
w2v_model = None

# Train a w2v embedding model
def train_w2v(sentences, dataName):
    # Tokenize files
    tokenizer = get_tokenizer()
    sentences = list(map(tokenizer, sentences))
    # Train and save model
    w2v_model = Word2Vec(
        sentences,
        size=embedding_sz,
        window=5,
        min_count=0,
        max_vocab_size=10000,
        workers=num_workers,
        sg=1,
        seed=42,
    )
    w2v_model.save(dataName)


# Train and save a BoW feature model
def train_BoW(sentences, dataName):
    vectorizer = get_tokenizer(vectorize=True)
    vectorizer.fit(sentences)
    pickle.dump(vectorizer, open(dataName, "wb"))


# Convert a sentence to a feature vector using text embeddings
#  Vector averaging of words in sentence
def sen_to_vec(sen):
    sen_vec = np.array([0.0] * embedding_sz)
    cnt = 0

    for w in sen:
        try:
            sen_vec = sen_vec + w2v_model[w]
            cnt += 1
        except:
            pass
    if cnt == 0:
        return np.array([0.0] * embedding_sz)

    return sen_vec / (cnt * 1.0)


# Remove changes separation between files
def clean_changes(changes):
    return re.sub("##<END>##[\r\n]", "", changes)


def infer_nlp_features(data, scenario):
    global w2v_model
    # Clean the data
    data["Prev_Changes"] = data["Prev_Changes"].map(clean_changes)
    data["Cur_Changes"] = data["Cur_Changes"].map(clean_changes)
    # Create a tokenizer
    tokenizer = get_tokenizer()
    # Load NLP feature models
    w2v_model = Word2Vec.load("feature_models/w2v_" + scenario + ".model")
    bow_model = pickle.load(open("feature_models/BoW_" + scenario + ".model", "rb"))
    # Tokenize the features
    prev_sentences = data["Prev_Changes"].map(tokenizer)
    cur_sentences = data["Cur_Changes"].map(tokenizer)
    # Encode and concatenate the features
    w2v_features = np.hstack(
        [
            np.asarray(prev_sentences.map(sen_to_vec).values.tolist()),
            np.asarray(cur_sentences.map(sen_to_vec).values.tolist()),
        ]
    )
    bow_features = hstack(
        [
            bow_model.transform(data["Prev_Changes"].values),
            bow_model.transform(data["Cur_Changes"].values),
        ]
    )
    features = hstack([w2v_features, bow_features])
    return features


def infer_ast_nlp_features(data, classType, splitType, modelType):
    global w2v_model
    # Clean the data
    data["prev_data"] = data["prev_data"].map(clean_changes)
    data["cur_data"] = data["cur_data"].map(clean_changes)
    # Create a tokenizer
    tokenizer = get_tokenizer()
    # Tokenize the features
    data["prev_data"] = data["prev_data"].map(tokenizer)
    data["cur_data"] = data["cur_data"].map(tokenizer)

    if modelType == "w2v":
        w2v_model = Word2Vec.load(
            "../feature_models/ast_w2v_{}_{}.model".format(classType, splitType)
        )
        data["prev_data"] = data.prev_data.progress_apply(sen_to_vec)
        data["cur_data"] = data.cur_data.progress_apply(sen_to_vec)
        return data
    if modelType == "bow":
        model = pickle.load(
            open(
                "../feature_models/ast_bow_{}_{}.model".format(classType, splitType),
                "rb",
            )
        )
        data["prev_data"] = data.prev_data.progress_apply(
            lambda x: model.transform([" ".join(x)]) if len(x) > 0 else None
        )
        data["cur_data"] = data.cur_data.progress_apply(
            lambda x: model.transform([" ".join(x)]) if len(x) > 0 else None
        )
        return data


def evaluate(
    clf,
    x_train,
    y_train,
    x_test,
    y_test,
    clf_name,
    outfile,
    fold="",
    lb=None,
    loadbestkeras=None,
    binary=True,
):
    """
    From Roland's Code
    """
    # Train
    t_start = time.time()
    clf.fit(x_train, y_train)
    train_time = time.time() - t_start

    # If keras
    if loadbestkeras:
        clf = tensorflow.keras.models.load_model(loadbestkeras)

    # Predict
    p_start = time.time()
    y_pred = clf.predict(x_test)
    pred_time = time.time() - p_start

    if hasattr(y_pred, "ndim") and y_pred.ndim > 1:
        y_pred = lb.inverse_transform(y_pred)
    if hasattr(y_test, "ndim") and y_test.ndim > 1:
        y_test = lb.inverse_transform(y_test)

    # If multi
    avgval = "binary" if binary else "weighted"

    # Evaluate
    precision = precision_score(y_test, y_pred, average=avgval)
    recall = recall_score(y_test, y_pred, average=avgval)
    gmean = math.sqrt(recall * precision)

    outfile.write(
        fold
        + ","
        + clf_name
        + ","
        + "{:.3f}".format(accuracy_score(y_test, y_pred))
        + ","
        + "{:.3f}".format(precision)
        + ","
        + "{:.3f}".format(recall)
        + ","
        + "{:.3f}".format(f1_score(y_test, y_pred, average="micro"))
        + ","
        + "{:.3f}".format(f1_score(y_test, y_pred, average="macro"))
        + ","
        + "{:.3f}".format(f1_score(y_test, y_pred, average=avgval))
        + ","
        + "{:.3f}".format(gmean)
        + ","
        + "{:.3f}".format(matthews_corrcoef(y_test, y_pred))
        + ","
        + "{:.3f}".format(train_time)
        + ","
        + "{:.3f}".format(pred_time)
        + "\n"
    )

    # outfile.write(fold + "," + clf_name + "," + "{:.3f}".format(accuracy_score(y_test, y_pred)) + "," + "{:.3f}".format(precision) + "," + \
    # 	"{:.3f}".format(recall) + "," + "{:.3f}".format(f1_score(y_test, y_pred, average='micro')) + "," + \
    # 	"{:.3f}".format(f1_score(y_test, y_pred, average='macro')) + "," + "{:.3f}".format(f1_score(y_test, y_pred, average=avgval)) + "," + \
    # 	"{:.3f}".format(gmean) + "," + "{:.3f}".format(matthews_corrcoef(y_test, y_pred)) + "," + \
    # 	"{:.3f}".format(train_time) + "," + "{:.3f}".format(pred_time) + "\n")

    return clf
