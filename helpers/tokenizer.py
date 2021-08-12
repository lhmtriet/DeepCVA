import re
from sklearn.feature_extraction.text import TfidfVectorizer


def gen_tok_pattern():
    single_toks = [
        "<=",
        ">=",
        "<",
        ">",
        "\\?",
        "\\/=",
        "\\+=",
        "\\-=",
        "\\+\\+",
        "--",
        "\\*=",
        "\\+",
        "-",
        "\\*",
        "\\/",
        "!=",
        "==",
        "=",
        "!",
        "&=",
        "&",
        "\\%",
        "\\|\\|",
        "\\|=",
        "\\|",
        "\\$",
        "\\:",
    ]
    single_toks = "(?:" + "|".join(single_toks) + ")"
    word_toks = "(?:[a-zA-Z0-9]+)"
    return single_toks + "|" + word_toks


# Extract features
def extract_features(start_n_gram, end_n_gram, token_pattern=None, vocabulary=None):
    return TfidfVectorizer(
        stop_words=None,
        ngram_range=(1, 1),
        use_idf=False,
        min_df=0.0,
        max_df=1.0,
        max_features=10000,
        norm=None,
        smooth_idf=False,
        lowercase=False,
        token_pattern=token_pattern,
        vocabulary=vocabulary,
    )


def get_tokenizer(vectorize=False):
    code_token_pattern = gen_tok_pattern()
    vectorizer = extract_features(
        start_n_gram=1, end_n_gram=1, token_pattern=code_token_pattern
    )
    if vectorize:
        return vectorizer
    return vectorizer.build_analyzer()
