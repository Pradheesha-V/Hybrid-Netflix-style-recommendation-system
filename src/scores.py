import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------- NORMALIZE ----------------
def normalize(x):
    if x is None or np.isnan(x):
        return 0
    return float(x) / 5.0

def svd_score(user_id, movie_id, svd_model):
    try:
        return svd_model.predict(user_id, movie_id).est
    except:
        return 0


def knn_score(user_id, movie_id, knn_model):
    try:
        return knn_model.predict(user_id, movie_id).est
    except:
        return 0

def xgb_score(movie_features, xgb_model):
    try:
        return float(xgb_model.predict(movie_features)[0])
    except:
        return 0

def lstm_score(user_seq, movie_idx, lstm_model):

    if len(user_seq) < 2:
        return 0

    seq = pad_sequences([user_seq[-10:]], maxlen=10)

    try:
        pred = lstm_model.predict(seq, verbose=0)
        if movie_idx >= pred.shape[1]:
            return 0
        return float(pred[0][movie_idx])
    except:
        return 0