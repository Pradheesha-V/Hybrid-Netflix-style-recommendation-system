# ---------------- IMPORTS ----------------
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from tensorflow.keras.models import load_model

from src.hybrid import final_hybrid
from src.utils import add_movie_encoding


# ---------------- LOAD DATA ----------------
df = pd.read_csv("processed_df.csv")
movies = pd.read_csv("movie.csv")

df = add_movie_encoding(df)
df = df.sort_values("timestamp")

# Train-Test Split (per user)
train = df.groupby("userId").head(int(0.8 * len(df) / df['userId'].nunique()))
test = df.drop(train.index)

train = add_movie_encoding(train)
test = add_movie_encoding(test)

# Ground truth
ground_truth = test.groupby("userId")["movieId"].apply(list).to_dict()
users = list(ground_truth.keys())[:50]   # limit for speed


# ---------------- LOAD MODELS ----------------
svd_model = pickle.load(open("models/svd_model.pkl", "rb"))
knn_model = pickle.load(open("models/knn_model.pkl", "rb"))
xgb_model = pickle.load(open("models/xgb_model.pkl", "rb"))

lstm_model = load_model("models/lstm_model.h5")

movie_map = pickle.load(open("models/movie_map.pkl", "rb"))
feature_cols = pickle.load(open("models/feature_cols.pkl", "rb"))

user_features = pd.read_csv("models/user_features.csv", index_col="userId")

kmeans_cluster_rating = pickle.load(open("models/kmeans_cluster_rating.pkl", "rb"))
dbscan_cluster_rating = pickle.load(open("models/dbscan_cluster_rating.pkl", "rb"))

best_weights = pickle.load(open("models/best_weights.pkl", "rb"))


# ---------------- METRICS ----------------
def precision_at_k(rec, rel, k=10):
    rec = rec[:k]
    rel = set(rel)
    if len(rel) == 0:
        return 0
    return len(set(rec) & rel) / k


def recall_at_k(rec, rel, k=10):
    rec = rec[:k]
    rel = set(rel)
    if len(rel) == 0:
        return 0
    return len(set(rec) & rel) / len(rel)


def average_precision_at_k(rec, rel, k=10):
    rec = rec[:k]
    rel = set(rel)

    score = 0.0
    hits = 0

    for i, item in enumerate(rec):
        if item in rel:
            hits += 1
            score += hits / (i + 1)

    if len(rel) == 0:
        return 0

    return score / min(len(rel), k)


# ---------------- EVALUATION ----------------
precisions = []
recalls = []
maps = []

for user in tqdm(users):

    recs = final_hybrid(
        user,
        train,
        movies,
        svd_model,
        knn_model,
        xgb_model,
        lstm_model,
        movie_map,
        user_features,
        feature_cols,
        kmeans_cluster_rating,
        dbscan_cluster_rating,
        best_weights,
        n=10
    )

    if recs is None or recs.empty:
        continue

    rec_list = recs["movieId"].tolist()
    rel_list = ground_truth.get(user, [])

    precisions.append(precision_at_k(rec_list, rel_list, k=10))
    recalls.append(recall_at_k(rec_list, rel_list, k=10))
    maps.append(average_precision_at_k(rec_list, rel_list, k=10))


# ---------------- RESULTS ----------------
print("\n🎯 FINAL METRICS:")
print("Precision@10:", np.mean(precisions))
print("Recall@10:", np.mean(recalls))
print("MAP@10:", np.mean(maps))