import numpy as np
import pandas as pd
import pickle
import time
from tqdm import tqdm
from src.hybrid import final_hybrid
from src.utils import add_movie_encoding

# ---------------- LOAD DATA ----------------
df = pd.read_csv("processed_df.csv")
movies = pd.read_csv("movie.csv")

df = add_movie_encoding(df)
df = df.sort_values("timestamp")

train = df.groupby("userId").head(int(0.8 * len(df) / df['userId'].nunique()))
test = df.drop(train.index)

train = add_movie_encoding(train)
test = add_movie_encoding(test)

ground_truth = test.groupby("userId")["movieId"].apply(list).to_dict()
users = list(ground_truth.keys())[:20]


# ---------------- LOAD MODELS ----------------
from tensorflow.keras.models import load_model

svd_model = pickle.load(open("models/svd_model.pkl", "rb"))
knn_model = pickle.load(open("models/knn_model.pkl", "rb"))
xgb_model = pickle.load(open("models/xgb_model.pkl", "rb"))

lstm_model = load_model("models/lstm_model.h5")

movie_map = pickle.load(open("models/movie_map.pkl", "rb"))
feature_cols = pickle.load(open("models/feature_cols.pkl", "rb"))

user_features = pd.read_csv("models/user_features.csv", index_col="userId")

kmeans_cluster_rating = pickle.load(open("models/kmeans_cluster_rating.pkl", "rb"))
dbscan_cluster_rating = pickle.load(open("models/dbscan_cluster_rating.pkl", "rb"))


# ---------------- METRIC ----------------
def precision_at_k(rec, rel):
    if len(rel) == 0 or len(rec) == 0:
        return 0.0

    rec = set(rec)
    rel = set(rel)

    return len(rec & rel) / 10


def random_weights():
    w = np.random.rand(5)
    return w / w.sum()


# ---------------- TUNING ----------------
best_score = -1
best_weights = None

print("\n🚀 STARTING RANDOM SEARCH TUNING...\n")

start_total = time.time()

for i in tqdm(range(30), desc="🔁 Tuning Progress"):

    t0 = time.time()

    weights = random_weights()
    scores = []

    print(f"\n==============================")
    print(f"🔁 TRIAL {i+1}/30")
    print(f"⚖ Weights: {weights}")
    print(f"==============================")

    for user in users:

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
            weights,
            n=10
        )

        if recs is None or recs.empty:
            continue

        rec_list = recs["movieId"].tolist()
        rel_list = ground_truth.get(user, [])

        scores.append(precision_at_k(rec_list, rel_list))

    avg = np.mean(scores) if len(scores) > 0 else 0

    trial_time = time.time() - t0

    print(f"✔ Trial {i+1} DONE")
    print(f"📊 Precision@10: {avg:.4f}")
    print(f"⏱ Time: {trial_time:.2f} sec")

    # ---------------- SAVE BEST ----------------
    if avg > best_score:
        best_score = avg
        best_weights = weights
        print("\n🔥 NEW BEST FOUND!")
        print(f"🏆 Score: {best_score:.4f}")
        print(f"⚖ Weights: {best_weights}")

    # ---------------- HEARTBEAT ----------------
    if (i + 1) % 2 == 0:
        print(f"\n🔥 HEARTBEAT: Completed {i+1}/30 trials\n")

total_time = time.time() - start_total

print("\n==============================")
print("🏁 TUNING COMPLETE")
print(f"🏆 BEST SCORE: {best_score:.4f}")
print(f"⚖ BEST WEIGHTS: {best_weights}")
print(f"⏱ TOTAL TIME: {total_time:.2f} sec")
print("==============================")

# ---------------- SAVE ----------------
with open("models/best_weights.pkl", "wb") as f:
    pickle.dump(best_weights, f)

print("\n💾 Best weights saved successfully!")