import numpy as np
import pandas as pd
from src.scores import normalize, svd_score, knn_score, lstm_score


def build_xgb_features(user_id, movie_id, df, user_features):
    try:
        user_row = user_features.loc[user_id]
    except:
        user_row = {}

    movie_df = df[df['movieId'] == movie_id]

    if movie_df.empty:
        return None

    movie_row = movie_df.iloc[0]

    features = {
        "user_avg_rating": user_row.get("user_avg_rating", 0),
        "user_rating_count": user_row.get("rating_count", 0),
        "movie_avg_rating": movie_row.get("movie_avg_rating", 0),
        "movie_rating_count": movie_row.get("rating_count", 0)
    }

    return pd.DataFrame([features])


def final_hybrid(
    user_id,
    df,
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
):

    w_svd, w_knn, w_xgb, w_cluster, w_lstm = weights

    user_df = df[df['userId'] == user_id]
    if user_df.empty:
        return pd.DataFrame()

    seen = set(user_df['movieId'])

    # candidate movies
    popular_movies = df['movieId'].value_counts().head(200).index
    candidates = [m for m in popular_movies if m not in seen]

    if len(candidates) == 0:
        return pd.DataFrame()

    # LSTM sequence (SAFE)
    user_seq = user_df.sort_values("timestamp")["movieId"].tolist()
    user_seq = [movie_map.get(m, 0) for m in user_seq]

    # cluster info
    try:
        user_cluster = user_features.loc[user_id].get("cluster", None)
        user_db_cluster = user_features.loc[user_id].get("db_cluster", None)
    except:
        user_cluster = None
        user_db_cluster = None

    results = []

    for movie_id in candidates:

        # ---------------- SVD ----------------
        s1 = normalize(svd_score(user_id, movie_id, svd_model))

        # ---------------- KNN ----------------
        s2 = normalize(knn_score(user_id, movie_id, knn_model))

        # ---------------- XGB (FIXED REAL FEATURES) ----------------
        X = build_xgb_features(user_id, movie_id, df, user_features)

        if X is None:
            s3 = 0
        else:
            try:
                X = X.reindex(columns=feature_cols, fill_value=0)
                s3 = normalize(xgb_model.predict(X)[0])
            except:
                s3 = 0

        # ---------------- CLUSTER ----------------
        s4 = 0

        if user_db_cluster is not None:
            s4 = dbscan_cluster_rating.get((user_db_cluster, movie_id), 0)

        if s4 == 0 and user_cluster is not None:
            s4 = kmeans_cluster_rating.get((user_cluster, movie_id), 0)

        # ---------------- LSTM ----------------
        movie_idx = movie_map.get(movie_id, 0)
        s5 = lstm_score(user_seq, movie_idx, lstm_model)

        # ---------------- FINAL SCORE ----------------
        score = (
            w_svd * s1 +
            w_knn * s2 +
            w_xgb * s3 +
            w_cluster * s4 +
            w_lstm * s5
        )

        if not np.isnan(score):
            results.append((movie_id, score))

    recs = pd.DataFrame(results, columns=["movieId", "score"])

    if recs.empty:
        return pd.DataFrame()

    recs = recs.merge(movies, on="movieId", how="left")

    return recs.sort_values("score", ascending=False).head(n)