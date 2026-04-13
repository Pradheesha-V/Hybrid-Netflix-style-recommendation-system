import pandas as pd
import pickle

def add_movie_encoding(df, movie_map_path="models/movie_map.pkl"):
    movie_map = pickle.load(open(movie_map_path, "rb"))

    df = df.copy()
    df["movie_encoded"] = df["movieId"].map(movie_map)

    df["movie_encoded"] = df["movie_encoded"].fillna(0).astype(int)

    return df