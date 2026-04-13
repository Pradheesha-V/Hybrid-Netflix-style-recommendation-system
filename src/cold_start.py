import pandas as pd

def popular_recommend(df, movies, n=10):

    pop = (
        df.groupby('movieId')['rating']
        .mean()
        .dropna()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )

    return pop.merge(movies, on='movieId', how='left')


def cold_start(user_id, df, movies, n=10):

    if user_id not in df['userId'].values:
        return popular_recommend(df, movies, n)

    return None