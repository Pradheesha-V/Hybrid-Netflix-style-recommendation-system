import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from src.hybrid import final_hybrid
from src.cold_start import cold_start


# ================= PAGE CONFIG =================
st.set_page_config(layout="wide", page_title="Movie Recommender")


# ================= THEME =================
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


# ================= POSTER GENERATOR =================
def create_title_poster(title):
    img = Image.new('RGB', (300, 450), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except:
        font = ImageFont.load_default()

    words = title.split()
    lines, line = [], ""

    for word in words:
        if len(line + word) < 18:
            line += word + " "
        else:
            lines.append(line)
            line = word + " "
    lines.append(line)

    y = 150
    for l in lines:
        draw.text((20, y), l.strip(), fill=(255, 255, 255), font=font)
        y += 35

    return np.array(img)


# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_parquet('processed_df.parquet')
    movies = pd.read_parquet('movie.parquet')
    metadata = pd.read_parquet('movie_metadata.parquet')

    movie_map = pickle.load(open('models/movie_map.pkl', 'rb'))

    df['movie_encoded'] = df['movieId'].map(movie_map)
    df['movie_encoded'] = df['movie_encoded'].fillna(0).astype(int)

    metadata_dict = metadata.set_index('movieId').to_dict('index')
    avg_rating = metadata['rating'].mean()

    return df, movies, metadata_dict, movie_map, avg_rating


# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    svd_model = pickle.load(open('models/svd_model.pkl', 'rb'))
    knn_model = pickle.load(open('models/knn_model.pkl', 'rb'))
    xgb_model = pickle.load(open('models/xgb_model.pkl', 'rb'))

    lstm_model = load_model('models/lstm_model.h5')

    feature_cols = pickle.load(open('models/feature_cols.pkl', 'rb'))
    user_features = pd.read_csv('models/user_features.csv', index_col='userId')

    kmeans_cluster_rating = pickle.load(open('models/kmeans_cluster_rating.pkl', 'rb'))
    dbscan_cluster_rating = pickle.load(open('models/dbscan_cluster_rating.pkl', 'rb'))

    return (svd_model, knn_model, xgb_model, lstm_model,
            feature_cols, user_features,
            kmeans_cluster_rating, dbscan_cluster_rating)


# ================= INIT =================
df, movies, metadata_dict, movie_map, avg_rating = load_data()

(svd_model, knn_model, xgb_model, lstm_model,
 feature_cols, user_features,
 kmeans_cluster_rating, dbscan_cluster_rating) = load_models()

best_weights = pickle.load(open("models/best_weights.pkl", "rb"))


# ================= MOVIE INFO =================
def get_movie_info(movie_id, title):
    row = metadata_dict.get(movie_id)

    if row is None:
        return create_title_poster(title), round(avg_rating, 2)

    poster = row.get('poster')
    rating = row.get('rating')

    # ✅ USE URL if exists
    if isinstance(poster, str) and poster.startswith("http"):
        final_poster = poster
    else:
        final_poster = create_title_poster(title)

    # ✅ SAFE RATING
    if pd.isna(rating):
        rating = avg_rating

    return final_poster, rating


# ================= RECOMMENDATION ENGINE =================
@st.cache_data
def get_recommendations(user_id, n, weights_tuple):

    weights = np.array(weights_tuple)

    if user_id not in df['userId'].values:
        return cold_start(user_id, df, movies, n)

    recs = final_hybrid(
        user_id=user_id,
        df=df,
        movies=movies,
        svd_model=svd_model,
        knn_model=knn_model,
        xgb_model=xgb_model,
        lstm_model=lstm_model,
        movie_map=movie_map,
        user_features=user_features,
        feature_cols=feature_cols,
        kmeans_cluster_rating=kmeans_cluster_rating,
        dbscan_cluster_rating=dbscan_cluster_rating,
        weights=weights,
        n=n
    )

    return recs


# ================= UI =================
st.title("🎬 Hybrid Movie Recommendation System")

user_id = st.number_input("Enter User ID", min_value=1, step=1)
n = st.slider("Number of Recommendations", 5, 20, 10)


# ================= BUTTON =================
if st.button("Recommend"):

    recs = get_recommendations(user_id, n, tuple(best_weights))

    if recs is None or recs.empty:
        st.warning("No recommendations found for this user.")
    else:
        st.subheader("Top Recommendations")

        cols = st.columns(min(n, 5))

        for i, (_, row) in enumerate(recs.iterrows()):
            col = cols[i % len(cols)]

            poster, rating = get_movie_info(row['movieId'], row['title'])

            with col:
                st.image(poster, use_container_width=True)
                st.caption(row['title'])
                st.write(f"⭐ {round(rating, 2)}")

                with st.expander("Details"):
                    st.write(f"Genres: {row.get('genres', 'N/A')}")
                    st.write(f"Hybrid Score: {round(row['score'], 3)}")