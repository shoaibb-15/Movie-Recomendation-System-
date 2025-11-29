import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# ===== Load data =====
movies = pd.read_csv("movies.csv")  # your metadata file
ratings = pd.read_csv("ratings.csv")  # user-movie interactions

# (Assume you already computed cosine_sim and svd_reconstructed earlier)
# If not, you can just fake a similarity matrix for demo
if 'cosine_sim' not in locals():
    cosine_sim = np.random.rand(len(movies), len(movies))

# normalize SVD reconstructed matrix (demo)
scaler = MinMaxScaler()
svd_norm = scaler.fit_transform(np.random.rand(50, len(movies)))  # pretend 50 users

# ===== Streamlit App =====
st.title("🎥 Personalized Movie Recommendation System")
st.write("Get movie suggestions based on your preferences and viewing history.")

# sidebar for user input
user_id = st.sidebar.number_input("Enter User ID", min_value=1, max_value=50, value=10)
movie_choice = st.sidebar.selectbox("Pick a Movie You Like", movies["title"].values)

alpha = st.sidebar.slider("Collaborative Weight (α)", 0.0, 1.0, 0.6)
beta = st.sidebar.slider("Content Weight (β)", 0.0, 1.0, 0.4)
gamma = st.sidebar.slider("Feedback Boost (γ)", 0.0, 0.5, 0.1)

# ===== Hybrid Recommendation Function =====
def hybrid_recommend(selected_movie, user_index, alpha=0.6, beta=0.4, gamma=0.1, top_n=10):
    collab_scores = svd_norm[user_index]
    idx = movies[movies['title'] == selected_movie].index[0]
    content_scores = cosine_sim[idx]
    
    final_scores = alpha * collab_scores + beta * content_scores
    top_indices = final_scores.argsort()[::-1][1:top_n+1]
    return movies.iloc[top_indices][['title']]

# ===== Show Recommendations =====
if st.button("Show Recommendations"):
    st.subheader(f"🎬 Recommended Movies for User {user_id}")
    recs = hybrid_recommend(movie_choice, user_index=user_id, alpha=alpha, beta=beta, gamma=gamma)
    st.table(recs)

# ===== Optional Feedback =====
st.write("👍 **Rate Recommendations**")
feedback_movie = st.selectbox("Select movie to rate:", movies["title"].values[:20])
feedback = st.radio("Did you like it?", ["👍 Yes", "👎 No"])

if st.button("Submit Feedback"):
    fb_value = 1 if feedback == "👍 Yes" else 0
    new_entry = pd.DataFrame([[user_id, feedback_movie, fb_value]], columns=["user_id", "movie_title", "feedback"])
    new_entry.to_csv("user_feedback.csv", mode='a', header=False, index=False)
    st.success(f"Feedback saved for {feedback_movie} ({'Liked' if fb_value else 'Disliked'})")
