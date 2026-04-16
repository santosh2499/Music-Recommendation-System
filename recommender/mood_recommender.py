import pandas as pd
import os
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Constants
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed_music_dataset2.csv")
SCALER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "scaler.pkl")

# Load static data
df = pd.read_csv(DATA_PATH)
scaler = joblib.load(SCALER_PATH)

# Feature columns
feature_cols = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

# Scale features once
X_scaled = scaler.transform(df[feature_cols])

def get_mood_recommendations(mood, genre=None, top_n=10):
    """
    Find songs that match the given mood and optional genre.
    """
    # Filter by mood
    if mood:
        filtered = df[df['mood'].str.lower() == mood.lower()]
    else:
        filtered = df.copy()
    
    if genre:
        genre_filter = filtered[filtered['track_genre'].str.lower() == genre.lower()]
        if not genre_filter.empty:
            filtered = genre_filter
    
    if len(filtered) == 0:
        return []

    # Calculate average vector for the filtered set to find representative songs
    indices = filtered.index
    vectors = X_scaled[indices]
    
    # Use the mean vector as the "target" for this mood context
    target_vector = vectors.mean(axis=0).reshape(1, -1)
    
    # Calculate similarity to all songs in the filtered set
    sim = cosine_similarity(target_vector, vectors)[0]
    
    # Prepare results
    results = []
    for i, s in enumerate(sim):
        song = filtered.iloc[i]
        results.append({
            "track_id": song.name,
            "track_name": song['track_name'],
            "artist": song['artists'],
            "score": float(s)
        })
    
    # Sort and return top N
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_n]

def detect_mood_from_text(text):
    """
    Detect mood from user input text.
    """
    text = text.lower()
    
    mood_keywords = {
        "Happy": ["happy", "joy", "cheerful", "upbeat", "celebrate", "party", "good vibes"],
        "Sad": ["sad", "depressed", "lonely", "breakup", "crying", "emotional", "sorry"],
        "Energetic": ["workout", "energetic", "gym", "fast", "power", "hype", "running"],
        "Calm": ["relax", "chill", "sleep", "calm", "focus", "study", "peace", "soft"],
        "Romantic": ["love", "romance", "date", "romantic", "sweetheart", "kiss"]
    }
    
    for mood, keywords in mood_keywords.items():
        if any(keyword in text for keyword in keywords):
            return mood
    
    return None
