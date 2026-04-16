import pandas as pd
import numpy as np
import joblib
import os
import sys
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity

# Fix relative imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from recommender.time_pattern_model import get_time_cluster
from recommender.mood_recommender import get_mood_recommendations, detect_mood_from_text
from utils.database import log_history

# Paths
DATA_PATH = os.path.join(BASE_DIR, "data", "processed_music_dataset2.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load data and models
df = pd.read_csv(DATA_PATH)
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
model = load_model(os.path.join(MODELS_DIR, "transformer_model.h5"))

feature_cols = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

X_scaled = scaler.transform(df[feature_cols])

def get_search_suggestions(query, limit=10):
    """
    Returns a list of suggestions for autocomplete.
    """
    if not query or len(query) < 2:
        return []
    
    query = query.lower()
    
    # Check tracks
    track_matches = df[df['track_name'].str.lower().str.contains(query, na=False, regex=False)]['track_name'].unique().tolist()
    
    # Check artists
    artist_matches = df[df['artists'].str.lower().str.contains(query, na=False, regex=False)]['artists'].unique().tolist()
    
    # Check genres
    genre_matches = df[df['track_genre'].str.lower().str.contains(query, na=False, regex=False)]['track_genre'].unique().tolist()
    
    suggestions = list(set(track_matches[:limit] + artist_matches[:limit] + genre_matches[:limit]))
    return suggestions[:limit]

def smart_search(query):
    """
    Enhanced search: Song -> Artist -> Mood -> Genre
    """
    query = query.lower().strip()
    
    # 1. Direct or partial song match (priority) - Exact filter first
    exact_matches = df[df['track_name'].str.lower() == query]
    if not exact_matches.empty:
        return {"type": "song", "index": exact_matches.index[0], "name": exact_matches.iloc[0]['track_name']}

    partial_matches = df[df['track_name'].str.lower().str.contains(query, na=False, regex=False)]
    if not partial_matches.empty:
        # If it's a very short query, check if an artist match is more likely
        if len(query) > 3 or artist_matches.empty:
             return {"type": "song", "index": partial_matches.index[0], "name": partial_matches.iloc[0]['track_name']}
    
    # 2. Artist match
    artist_matches = df[df['artists'].str.lower().str.contains(query, na=False, regex=False)]
    if not artist_matches.empty:
        # Return the most popular artist name found
        artist_name = artist_matches.sort_values('popularity', ascending=False).iloc[0]['artists']
        return {"type": "artist", "artist": artist_name}

    # 3. Try detecting mood
    detected_mood = detect_mood_from_text(query)
    
    # 4. Try detecting genre
    detected_genre = None
    all_genres = df['track_genre'].unique()
    for g in all_genres:
        if str(g).lower() in query:
            detected_genre = g
            break
    
    if detected_mood:
        return {"type": "mood", "mood": detected_mood, "genre": detected_genre}
    
    # If we found song matches but they were short, return them as fallback
    if not partial_matches.empty:
        return {"type": "song", "index": partial_matches.index[0], "name": partial_matches.iloc[0]['track_name']}
        
    return {"type": "none"}

def get_comprehensive_search_results(query, location=None, limit=10):
    """
    Returns a unified list of search results including songs, artists, and genres.
    If location is provided, it prioritizes songs matching the local vibe.
    """
    if not query or len(query) < 2:
        return []

    query = query.lower().strip()
    
    # Remove common suffixes
    for suffix in [" song", " songs", " music", " track", " tracks"]:
        if query.endswith(suffix):
            query = query[:-len(suffix)].strip()
            
    results = []

    # 1. Search Songs
    song_matches = df[df['track_name'].str.lower().str.contains(query, na=False, regex=False)]
    
    # If location is India, prioritize pop/bollywood if relevant, etc.
    if location and location != "None":
        loc_genre = {"USA": "rock", "India": "pop", "UK": "indie", "Brazil": "latin"}.get(location)
        if loc_genre:
            # Sort so matches in the popular regional genre come first
            song_matches['is_loc'] = song_matches['track_genre'].str.lower() == loc_genre.lower()
            song_matches = song_matches.sort_values(['is_loc', 'popularity'], ascending=False)
    else:
        song_matches = song_matches.sort_values('popularity', ascending=False)

    for idx, row in song_matches.head(limit).iterrows():
        results.append({
            "type": "song",
            "name": row['track_name'],
            "artist": row['artists'],
            "index": idx,
            "display": f"🎵 {row['track_name']} - {row['artists']}"
        })

    # 2. Search Artists
    artist_matches = df[df['artists'].str.lower().str.contains(query, na=False, regex=False)].sort_values('popularity', ascending=False).head(5)
    for artist in artist_matches['artists'].unique():
        results.append({
            "type": "artist",
            "name": artist,
            "display": f"👤 Artist: {artist}"
        })

    # ... remaining logic for genres and mood
    genre_matches = df[df['track_genre'].str.lower().str.contains(query, na=False, regex=False)].head(5)
    for genre in genre_matches['track_genre'].unique():
        results.append({
            "type": "genre",
            "name": genre,
            "display": f"🌈 Genre/Vibe: {genre.capitalize()}"
        })

    detected_mood = detect_mood_from_text(query)
    if detected_mood:
        results.append({
            "type": "mood",
            "name": detected_mood,
            "display": f"🎭 Mood: {detected_mood.capitalize()}"
        })

    return results

def get_artist_recommendations(artist_name):
    """
    Return popular songs by a specific artist.
    """
    artist_df = df[df['artists'].str.lower() == artist_name.lower()]
    if artist_df.empty:
        # Fallback to fuzzy artist match
        artist_df = df[df['artists'].str.lower().str.contains(artist_name.lower(), na=False, regex=False)]
    
    # Sort by popularity or just return top
    artist_df = artist_df.sort_values(by='popularity', ascending=False).head(10)
    
    return [(row['track_name'], row['artists'], 0.95) for _, row in artist_df.iterrows()]

def mood_genre_recommend(mood, genre=None):
    recs = get_mood_recommendations(mood, genre)
    # Convert to the standard list of tuples format (name, artist, score)
    return [(r['track_name'], r['artist'], r['score']) for r in recs]

def hybrid_recommend(song_index):
    # Log to history
    song_data = df.iloc[song_index]
    log_history(song_data['track_name'], song_data['artists'])
    
    # Simple sequence for transformer demo
    seq = np.array([[song_index] * 5])
    pred = model.predict(seq)
    
    transformer_cluster = np.argmax(pred)
    transformer_score = float(np.max(pred))
    
    cluster_df = df[df['cluster'] == transformer_cluster]
    
    song_vector = X_scaled[song_index].reshape(1, -1)
    cluster_vectors = X_scaled[cluster_df.index]
    
    sim = cosine_similarity(song_vector, cluster_vectors)[0]
    
    input_artist = df.iloc[song_index]['artists']
    results = []
    
    for i, s in enumerate(sim):
        song = cluster_df.iloc[i]
        artist_match = 1 if song['artists'] == input_artist else 0
        score = 0.7 * s + 0.3 * transformer_score + 0.1 * artist_match
        results.append((song['track_name'], song['artists'], score))
        
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:10]

def apply_time_filter(recs, selected_time):
    cluster = get_time_cluster(selected_time)
    if cluster is None:
        return recs
        
    filtered = []
    for r in recs:
        row = df[df['track_name'] == r[0]]
        if not row.empty and row['cluster'].values[0] == cluster:
            filtered.append(r)
            
    return filtered if filtered else recs

def get_time_based_discovery(selected_time, location=None, limit=6):
    """
    Get random discovery songs that suit the selected time of day and location.
    """
    if not selected_time or selected_time == "None":
        return []

    # Audio profiles
    time_map = {
        "Morning": {"energy": (0.6, 1.0), "valence": (0.5, 1.0)},
        "Afternoon": {"energy": (0.4, 0.8), "valence": (0.4, 0.8)},
        "Evening": {"energy": (0.2, 0.6), "valence": (0.2, 0.7)},
        "Night": {"energy": (0.0, 0.4), "valence": (0.0, 0.5)}
    }
    profile = time_map.get(selected_time)
    if not profile:
        return []
    
    # Filter by audio profile
    mask = (df['energy'] >= profile['energy'][0]) & (df['energy'] <= profile['energy'][1]) & \
           (df['valence'] >= profile['valence'][0]) & (df['valence'] <= profile['valence'][1])
    
    filtered = df[mask]

    # Additionally filter/prioritize by location if provided
    if location and location != "None":
        loc_genre = {"USA": "rock", "India": "pop", "UK": "indie", "Brazil": "latin"}.get(location)
        if loc_genre:
            loc_filtered = filtered[filtered['track_genre'].str.lower() == loc_genre.lower()]
            if not loc_filtered.empty:
                filtered = loc_filtered

    if filtered.empty:
        filtered = df.sort_values('popularity', ascending=False).head(100)
        
    discovery = filtered.sample(min(limit, len(filtered)))
    return [(row['track_name'], row['artists'], 0.85) for _, row in discovery.iterrows()]

def get_location_recommendations(location):
    """
    Simulate location-based recommendations.
    In real apps, use IP/GPS and correlate with popular trends in the region.
    """
    location_map = {
        "USA": "rock",
        "India": "pop",
        "UK": "indie",
        "Brazil": "latin"
    }
    
    genre = location_map.get(location, "pop")
    # Filter by popular songs in that genre
    filtered = df[df['track_genre'].str.lower() == genre.lower()].head(10)
    return [(row['track_name'], row['artists'], 0.9) for _, row in filtered.iterrows()]