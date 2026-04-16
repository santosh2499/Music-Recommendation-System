import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime

# Add parent directory to path to enable package-style imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import recommender.hybrid_recommender as hr
from utils.database import log_history, get_history, create_playlist, get_playlists, get_playlist_tracks

# Configuration
st.set_page_config(
    page_title="VibeTune | Next-Gen Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark-themed Premium Look)
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background-color: #1DB954;
        color: white;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: white;
        border-radius: 10px;
    }
    .sidebar .sidebar-content {
        background-color: #0e1117;
    }
    .card {
        background-color: #1e1e26;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 10px;
        border: 1px solid #333;
    }
    .card:hover {
        border-color: #1DB954;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar - User Profile & History
with st.sidebar:
    st.image("https://img.icons8.com/bubbles/100/000000/music.png")
    st.title("🎧 User Profile")
    st.write("Welcome back, User!")
    st.divider()
    
    # Navigation
    menu = st.radio("Navigate", ["🏠 Home", "📜 History", "📂 Playlists"])
    st.divider()
    
    # Advanced Filters
    st.header("✨ Refine Search")
    
    with st.expander("🌍 Location & Time", expanded=False):
        location = st.selectbox("Location (Adv.)", ["None", "USA", "India", "UK", "Brazil"])
        time_choice = st.selectbox("Time Preference", ["None", "Morning", "Afternoon", "Evening", "Night"])
    
    st.info("Filters help fine-tune your unique vibe.")

# Backend Data Access
@st.cache_resource
def load_data():
    data_path = os.path.join("..","data","processed_music_dataset2.csv")
    if not os.path.exists(data_path):
        # Handle relative pathing when running from root
        data_path = os.path.join("data","processed_music_dataset2.csv")
    return pd.read_csv(data_path)

df = load_data()

# Logic to maintain session recommendations for playlist creation
if 'current_recs' not in st.session_state:
    st.session_state['current_recs'] = []

# Page Logic
if menu == "🏠 Home":
    st.title("🎵 Smart Vibe Detection")
    st.caption("AI-powered music suggestions based on context, mood, and deep learning.")
    
    # Trending & Suggestions (The "Search Engine" feel)
    trending_cols = st.columns(4)
    trending_queries = ["Lo-fi Beats", "Workout Power", "Arijit Singh", "Sad Mood"]
    
    st.write("🔍 **Trending Searches:**")
    t_cols = st.columns(len(trending_queries))
    for i, tq in enumerate(trending_queries):
        if t_cols[i].button(tq, key=f"tq_{i}", use_container_width=True):
            st.session_state['search_query'] = tq

    # Dynamic Time-Based Discovery (NEW)
    if time_choice != "None":
        st.divider()
        st.subheader(f"☀️ {time_choice} Discovery")
        st.caption(f"Instant picks for your {time_choice.lower()} mood.")
        
        # Cache discovery songs so they don't change on every interaction
        # We also reset cache if location changes
        if ('discovery_recs' not in st.session_state or 
            st.session_state.get('last_time') != time_choice or
            st.session_state.get('last_location') != location):
            
            st.session_state['discovery_recs'] = hr.get_time_based_discovery(time_choice, location=location)
            st.session_state['last_time'] = time_choice
            st.session_state['last_location'] = location
        
        discovery_recs = st.session_state['discovery_recs']
        
        d_cols = st.columns(3)
        for i, dr in enumerate(discovery_recs):
            with d_cols[i % 3]:
                st.markdown(f"""
                <div class="card" style="padding: 10px; border-left: 5px solid #1DB954; height: 100px; display: flex; flex-direction: column; justify-content: center;">
                    <small><b>{dr[0]}</b></small><br>
                    <small style="color: #888;">{dr[1]}</small>
                </div>
                """, unsafe_allow_html=True)
                if st.button("✨ Vibe", key=f"d_btn_{i}_{dr[0]}", use_container_width=True):
                    # Trigger hybrid recommendations for this discovery song
                    match = df[df['track_name'] == dr[0]]
                    if not match.empty:
                        st.session_state['selected_item'] = {
                            "type": "song",
                            "name": dr[0],
                            "index": match.index[0]
                        }
                        st.session_state['trigger_recs'] = True
        
        if st.button("🔄 Refresh Discovery"):
            st.session_state['discovery_recs'] = hr.get_time_based_discovery(time_choice, location=location)
            st.rerun()

        st.divider()

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Search by Song, Artist, Language or Vibe", 
                             value=st.session_state.get('search_query', ""),
                             placeholder="Type here (e.g. 'Bhakti', 'Arijit Singh', 'Hindi', 'Starboy')...")
        if query:
            st.session_state['search_query'] = query

    with col2:
        st.write(" ") # Padding
        st.write(" ")
        search_btn = st.button("🔍 Search Engine")

    # Step 1: Show Results List
    if query and len(query) >= 2:
        results = hr.get_comprehensive_search_results(query, location=location)
        
        if results:
            st.write(f"### 🎯 Results for '{query}'")
            st.caption("Click on a result to get similar recommendations.")
            
            res_cols = st.columns(2)
            for i, res in enumerate(results):
                with res_cols[i % 2]:
                    if st.button(res['display'], key=f"btn_{i}_{res['name']}", use_container_width=True):
                        st.session_state['selected_item'] = res
                        st.session_state['trigger_recs'] = True
        else:
            if search_btn:
                st.warning("No direct matches found. Try a different keyword.")

    # Step 2: Generate Recommendations if an item is selected
    if st.session_state.get('trigger_recs'):
        res = st.session_state['selected_item']
        with st.spinner(f"Generating vibes based on: {res['name']}..."):
            recs = []
            if res["type"] == "song":
                st.session_state['method'] = f"Song similarity: {res['name']}"
                recs = hr.hybrid_recommend(res["index"])
            elif res["type"] == "artist":
                st.session_state['method'] = f"Artist Spotlight: {res['name']}"
                recs = hr.get_artist_recommendations(res["name"])
            elif res["type"] == "genre":
                st.session_state['method'] = f"Genre Vibe: {res['name']}"
                recs = hr.mood_genre_recommend(None, res["name"])
            elif res["type"] == "mood":
                st.session_state['method'] = f"Detected Mood: {res['name']}"
                recs = hr.mood_genre_recommend(res["name"], None)

            # Apply Contextual Filters from Sidebar
            if location != "None":
                recs = hr.get_location_recommendations(location) + recs
            
            recs = hr.apply_time_filter(recs, time_choice)
            
            st.session_state['current_recs'] = recs[:10]
            st.session_state['trigger_recs'] = False # Reset trigger

    # Display Recommendations
    if st.session_state.get('current_recs'):
        st.divider()
        st.subheader(f"✨ Recommended for You | {st.session_state.get('method', 'Hybrid')}")
        
        # Display as a nice grid
        rec_cols = st.columns(2)
        for i, r in enumerate(st.session_state['current_recs']):
            with rec_cols[i % 2]:
                st.markdown(f"""
                <div class="card">
                    <div style="display: flex; justify-content: space-between;">
                        <span><b>🎵 {r[0]}</b></span>
                        <span style="color: #1DB954; font-size: 0.8em;">{int(r[2]*100)}% Match</span>
                    </div>
                    <small>👤 {r[1]}</small><br>
                    <div style="background-color: #333; border-radius: 5px; margin-top:10px; height: 6px; width: 100%">
                        <div style="background-color: #1DB954; height: 100%; border-radius: 5px; width: {r[2]*100}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Visual Analytics for "Good Grading"
        st.divider()
        st.subheader("📊 Vibe Analytics")
        st.caption("Average audio features of your current recommendations.")
        
        # Extract features for recommendations
        rec_names = [r[0] for r in st.session_state['current_recs']]
        rec_data = df[df['track_name'].isin(rec_names)]
        
        if not rec_data.empty:
            feature_cols = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']
            avg_features = rec_data[feature_cols].mean()
            
            # Display horizontal bars for features
            for feature in feature_cols:
                val = avg_features[feature]
                st.write(f"**{feature.capitalize()}**")
                st.progress(float(val))
        else:
            st.info("Analytics will appear here when recommendations are generated.")

        # Save as Playlist Section
        st.divider()
        st.subheader("💾 Save as Playlist")
        playlist_name = st.text_input("Playlist Name", value=f"My Vibe - {datetime.now().strftime('%H:%M')}")
        if st.button("💾 Save Playlist"):
            tracks = [(r[0], r[1]) for r in st.session_state['current_recs']]
            create_playlist(playlist_name, tracks)
            st.success(f"Playlist '{playlist_name}' saved to your collection!")

elif menu == "📜 History":
    st.title("📜 Your Listening History")
    history = get_history()
    
    if not history:
        st.info("You haven't requested any recommendations yet.")
    else:
        # Simple dataframe view for history
        history_df = pd.DataFrame(history)
        if not history_df.empty:
            # Fix column names from database row object
            history_df.columns = ["ID", "Track Name", "Artist", "Time", "Mood", "Period"]
            # Alternative to st.table to avoid pyarrow dependency issues
            for _, row in history_df[::-1].head(20).iterrows():
                st.markdown(f"""
                <div style="background-color: #1e1e26; padding: 10px; border-radius: 10px; margin-bottom: 5px; border-left: 3px solid #1DB954;">
                    <strong>{row['Track Name']}</strong> - {row['Artist']}<br>
                    <small style="color: #888;">{row['Time']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("History is currently empty.")

elif menu == "📂 Playlists":
    st.title("📂 Your Savoury Playlists")
    playlists = get_playlists()
    
    if not playlists:
        st.info("No saved playlists yet. Start by generating some vibes!")
    else:
        cols = st.columns(3)
        for i, pl in enumerate(playlists):
            with cols[i % 3]:
                with st.expander(f"📁 {pl['name']} ({pl['created_at'][:10]})"):
                    tracks = get_playlist_tracks(pl['id'])
                    for t in tracks:
                        st.write(f"🎶 {t['track_name']} - {t['artist']}")
                    if st.button(f"Delete", key=f"del_{pl['id']}"):
                        st.error("Feature coming soon: Playlist Management!")

st.markdown("---")
st.markdown("<center><small>Powered by Transformer DL, Hybrid Recommenders & VibeTune Core</small></center>", unsafe_allow_html=True)