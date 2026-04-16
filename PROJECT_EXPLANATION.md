# 🎵 VibeTune: Next-Gen Music Recommendation System
## Project Documentation & Professor-Facing Guide

This document provides a comprehensive breakdown of the **VibeTune** project, covering its architecture, algorithms, data processing, and design choices. It is designed to help you explain every technical detail to your professor.

---

## 1. Project Overview
**VibeTune** is an advanced music recommendation platform that moves beyond simple keyword matching. It uses a **Hybrid Recommendation Engine** combining Deep Learning (Transformers), Content-Based Filtering (Cosine Similarity), and Contextual Awareness (Mood, Time, and Location) to provide highly personalized "vibes" rather than just song lists.

### Key Objectives:
- **Accuracy**: Provide relevant suggestions based on audio features.
- **Discovery**: Help users find new music through "Smart Search" and "Time-Based Discovery."
- **Persistence**: Save user history and playlists for a long-term personalized experience.
- **Professional UX**: A premium, dark-themed dashboard built with Streamlit.

---

## 2. Technical Stack
| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Frontend** | Streamlit | Rapidly built, modern, and interactive dashboard. |
| **Backend API** | FastAPI | RESTful API for serving recommendation logic. |
| **Deep Learning** | TensorFlow / Keras | Transformer model for sequence-based clustering. |
| **Machine Learning** | Scikit-Learn | Cosine similarity and feature scaling. |
| **Database** | SQLite | Persistent storage for History and Playlists. |
| **Data Handling** | Pandas / NumPy | Data manipulation and numerical operations. |
| **Containerization** | Docker | Ensuring the app runs consistently across all environments. |

---

## 3. System Architecture
VibeTune follows a modular architecture:
1.  **UI Layer (`ui/app.py`)**: The visual interface where users interact, search, and view analytics.
2.  **API Layer (`main.py`)**: A backend bridge that can expose the recommender to other apps.
3.  **Logic Layer (`recommender/`)**: 
    - `hybrid_recommender.py`: The heart of the system that merges all signals.
    - `mood_recommender.py`: Handles sentiment analysis of keywords to detect mood.
    - `time_pattern_model.py`: Analyzes temporal patterns to suggest music for specific times of day.
4.  **Data Layer (`utils/database.py` & `data/`)**: Manages the SQLite database and the primary CSV dataset.

---

## 4. The Recommendation Engine (Deep Dive)
If the professor asks **"How does your algorithm work?"**, you should explain the **Hybrid Approach**:

### A. Long-Term Patterns (Transformer Model)
We use a **Transformer-based Neural Network** (`transformer_model.h5`). 
- **The Concept**: Transformers are famous for handling sequences (like in GPT). In our case, we treat a user's listening sequence as "sentences."
- **Implementation**: The model predicts which "Cluster" (group of similar songs) the user is gravitating towards based on their current selection.

### B. Short-Term Similarity (Cosine Similarity)
- We represent each song as a vector based on its **Audio Features** (Danceability, Energy, Valence, etc.).
- When a user selects a song, we calculate the mathematical "distance" (Cosine Similarity) between that song and all other songs in the predicted cluster.
- **Score Formula**: $Score = (0.7 \times Similarity) + (0.3 \times TransformerPrediction) + (0.1 \times ArtistMatchWeight)$

### C. Contextual Filtering
- **Mood Detection**: If the user types "fast and energetic," the system detects the "Energetic" mood and filters the dataset to match those audio profiles.
- **Time-Based Discovery**: Automatically suggests "Morning" songs (higher energy/valence) or "Night" songs (lower energy/ambient) based on the user's current clock or preference.
- **Location Awareness**: Prioritizes genres popular in specific regions (e.g., Pop for India, Rock for USA).

---

## 5. Data Processing & Features
The system uses **Spotify Audio Features**:
- **Danceability**: How suitable a track is for dancing.
- **Energy**: Perceptual measure of intensity and activity.
- **Valence**: The "musical positiveness" (high valence = happy, low valence = sad).
- **Acousticness / Instrumentalness**: Measures of the physical nature of the sound.

**Preprocessing**: All features are normalized using a `StandardScaler` to ensure that one feature (like Tempo, which is large) doesn't dominate others (like Valence, which is small).

---

## 6. Database Schema
We use **SQLite** for lightweight, persistent storage:
1.  **`listening_history`**: Tracks every song clicked, its artist, and the timestamp.
2.  **`playlists`**: Stores user-created collections.
3.  **`playlist_tracks`**: A linking table that maps multiple songs to specific playlists.

---

## 7. Potential Professor Questions (Q&A)

**Q: Why use a Transformer instead of just simple K-Nearest Neighbors?**
> *Answer*: KNN is "lazy learning" and static. A Transformer can eventually learn the *order* in which people listen to music, understanding that "Morning Pop" often follows "Chill Lo-fi," making the system dynamic rather than just matching static features.

**Q: How do you handle "The Cold Start Problem"?**
> *Answer*: We use **Content-Based Filtering** (Cosine Similarity) as a fallback. Even if we have no history for a user, we can recommend songs based on the features of the song they just searched for.

**Q: What is the benefit of the FastAPI + Streamlit combination?**
> *Answer*: It follows the industry standard of separating the **Frontend** from the **Business Logic**. If we wanted to build a Mobile App later, we wouldn't need to rewrite the recommendation logic; the Mobile App could simply "call" the FastAPI backend.

**Q: What does "Valence" mean in your analytics?**
> *Answer*: Valence represents musical happiness. High valence sounds cheerful (Major keys, fast), while low valence sounds melancholy or serious (Minor keys, slower).

---

## 8. Summary of Unique Features
- **Visual Analytics**: Real-time progress bars showing the "energy" and "danceability" of your current playlist.
- **Smart Result Resolution**: Searching "workout" doesn't just look for songs with the word "workout"; it detects the mood and provides a curated high-energy list.
- **Regional Personalization**: Built-in support for location-based genre prioritization.

---

## 9. Deployment & Portability
- **Docker Support**: The project includes a `Dockerfile`, allowing it to be containerized. This means the professor can run the entire system (API + Frontend) with a single command without worrying about installing individual Python libraries.
- **REST Architecture**: By using FastAPI, the recommendation engine is decoupled from the UI, making it scalable and ready for cloud deployment (e.g., AWS, Heroku, or Azure).

---
*Created for VibeTune Project Evaluation*
