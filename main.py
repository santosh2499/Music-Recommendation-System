from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from recommender import hybrid_recommender as hr

app = FastAPI(title="Music Recommendation API")

class RecommendationRequest(BaseModel):
    song_names: list[str]

@app.get("/")
async def root():
    return {"message": "Welcome to the Music Recommendation API"}

@app.post("/recommend")
async def get_recommendations(request: RecommendationRequest):
    try:
        # Load dataset to map names to indices
        data_path = os.path.join(BASE_DIR, "data", "processed_music_dataset2.csv")
        df = pd.read_csv(data_path)
        
        # Find indices for requested songs
        indices = df[df['track_name'].isin(request.song_names)].index.tolist()
        
        if not indices:
            raise HTTPException(status_code=404, detail="Songs not found in dataset")
            
        # Get recommendations for the first valid song (standard hybrid behavior)
        recs = hr.hybrid_recommend(indices[0])
        
        # Determine metadata from the first song
        song_info = df.iloc[indices[0]]
        genre = song_info.get('track_genre', 'Unknown')
        mood = song_info.get('mood', 'Unknown')
        
        formatted_recs = [
            {"track_name": r[0], "artist": r[1], "score": float(r[2])}
            for r in recs
        ]
        
        return {
            "query_song": song_info['track_name'],
            "detected_genre": genre,
            "detected_mood": mood,
            "recommendations": formatted_recs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
