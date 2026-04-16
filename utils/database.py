import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "music_app.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # User History Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS listening_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        track_name TEXT,
        artist TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        mood TEXT,
        period TEXT
    )
    ''')
    
    # Playlists Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS playlists (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Playlist Tracks Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS playlist_tracks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        playlist_id INTEGER,
        track_name TEXT,
        artist TEXT,
        FOREIGN KEY (playlist_id) REFERENCES playlists (id)
    )
    ''')
    
    conn.commit()
    conn.close()

def log_history(track_name, artist, mood=None, period=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO listening_history (track_name, artist, mood, period) VALUES (?, ?, ?, ?)",
        (track_name, artist, mood, period)
    )
    conn.commit()
    conn.close()

def get_history(limit=10):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM listening_history ORDER BY timestamp DESC LIMIT ?", (limit,))
    history = cursor.fetchall()
    conn.close()
    return history

def create_playlist(name, tracks):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO playlists (name) VALUES (?)", (name,))
    playlist_id = cursor.lastrowid
    
    for track in tracks:
        # track is (track_name, artist)
        cursor.execute(
            "INSERT INTO playlist_tracks (playlist_id, track_name, artist) VALUES (?, ?, ?)",
            (playlist_id, track[0], track[1])
        )
    
    conn.commit()
    conn.close()
    return playlist_id

def get_playlists():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM playlists ORDER BY created_at DESC")
    playlists = cursor.fetchall()
    conn.close()
    return playlists

def get_playlist_tracks(playlist_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM playlist_tracks WHERE playlist_id = ?", (playlist_id,))
    tracks = cursor.fetchall()
    conn.close()
    return tracks

if __name__ == "__main__":
    init_db()
