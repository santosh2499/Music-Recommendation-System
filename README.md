# 🎵 VibeTune: Next-Gen Music Recommendation System

VibeTune is an advanced music recommendation platform that moves beyond simple keyword matching. It uses a **Hybrid Recommendation Engine** combining Deep Learning (Transformers), Content-Based Filtering (Cosine Similarity), and Contextual Awareness (Mood, Time, and Location) to provide highly personalized "vibes" rather than just song lists.

## 🚀 Key Features
- **Accuracy**: Provide relevant suggestions based on audio features.
- **Discovery**: Help users find new music through "Smart Search" and "Time-Based Discovery."
- **Persistence**: Save user history and playlists for a long-term personalized experience.
- **Professional UX**: A premium, dark-themed dashboard built with Streamlit.

## 🛠️ Technical Stack
- **Frontend**: Streamlit
- **Backend API**: FastAPI
- **Deep Learning**: TensorFlow / Keras (Transformer model)
- **Machine Learning**: Scikit-Learn
- **Database**: SQLite
- **Containerization**: Docker

## 📂 Project Structure
- `ui/`: Streamlit dashboard
- `main.py`: FastAPI backend
- `recommender/`: Core recommendation logic
- `models/`: Trained model files
- `data/`: Datasets
- `utils/`: Database and utility scripts

## 🚦 Getting Started

### Prerequisites
- Python 3.9+
- Docker (optional)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/VibeTune.git
   cd VibeTune
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the App
1. Start the FastAPI backend:
   ```bash
   python main.py
   ```
2. Run the Streamlit UI:
   ```bash
   streamlit run ui/app.py
   ```

## 🐳 Docker Support
Build and run using Docker:
```bash
docker build -t vibetune .
docker run -p 8501:8501 vibetune
```

## 📄 Documentation
For a detailed technical breakdown, see [PROJECT_EXPLANATION.md](PROJECT_EXPLANATION.md).
