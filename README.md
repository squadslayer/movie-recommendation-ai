# Movie Recommendation System

A complete movie recommendation system built with Python, using **local CSV datasets** for data storage and processing.

## Features

- 🎬 **Multiple Recommendation Algorithms**
  - Collaborative Filtering (User-based, Item-based, Matrix Factorization)
  - Content-Based Filtering (TF-IDF similarity)
  - Hybrid Approaches (Weighted, Switching, Cascade)

- 📊 **Comprehensive Evaluation**
  - RMSE, MAE for rating predictions
  - Precision@k, Recall@k, F1@k
  - NDCG, Hit Rate, Coverage

- 💡 **Explainability**
  - Human-readable explanations
  - Feature importance visualization
  - Similar user/item insights

- 📁 **Dataset-Based Workflow**
  - No API required - works offline
  - Uses CSV files from `data/` folder
  - Supports MovieLens and TMDB formats

## Quick Start

### 1. Setup Environment

```powershell
# Activate virtual environment
.\venv\Scripts\activate

# Verify dependencies (already installed)
pip list
```

### 2. Add Your Data

Place your movie dataset in the `data/` folder:
- `movies.csv` - Your movie data
- `ratings.csv` - User ratings (optional)

**Your current data:**
- ✅ `tmdb_top_rated_movies.csv` (3.2 MB) - Ready to use!

### 3. Run the Pipeline

```powershell
python pipeline.py
```

This will:
1. Load data from `data/` folder
2. Preprocess and extract features
3. Build all recommenders
4. Generate sample recommendations
5. Show explanations and evaluations

## Project Structure

```
movie-recommender/
├── data/                    # Put your CSV files here
│   └── tmdb_top_rated_movies.csv
├── src/
│   ├── data_loader.py       # Load CSV datasets
│   ├── preprocessing.py     # Data cleaning & features
│   ├── evaluation.py        # Metrics
│   ├── explainability.py    # Explanations
│   └── recommenders/
│       ├── collaborative_filtering.py
│       ├── content_based.py
│       └── hybrid.py
├── backend/                 # FastAPI endpoints (placeholder)
├── frontend/                # React app
├── models/                  # Saved models
├── pipeline.py              # End-to-end workflow
├── QUICK_START.md           # Detailed usage guide
└── venv/                    # Virtual environment
```

## Usage Examples

### Load and Recommend

```python
from src.data_loader import DatasetLoader
from src.preprocessing import MovieDataProcessor
from src.recommenders.content_based import ContentBasedRecommender

# Load data
loader = DatasetLoader("data")
movies_df = loader.load_movies("tmdb_top_rated_movies.csv")

# Process
processor = MovieDataProcessor()
movies_df = processor.clean_data(movies_df)
movies_df = processor.create_content_features(movies_df)
tfidf_matrix, _ = processor.create_tfidf_matrix(movies_df)

# Recommend
recommender = ContentBasedRecommender(movies_df, tfidf_matrix)
user_liked = ['movie_id_1', 'movie_id_2']
recommendations = recommender.recommend_for_user(user_liked, n_recommendations=10)
```

### Find Similar Movies

```python
similar = recommender.get_similar_movies('movie_id', n=10)
for movie_id, score in similar:
    info = recommender.get_movie_info(movie_id)
    print(f"{info['title']} - Score: {score:.3f}")
```

## Testing

```powershell
# Test individual components
python -m src.data_loader
python -m src.preprocessing
python -m src.recommenders.content_based

# Run full pipeline
python pipeline.py
```

## Dependencies

All dependencies are already installed in the virtual environment:
- pandas
- numpy
- scikit-learn
- scipy
- matplotlib
- seaborn
- requests
- python-dotenv

## Documentation

- [Quick Start Guide](QUICK_START.md) - Detailed usage instructions
- [Walkthrough](C:\Users\Krishna Sharma\.gemini\antigravity\brain\3f69bd5d-3e87-4a1a-9480-9043a0675b3e\walkthrough.md) - Implementation details

## Next Steps

1. **Use Your TMDB Data**: Update `pipeline.py` to use `tmdb_top_rated_movies.csv`
2. **Backend API**: Implement FastAPI endpoints in `backend/app.py`
3. **Frontend**: Connect React app to backend
4. **Deploy**: Set up production environment

## Status

✅ Project Setup Complete  
✅ All ML Components Implemented  
✅ Pipeline Tested Successfully  
✅ Dataset-Only Workflow Ready  
✅ Ready for Backend Integration
