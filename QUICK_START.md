# Movie Recommendation System - Quick Start Guide

This guide shows you how to use the recommendation system with your local dataset (no API required).

## Setup

1. **Activate virtual environment:**
```powershell
.\venv\Scripts\Activate.ps1
```

2. **Place your data files in the `data/` folder:**
   - `movies.csv` - Your movie dataset
   - `ratings.csv` - User ratings (optional)

## Expected Data Format

### movies.csv
Should contain columns like:
- Movie ID (e.g., `imdb_id`, `movieId`)
- `title`
- `year`
- `genre`
- `imdb_rating` or `rating`
- `director`, `actors`, `plot` (optional but recommended)

### ratings.csv (optional)
Should contain:
- `user_id` - User identifier
- `movie_id` - Movie identifier
- `rating` - Rating value (1-10 or 1-5 scale)

## Running the Pipeline

### Option 1: Run Complete Pipeline
```powershell
python pipeline.py
```

This will:
1. Load your data from the `data/` folder
2. Preprocess and create features
3. Build all recommenders (collaborative, content-based, hybrid)
4. Generate sample recommendations
5. Show explanations and evaluations

### Option 2: Use Individual Components

#### Load Data
```python
from src.data_loader import DatasetLoader

loader = DatasetLoader("data")
movies_df = loader.load_movies("movies.csv")
ratings_df = loader.load_ratings("ratings.csv")
```

#### Preprocess Data
```python
from src.preprocessing import MovieDataProcessor

processor = MovieDataProcessor()
clean_movies = processor.clean_data(movies_df)
processed = processor.create_content_features(clean_movies)
tfidf_matrix, _ = processor.create_tfidf_matrix(processed)
```

#### Get Recommendations
```python
from src.recommenders.content_based import ContentBasedRecommender

recommender = ContentBasedRecommender(processed, tfidf_matrix)

# Find similar movies
similar = recommender.get_similar_movies('movie_id_here', n=10)

# Get recommendations for user
user_liked = ['movie1', 'movie2', 'movie3']
recs = recommender.recommend_for_user(user_liked, n_recommendations=10)
```

#### With Collaborative Filtering (requires ratings)
```python
from src.recommenders.collaborative_filtering import CollaborativeFilteringRecommender

user_item_matrix = processor.create_user_item_matrix(ratings_df)
cf = CollaborativeFilteringRecommender(user_item_matrix)

# Get recommendations
recs = cf.recommend_user_based('user_1', n_recommendations=10)
```

#### Hybrid Recommendations
```python
from src.recommenders.hybrid import HybridRecommender

hybrid = HybridRecommender(cf_recommender=cf, cb_recommender=recommender)

# Weighted combination
recs = hybrid.weighted_hybrid(
    'user_1',
    cf_weight=0.6,
    cb_weight=0.4,
    user_liked_movies=user_liked
)
```

## Testing with Sample Data

If you don't have data yet, create sample datasets:

```python
from src.data_loader import DatasetLoader

loader = DatasetLoader("data")

# Create sample movies and ratings
loader.create_sample_dataset(n_movies=100)
loader.create_sample_ratings(
    movie_ids=[f'tt{str(i).zfill(7)}' for i in range(1, 101)],
    n_users=50
)
```

## Quick Test Commands

```powershell
# Test data loader
python -m src.data_loader

# Test preprocessing
python -m src.preprocessing

# Test recommenders
python -m src.recommenders.content_based
python -m src.recommenders.collaborative_filtering

# Run full pipeline
python pipeline.py
```

## Troubleshooting

**Missing data files?**
- Make sure files are in the `data/` folder
- Check column names match expected format
- Run `python -m src.data_loader` to create sample data

**Import errors?**
- Make sure virtual environment is activated
- Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn scipy`

**No recommendations generated?**
- Check if movie IDs in ratings match movie IDs in movies.csv
- Ensure data is properly formatted
- Try with sample data first to verify system works
