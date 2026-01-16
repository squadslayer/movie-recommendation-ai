# How to Check the Model is Working

Here are **3 simple ways** to verify your recommendation system is functioning correctly:

---

## ✅ Method 1: Quick Test Script (EASIEST)

Run the verification script I just created:

```powershell
# Activate virtual environment
.\venv\Scripts\activate

# Run test
python test_model.py
```

**What it tests:**
- ✅ Loads your TMDB data
- ✅ Preprocesses and creates features
- ✅ Builds recommender
- ✅ Generates similar movie recommendations
- ✅ Tests genre-based filtering

**Expected output:** "🎉 ALL TESTS PASSED!"

---

## ✅ Method 2: Full Pipeline Test

Run the complete end-to-end pipeline:

```powershell
.\venv\Scripts\activate
python pipeline.py
```

**What it does:**
1. Loads data from `data/` folder
2. Cleans and preprocesses
3. Builds ALL recommenders (CF, CB, Hybrid)
4. Generates recommendations
5. Shows explanations
6. Evaluates performance

---

## ✅ Method 3: Interactive Python Test

Try the recommender interactively:

```powershell
.\venv\Scripts\activate
python
```

Then in Python:

```python
from src.data_loader import DatasetLoader
from src.preprocessing import MovieDataProcessor
from src.recommenders.content_based import ContentBasedRecommender

# Load your TMDB data
loader = DatasetLoader("data")
movies_df = loader.load_movies("tmdb_top_rated_movies.csv")

print(f"Loaded {len(movies_df)} movies")
print(f"Columns: {movies_df.columns.tolist()}")

# Show first few movies
print(movies_df.head())

# Preprocess
processor = MovieDataProcessor()
clean_df = processor.clean_data(movies_df)
processed_df = processor.create_content_features(clean_df)
tfidf_matrix, _ = processor.create_tfidf_matrix(processed_df)

# Build recommender
recommender = ContentBasedRecommender(processed_df, tfidf_matrix)

# Get first movie ID (adjust column name if needed)
first_movie_id = processed_df.iloc[0]['id']  # or 'tmdb_id' or whatever your ID column is named

# Get info
info = recommender.get_movie_info(first_movie_id)
print(f"\nTest movie: {info['title']}")

# Get similar movies
similar = recommender.get_similar_movies(first_movie_id, n=5)

print("\nSimilar movies:")
for movie_id, score in similar:
    info = recommender.get_movie_info(movie_id)
    print(f"  {info['title']} - Score: {score:.3f}")
```

---

## 🔍 What to Look For

When testing, you should see:

### ✅ Good Signs:
- Data loads without errors
- Numbers make sense (e.g., 100+ movies loaded)
- Similarity scores between 0 and 1
- Recommended movies have similar genres/themes
- No crashes or exceptions

### ❌ Potential Issues:
- **ImportError**: Activate venv first (`.\venv\Scripts\activate`)
- **FileNotFoundError**: Check data file exists in `data/` folder
- **KeyError**: Column names don't match (check your CSV columns)
- **Empty recommendations**: May need to adjust data format

---

## 📊 Sample Expected Output

When `test_model.py` runs successfully:

```
==================================================================
MOVIE RECOMMENDATION SYSTEM - VERIFICATION TEST
==================================================================

Step 1: Loading TMDB movie data...
✅ Loaded 500 movies
   Columns: id, title, genre, rating, ...

Step 2: Sample movie data
[Shows first 3 movies]

Step 3: Preprocessing data...
✅ Cleaned 500 movies
✅ Created content features
✅ TF-IDF matrix shape: (500, 1000)

Step 4: Building content-based recommender...
✅ Recommender ready

Step 5: Testing recommendations
✅ Top 5 similar movies:
   1. Movie Title A - Similarity: 0.851
   2. Movie Title B - Similarity: 0.743
   ...

==================================================================
🎉 ALL TESTS PASSED! Your recommendation system is working!
==================================================================
```

---

## 🐛 Troubleshooting

If tests fail:

1. **Check Python environment:**
   ```powershell
   .\venv\Scripts\activate
   python --version  # Should be 3.x
   ```

2. **Verify data file:**
   ```powershell
   ls data\tmdb_top_rated_movies.csv  # Should exist
   ```

3. **Check data format:**
   ```python
   import pandas as pd
   df = pd.read_csv("data/tmdb_top_rated_movies.csv")
   print(df.columns)  # See what columns you have
   print(df.head())   # See sample data
   ```

4. **Re-run individual components:**
   ```powershell
   python -m src.data_loader        # Test data loading
   python -m src.preprocessing      # Test preprocessing
   python -m src.recommenders.content_based  # Test recomm
ender
   ```

---

## ✨ Quick Commands Cheat Sheet

```powershell
# Activate environment
.\venv\Scripts\activate

# Quick test (recommended)
python test_model.py

# Full pipeline
python pipeline.py

# Test individual modules
python -m src.data_loader
python -m src.recommenders.content_based
```

---

Your system is ready! Just run `python test_model.py` to verify everything works! 🚀
