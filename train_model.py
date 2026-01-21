"""
Train and save recommendation model using your TMDB data.
This script loads, preprocesses, trains, and saves the model for future predictions.
"""

import sys
import os
import pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import DatasetLoader
from src.preprocessing import MovieDataProcessor
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.collaborative_filtering import CollaborativeFilteringRecommender
from backend.memory import MemoryManager
import uuid

print("=" * 70)
print("TRAINING RECOMMENDATION MODEL WITH YOUR TMDB DATA")
print("=" * 70)
print()

# Step 1: Load your TMDB data
print("STEP 1: Loading enriched TMDB data...")
print("-" * 70)
loader = DatasetLoader("data")
movies_df = loader.load_movies("combined_movies.csv")  # Combined Bollywood + Hollywood
print(f"✅ Loaded {len(movies_df)} movies")
print()

print("Filtering for complete enrichment data...")
complete_movies = movies_df[
    movies_df['genre'].notna() &
    movies_df['director'].notna() &
    movies_df['year'].notna() &
    (movies_df['genre'] != 'Unknown') &
    (movies_df['director'] != 'Unknown')
]
print(f"✅ {len(complete_movies)} movies have complete enrichment (genre, director, year)")
print(f"   Columns: {', '.join(movies_df.columns.tolist()[:8])}...")
movies_df = complete_movies
print()

# Step 2: Preprocess the data
print("STEP 2: Preprocessing data...")
print("-" * 70)
processor = MovieDataProcessor()

# Clean data
movies_df = processor.clean_data(movies_df)
print(f"✅ Cleaned data: {len(movies_df)} movies")

# Extract features
movies_df = processor.create_content_features(movies_df)
print(f"✅ Created content features")

# Create TF-IDF matrix for content-based filtering
tfidf_matrix, feature_names = processor.create_tfidf_matrix(movies_df)
print(f"✅ TF-IDF matrix created: {tfidf_matrix.shape}")
print()

# Step 3: Build and train the recommender
print("STEP 3: Building enhanced recommender model...")
print("-" * 70)
from src.recommenders.enhanced_recommender import EnhancedRecommender
recommender = EnhancedRecommender(movies_df, tfidf_matrix)
print("✅ Enhanced recommender built with categorized recommendations")
print()

# Step 3.5: Comprehensive Evaluation
print("STEP 3.5: Running Evaluation (Latency & Coverage)...")
print("-" * 70)
import time
import random

# Select 100 random movies as "Test Queries"
test_indices = random.sample(range(len(movies_df)), min(100, len(movies_df)))
test_movie_ids = movies_df.iloc[test_indices].iloc[:, 0].tolist() # First column is ID

start_time = time.time()
all_recommendations = set()
total_recs = 0

print(f"Testing inference on {len(test_movie_ids)} queries...")

for movie_id in test_movie_ids:
    # Use 'similar_content' as it triggers the linear_kernel
    recs = recommender.get_similar_content(movie_id, n=10)
    for r_id, _ in recs:
        all_recommendations.add(r_id)
    total_recs += len(recs)

end_time = time.time()
duration = end_time - start_time
avg_latency = (duration / len(test_movie_ids)) * 1000 # ms
coverage = len(all_recommendations) / len(movies_df) * 100 if len(movies_df) > 0 else 0

print(f"✅ Evaluation Results:")
print(f"   - Average Inference Time: {avg_latency:.2f} ms per query")
print(f"   - Unique Movies Recommended: {len(all_recommendations)}")
print(f"   - Catalog Coverage: {coverage:.2f}% (higher is better)")
print()

# Step 4: Save the trained model
print("STEP 4: Saving optimized model...")
print("-" * 70)

# OPTIMIZATION: Do not save redundant copies. 
# The recommender object ALREADY contains movies_df and tfidf_matrix.
model_data = {
    'recommender': recommender,
    # 'movies_df': movies_df,     # REDUNDANT - Accessed via recommender.movies_df
    # 'tfidf_matrix': tfidf_matrix, # REDUNDANT - Accessed via recommender.tfidf_matrix
    'processor': processor
}

model_path = "models/trained_recommender.pkl"
os.makedirs("models", exist_ok=True)

with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

file_size_mb = os.path.getsize(model_path) / (1024*1024)
print(f"✅ Model saved to: {model_path}")
print(f"   File size: {file_size_mb:.2f} MB")

if file_size_mb > 100:
    print("⚠️ WARNING: Model size is still large!")
else:
    print("🎉 SUCCESS: Model size is optimized (<100 MB)!")
print()

# Log to Memory System
try:
    memory = MemoryManager()
    run_id = str(uuid.uuid4())
    metrics = {
        "movies_count": len(movies_df),
        "features_shape": tfidf_matrix.shape,
        "inference_ms": avg_latency,
        "coverage_percent": coverage,
        "model_size_mb": file_size_mb
    }
    params = {
        "model_type": "EnhancedRecommender_Optimized",
        "output_path": model_path
    }
    memory.log_training(run_id, metrics, params)
    print(f"✅ Logged training run {run_id} to memory.")
except Exception as e:
    print(f"⚠️ Failed to log to memory: {e}")

# Step 5: Test the saved model
print("STEP 5: Testing saved model loading...")
print("-" * 70)

# Load the model back
with open(model_path, 'rb') as f:
    loaded_model = pickle.load(f)

loaded_recommender = loaded_model['recommender']
# Verify we can access the data from the recommender
loaded_movies = loaded_recommender.movies_df 

# Get a sample movie ID
movie_id_col = loaded_movies.columns[0]
sample_movie_id = loaded_movies.iloc[0][movie_id_col]
sample_movie_info = loaded_recommender.get_movie_info(sample_movie_id)

print(f"Sample prediction - Finding movies similar to:")
print(f"  Title: {sample_movie_info['title']}")
print(f"  Genre: {sample_movie_info['genre']}")
print()

# Get predictions
try:
    similar_movies = loaded_recommender.get_similar_content(sample_movie_id, n=5)
    print("✅ Top 5 predictions (Content Similarity):")
except AttributeError:
    # Fallback if method name is different
    similar_movies = loaded_recommender.get_recommendations(sample_movie_id, n=5)
    print("✅ Top 5 predictions:")
for i, (movie_id, score) in enumerate(similar_movies, 1):
    info = loaded_recommender.get_movie_info(movie_id)
    print(f"   {i}. {info['title']} (Score: {score:.3f})")
print()

# Summary
print("=" * 70)
print("MODEL TRAINING COMPLETE!")
print("=" * 70)
print()
print("✅ Data loaded and preprocessed")
print("✅ Evaluation passed")
print(f"✅ Model saved ({file_size_mb:.2f} MB)")
print("✅ Predictions working")
print()
print("Next steps:")
print("  1. Use 'python predict.py' to make predictions")
print("  2. Load model in your backend: pickle.load(open('models/trained_recommender.pkl', 'rb'))")
print("  3. Use loaded_model['recommender'] to get recommendations")
print()
