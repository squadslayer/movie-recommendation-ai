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

print("=" * 70)
print("TRAINING RECOMMENDATION MODEL WITH YOUR TMDB DATA")
print("=" * 70)
print()

# Step 1: Load your TMDB data
print("STEP 1: Loading enriched TMDB data...")
print("-" * 70)
loader = DatasetLoader("data")
movies_df = loader.load_movies("movies_enriched.csv")
print(f"✅ Loaded {len(movies_df)} movies")

# Filter for movies with complete enrichment data
print("\nFiltering for complete enrichment data...")
complete_data = movies_df.dropna(subset=['genre', 'director', 'year'])
print(f"✅ {len(complete_data)} movies have complete enrichment (genre, director, year)")
print(f"   Columns: {', '.join(complete_data.columns.tolist()[:8])}...")
movies_df = complete_data
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

# Step 4: Save the trained model
print("STEP 4: Saving trained model...")
print("-" * 70)
model_data = {
    'recommender': recommender,
    'movies_df': movies_df,
    'tfidf_matrix': tfidf_matrix,
    'processor': processor
}

model_path = "models/trained_recommender.pkl"
os.makedirs("models", exist_ok=True)

with open(model_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"✅ Model saved to: {model_path}")
print(f"   File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
print()

# Step 5: Test the saved model
print("STEP 5: Testing saved model...")
print("-" * 70)

# Load the model back
with open(model_path, 'rb') as f:
    loaded_model = pickle.load(f)

loaded_recommender = loaded_model['recommender']
loaded_movies = loaded_model['movies_df']

# Get a sample movie ID
movie_id_col = loaded_movies.columns[0]
sample_movie_id = loaded_movies.iloc[0][movie_id_col]
sample_movie_info = loaded_recommender.get_movie_info(sample_movie_id)

print(f"Sample prediction - Finding movies similar to:")
print(f"  Title: {sample_movie_info['title']}")
print(f"  Genre: {sample_movie_info['genre']}")
print()

# Get predictions
similar_movies = loaded_recommender.get_similar_movies(sample_movie_id, n=5)

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
print("✅ Model trained")
print(f"✅ Model saved to: {model_path}")
print("✅ Predictions working")
print()
print("Next steps:")
print("  1. Use 'python predict.py' to make predictions")
print("  2. Load model in your backend: pickle.load(open('models/trained_recommender.pkl', 'rb'))")
print("  3. Use loaded_model['recommender'] to get recommendations")
print()
