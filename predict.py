"""
Make predictions using the enhanced recommendation model.
Provides categorized recommendations: similar genre, same director, popular that year, and similar content.
"""

import sys
import os
import pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_model():
    """Load the trained model."""
    model_path = "models/trained_recommender.pkl"
    
    if not os.path.exists(model_path):
        print("❌ Model not found! Please run 'python train_model.py' first.")
        return None
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data

def print_recommendations(movie_title, n=10):
    """
    Get and display categorized recommendations for a movie.
    
    Args:
        movie_title: Name of the movie
        n: Number of recommendations per category (max 10)
    """
    model_data = load_model()
    if not model_data:
        return
    
    recommender = model_data['recommender']
    movies_df = model_data['movies_df']
    
    # Find movie by title
    matches = movies_df[movies_df['title'].str.contains(movie_title, case=False, na=False)]
    
    if matches.empty:
        print(f"❌ Movie '{movie_title}' not found in database")
        print("\nTry one of these popular titles:")
        print(movies_df.head(5)['title'].tolist())
        return
    
    # Get first match
    movie_id_col = movies_df.columns[0]
    movie_id = matches.iloc[0][movie_id_col]
    movie_info = recommender.get_movie_info(movie_id)
    
    # Print header
    print("=" * 75)
    print(f"RECOMMENDATIONS FOR: {movie_info['title']} ({movie_info['year']})")
    print(f"Genre: {movie_info['genre']} | Director: {movie_info['director']} | Rating: {movie_info['rating']}")
    print("=" * 75)
    print()
    
    # Get categorized recommendations
    recs = recommender.get_categorized_recommendations(movie_id, n_per_category=n)
    
    # 1. Similar Genre
    if recs['similar_genre']:
        print("📽️  MOVIES FROM SIMILAR GENRE")
        print("─" * 75)
        for i, (rec_id, rating) in enumerate(recs['similar_genre'], 1):
            info = recommender.get_movie_info(rec_id)
            print(f"{i}. {info['title']} ({info['year']}) - Rating: {rating:.1f}")
            print(f"   {info['genre']} | Dir: {info['director']}")
        print()
    
    # 2. Same Director
    if recs['same_director']:
        print(f"🎬 MOVIES BY SAME DIRECTOR ({movie_info['director']})")
        print("─" * 75)
        for i, (rec_id, rating) in enumerate(recs['same_director'], 1):
            info = recommender.get_movie_info(rec_id)
            print(f"{i}. {info['title']} ({info['year']}) - Rating: {rating:.1f}")
            print(f"   {info['genre']}")
        print()
    
    # 3. Popular That Year
    if recs['popular_that_year']:
        print(f"📅 POPULAR MOVIES FROM {movie_info['year']}")
        print("─" * 75)
        for i, (rec_id, rating) in enumerate(recs['popular_that_year'], 1):
            info = recommender.get_movie_info(rec_id)
            print(f"{i}. {info['title']} - Rating: {rating:.1f}")
            print(f"   {info['genre']} | Dir: {info['director']}")
        print()
    
    # 4. Similar Content
    if recs['similar_content']:
        print("💫 SIMILAR CONTENT (Based on Plot)")
        print("─" * 75)
        for i, (rec_id, similarity) in enumerate(recs['similar_content'], 1):
            info = recommender.get_movie_info(rec_id)
            print(f"{i}. {info['title']} ({info['year']}) - Similarity: {similarity:.3f}")
            print(f"   {info['genre']} | Dir: {info['director']}")
        print()
    
    print("=" * 75)

if __name__ == "__main__":
    # Example usage
    print("\n" + "=" * 75)
    print("ENHANCED MOVIE RECOMMENDATION SYSTEM")
    print("=" * 75 + "\n")
    
    # Example 1: The Shawshank Redemption
    print_recommendations("Shawshank", n=10)
    
    print("\n" + "=" * 75)
    print("USAGE:")
    print("  python predict.py                           # Run examples")
    print("  from predict import print_recommendations")
    print("  print_recommendations('Inception', n=5)     # Custom search")
    print("=" * 75)
