import sys
import os
import pickle
import pandas as pd

# Add src to path for pickle loading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.recommenders.enhanced_recommender import EnhancedRecommender

def main():
    print("Loading model... please wait.")
    model_path = "models/trained_recommender.pkl"
    
    if not os.path.exists(model_path):
        print("Error: Model file not found. Please run 'train_model.py' first.")
        return

    try:
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    recommender = loaded_model['recommender']
    movies_df = loaded_model['movies_df']
    
    # Create a case-insensitive title mapping
    # Map lowercase title to movie_id
    title_map = {}
    print("Indexing movies...")
    for idx, row in movies_df.iterrows():
        # Handle potential non-string titles or NaNs
        if pd.notna(row['title']):
            title = str(row['title']).strip().lower()
            movie_id = row['id']
            title_map[title] = movie_id
        
    print("\n" + "="*60)
    print("🎬 MOVIE RECOMMENDATION SYSTEM INTERACTIVE DEMO")
    print("="*60)
    print(f"Loaded {len(movies_df)} movies.")
    print("Enter a movie name to get recommendations")
    print("Type 'q' or 'exit' to quit")
    
    while True:
        print("\n" + "-"*60)
        try:
            user_input = input("Enter movie name: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
            
        if user_input.lower() in ('q', 'quit', 'exit'):
            print("Goodbye! 👋")
            break
            
        if not user_input:
            continue
            
        movie_id = title_map.get(user_input.lower())
        
        if not movie_id:
            print(f"❌ '{user_input}' is not in our database.")
            # Optional: Simple suggestion for close matches could be added here
            continue
            
        # Get info
        info = recommender.get_movie_info(movie_id)
        # Handle missing key year/genre etc nicely
        year = info.get('year', 'N/A')
        rating = info.get('rating', 'N/A')
        
        print(f"\n✅ Found: {info['title']} ({year})")
        print(f"   Genre: {info.get('genre', 'N/A')}")
        print(f"   Director: {info.get('director', 'N/A')}")
        print(f"   Rating: {rating}/10")
        
        # Get all recommendations
        print(f"\n📊 Generating recommendations...")
        
        recs = recommender.get_categorized_recommendations(movie_id, n_per_category=5)
        
        # 1. Content Similarity
        print(f"\n🔸 Similar Content (Plot & Cast & Features):")
        if recs['similar_content']:
            for mid, score in recs['similar_content']:
                m = recommender.get_movie_info(mid)
                print(f"   - {m['title']} ({m.get('year', 'N/A')})")
        else:
            print("   (No recommendations found)")

        # 2. Similar Genre
        print(f"\n🔸 Similar Genre:")
        if recs['similar_genre']:
            for mid, score in recs['similar_genre']:
                m = recommender.get_movie_info(mid)
                print(f"   - {m['title']} ({m.get('year', 'N/A')})")
        else:
            print("   (No recommendations found)")

        # 3. Same Director
        director_name = info.get('director', 'Unknown')
        print(f"\n🔸 Same Director ({director_name}):")
        if recs['same_director']:
            for mid, score in recs['same_director']:
                m = recommender.get_movie_info(mid)
                print(f"   - {m['title']} ({m.get('year', 'N/A')})")
        else:
            print("   (No other movies by this director in database)")

        # 4. Popular in same year
        year_val = str(year)[:4]
        print(f"\n🔸 Popular in {year_val}:")
        if recs['popular_that_year']:
            for mid, score in recs['popular_that_year']:
                m = recommender.get_movie_info(mid)
                print(f"   - {m['title']}")
        else:
             print("   (No recommendations found)")

if __name__ == "__main__":
    main()
