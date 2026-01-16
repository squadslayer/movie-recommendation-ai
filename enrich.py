import requests
import pandas as pd
import os
import time
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")
print("DEBUG: API_KEY =", API_KEY[:10] + "..." if API_KEY else "None")
BASE_URL = "https://api.themoviedb.org/3"


def search_movie(title, year=None, retries=3):
    for attempt in range(retries):
        try:
            url = f"{BASE_URL}/search/movie"
            params = {
                "api_key": API_KEY,
                "query": title,
                "year": year
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            results = response.json().get("results", [])
            return results[0] if results else None
        except Exception as e:
            if attempt < retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"  Retry {attempt + 1}/{retries} for '{title}' after {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"ERROR searching movie '{title}': {e}")
                return None


def get_movie_details(movie_id, retries=3):
    for attempt in range(retries):
        try:
            url = f"{BASE_URL}/movie/{movie_id}"
            params = {
                "api_key": API_KEY,
                "append_to_response": "credits"
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                print(f"  Retry {attempt + 1}/{retries} for movie ID {movie_id} after {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"ERROR getting details for movie ID {movie_id}: {e}")
                return {}


def extract_director(credits):
    for person in credits.get("crew", []):
        if person.get("job") == "Director":
            return person.get("name")
    return None


def save_checkpoint(df, filename="data/movies_enriched_checkpoint.csv"):
    """Save progress periodically"""
    df.to_csv(filename, index=False)


def enrich_movies(limit=None, checkpoint_interval=100):
    print("DEBUG: enrich_movies() started")
    df = pd.read_csv("data/tmdb_top_rated_movies.csv")
    print(f"DEBUG: CSV loaded, total rows = {len(df)}")
    
    # Check if enriched file already exists (for incremental updates)
    enriched_file = "data/movies_enriched.csv"
    if os.path.exists(enriched_file):
        print(f"Found existing enriched file. Loading previous data...")
        df_existing = pd.read_csv(enriched_file)
        # Merge to get existing enrichment data
        df = df.merge(df_existing[['id', 'genre', 'director', 'year']], on='id', how='left', suffixes=('', '_existing'))
        # Use existing data where available
        if 'genre_existing' in df.columns:
            df['genre'] = df['genre_existing']
            df['director'] = df['director_existing']
            df['year'] = df['year_existing']
            df = df.drop(columns=['genre_existing', 'director_existing', 'year_existing'])
        # Count how many need enrichment
        needs_enrichment = df['genre'].isna().sum()
        print(f"Found {needs_enrichment} movies that need enrichment (out of {len(df)} total)")
    else:
        # Initialize columns for first run
        df['genre'] = None
        df['director'] = None
        df['year'] = None
    
    # Check if checkpoint exists
    checkpoint_file = "data/movies_enriched_checkpoint.csv"
    start_idx = 0
    
    # Limit to first N movies for testing (None = process all)
    if limit:
        print(f"DEBUG: Processing first {limit} movies...")
        df = df.head(limit)
    else:
        print(f"DEBUG: Processing ALL {len(df)} movies...")

    for i, row in df.iterrows():
        # Skip movies that already have enrichment data
        if pd.notna(row.get('genre')) and pd.notna(row.get('director')):
            continue
            
        if i % 10 == 0:
            processed = i + 1
            total = len(df)
            needs_processing = df['genre'].isna().sum()
            print(f"Progress: {processed}/{total} checked ({needs_processing} still need enrichment)...")
        
        title = row["title"]
        release_date = row.get("release_date")

        year = None
        if pd.notna(release_date):
            year = str(release_date)[:4]

        movie = search_movie(title, year)

        if not movie:
            # Keep year from release_date even if API search fails
            df.at[i, 'year'] = year
            continue

        details = get_movie_details(movie["id"])

        genre_names = [g["name"] for g in details.get("genres", [])]
        director = extract_director(details.get("credits", {}))
        
        # Update the dataframe row with new data
        df.at[i, 'genre'] = ", ".join(genre_names)
        df.at[i, 'director'] = director
        df.at[i, 'year'] = details.get("release_date", "")[:4]
        
        # Small delay to avoid rate limiting
        time.sleep(0.25)
        
        # Save checkpoint every N movies
        if (i + 1) % checkpoint_interval == 0:
            save_checkpoint(df)
            print(f"  💾 Checkpoint saved at {i + 1} movies")

    print("DEBUG: Writing output CSV")
    df.to_csv("data/movies_enriched.csv", index=False)
    
    # Show statistics
    enriched_count = df['genre'].notna().sum()
    total_count = len(df)
    print(f"✅ movies_enriched.csv created successfully")
    print(f"📊 Stats: {enriched_count}/{total_count} movies have enrichment data ({enriched_count/total_count*100:.1f}%)")
    
    # Clean up checkpoint file
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("Checkpoint file removed")


if __name__ == "__main__":
    enrich_movies()
