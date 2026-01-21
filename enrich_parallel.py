"""
Parallel Movie Enrichment Script with Thread Safety
Optimized version using ThreadPoolExecutor for 3-4x speedup
"""

import requests
import pandas as pd
import os
import time
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import logging
from datetime import datetime
from backend.memory import MemoryManager

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"

# Configuration
DEBUG_MODE = False  # Set to True for single-threaded debugging
MAX_WORKERS = 4      # Number of parallel threads
BATCH_SIZE = 100     # Checkpoint every N movies
RATE_LIMIT_DELAY = 0.25  # Delay between API calls per thread

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(threadName)-12s] %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

print(f"DEBUG: API_KEY = {'SET' if API_KEY else 'NOT FOUND'}")
print(f"Configuration: Workers={MAX_WORKERS}, BatchSize={BATCH_SIZE}, Debug={DEBUG_MODE}")

# Thread-safe DataFrame lock


# Create session with connection pooling for each thread
import threading
thread_local = threading.local()

def get_thread_session():
    """Get or create a session for the current thread"""
    if not hasattr(thread_local, 'session'):
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False
        )
        
        # Configure adapter
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=5,
            pool_maxsize=10
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({
            'Connection': 'keep-alive',
            'User-Agent': 'MovieRecommender/1.0'
        })
        
        thread_local.session = session
        logger.debug(f"Created new session for thread")
    
    return thread_local.session


def search_movie(title, year=None, retries=3):
    """Search for movie using thread-local session"""
    session = get_thread_session()
    
    for attempt in range(retries):
        try:
            url = f"{BASE_URL}/search/movie"
            params = {
                "api_key": API_KEY,
                "query": title,
                "year": year
            }
            response = session.get(url, params=params, timeout=(5, 10))
            response.raise_for_status()
            results = response.json().get("results", [])
            return results[0] if results else None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(0.5)
            else:
                logger.exception(f"Error searching '{title}'")
                return None


def get_movie_details(movie_id, retries=3):
    """Get movie details using thread-local session"""
    session = get_thread_session()
    
    for attempt in range(retries):
        try:
            url = f"{BASE_URL}/movie/{movie_id}"
            params = {
                "api_key": API_KEY,
                "append_to_response": "credits"
            }
            response = session.get(url, params=params, timeout=(5, 10))
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(0.5)
            else:
                logger.exception(f"Error getting details for ID {movie_id}")
                return {}


def extract_director(credits):
    """Extract director from credits"""
    for person in credits.get("crew", []):
        if person.get("job") == "Director":
            return person.get("name")
    return None


def extract_cast(credits, max_actors=5):
    """Extract top N actors from credits"""
    cast_list = credits.get("cast", [])
    top_actors = [person.get("name") for person in cast_list[:max_actors] if person.get("name")]
    return ", ".join(top_actors) if top_actors else None


def enrich_single_movie(movie_row):
    """
    Enrich a single movie (thread-safe, pure function)
    
    Args:
        movie_row: Tuple of (index, Series) from DataFrame.iterrows()
        
    Returns:
        Dictionary with enriched data
    """
    idx, row = movie_row
    movie_id = row[row.index[0]]  # First column is ID
    
    try:
        # Rate limiting per thread
        time.sleep(RATE_LIMIT_DELAY)
        
        # Get movie details
        details = get_movie_details(movie_id)
        
        if not details:
            return {'index': idx, 'error': True}
        
        # Extract data
        credits = details.get("credits", {})
        genre_names = [g["name"] for g in details.get("genres", [])]
        director = extract_director(credits)
        cast = extract_cast(credits, max_actors=5)
        year = details.get("release_date", "")[:4]
        
        return {
            'index': idx,
            'genre': ", ".join(genre_names) if genre_names else None,
            'director': director,
            'cast': cast,
            'year': year,
            'error': False
        }
        
    except Exception as e:
        logger.error(f"Error enriching movie {movie_id}: {e}")
        return {'index': idx, 'error': True}


def save_checkpoint(df, filename="data/movies_enriched_checkpoint.csv"):
    """Thread-safe checkpoint saving"""
    import shutil
    from pathlib import Path
    
    temp_file = filename + ".tmp"
    
    try:
        # Write to temp file
        df.to_csv(temp_file, index=False)
        
        # Atomic rename
        shutil.move(temp_file, filename)
        logger.info(f"💾 Checkpoint saved at {filename}")
        
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")
        if Path(temp_file).exists():
            Path(temp_file).unlink()


def enrich_batch(df, batch_indices, worker_id=0):
    """
    Enrich a batch of movies in parallel
    
    Args:
        df: DataFrame
        batch_indices: List of indices to process
        worker_id: Batch number for logging
        
    Returns:
        Dictionary mapping index to enriched data
    """
    results = {}
    movies_to_process = [(idx, df.loc[idx]) for idx in batch_indices]
    
    if DEBUG_MODE:
        # Sequential for debugging
        logger.info(f"Batch {worker_id}: Processing {len(movies_to_process)} movies (DEBUG MODE)")
        for movie_row in movies_to_process:
            result = enrich_single_movie(movie_row)
            results[result['index']] = result
    else:
        # Parallel processing
        logger.info(f"Batch {worker_id}: Processing {len(movies_to_process)} movies with {MAX_WORKERS} workers")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all movies
            future_to_idx = {
                executor.submit(enrich_single_movie, movie_row): movie_row[0]
                for movie_row in movies_to_process
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result(timeout=60)  # 60s timeout per movie
                    results[result['index']] = result
                except TimeoutError:
                    logger.error(f"Timeout processing movie at index {idx}")
                    results[idx] = {'index': idx, 'error': True}
                except Exception as e:
                    logger.error(f"Exception processing movie at index {idx}: {e}")
                    results[idx] = {'index': idx, 'error': True}
    
    return results


def enrich_movies(limit=None, checkpoint_interval=100):
    """Main enrichment function with parallel processing"""
    logger.info("Starting enrichment process...")
    
    # Load data
    df = pd.read_csv("data/tmdb_top_rated_movies.csv")
    logger.info(f"Loaded {len(df)} movies from CSV")
    
    # Check for existing enriched file
    enriched_file = "data/movies_enriched.csv"
    if os.path.exists(enriched_file):
        logger.info("Found existing enriched file, loading...")
        df_existing = pd.read_csv(enriched_file)
        
        # Determine which columns to merge
        merge_cols = ['id', 'genre', 'director', 'year']
        if 'cast' in df_existing.columns:
            merge_cols.append('cast')
        
        # Merge existing data
        df = df.merge(df_existing[merge_cols], on='id', how='left', suffixes=('', '_existing'))
        
        # Use existing values
        if 'genre_existing' in df.columns:
            df['genre'] = df['genre_existing']
            df['director'] = df['director_existing']
            df['year'] = df['year_existing']
            
            if 'cast_existing' in df.columns:
                df['cast'] = df['cast_existing']
                df = df.drop(columns=['genre_existing', 'director_existing', 'year_existing', 'cast_existing'])
            else:
                df['cast'] = None
                df = df.drop(columns=['genre_existing', 'director_existing', 'year_existing'])
        else:
            # No existing enrichment columns, initialize
            if 'cast' not in df.columns:
                df['cast'] = None
        
        # Count movies needing enrichment
        needs_enrichment = (df['genre'].isna() | df['cast'].isna()).sum()
        logger.info(f"{needs_enrichment} movies need enrichment (out of {len(df)} total)")
    else:
        # Initialize columns
        df['genre'] = None
        df['director'] = None
        df['year'] = None
        df['cast'] = None
        logger.info(f"Processing ALL {len(df)} movies (first run)")
    
    # Limit for testing
    if limit:
        df = df.head(limit)
        logger.info(f"Limited to {limit} movies for testing")
    
    # Find movies needing enrichment
    needs_enrichment_mask = df['genre'].isna() | df['cast'].isna()
    indices_to_enrich = df[needs_enrichment_mask].index.tolist()
    
    if not indices_to_enrich:
        logger.info("✅ All movies already enriched!")
        return df
    
    logger.info(f"Will enrich {len(indices_to_enrich)} movies in batches of {checkpoint_interval}")
    
    # Process in batches
    total_enriched = 0
    for batch_num, start in enumerate(range(0, len(indices_to_enrich), checkpoint_interval)):
        end = min(start + checkpoint_interval, len(indices_to_enrich))
        batch_indices = indices_to_enrich[start:end]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"BATCH {batch_num + 1}: Movies {start + 1}-{end} of {len(indices_to_enrich)}")
        logger.info(f"{'='*60}")
        
        # Enrich this batch
        batch_start_time = time.time()
        results = enrich_batch(df, batch_indices, batch_num + 1)
        batch_duration = time.time() - batch_start_time
        
        # Apply results (single-threaded, thread-safe)
        success_count = 0
        error_count = 0
        
        for idx, data in results.items():
            if not data.get('error', False):
                df.at[idx, 'genre'] = data.get('genre')
                df.at[idx, 'director'] = data.get('director')
                df.at[idx, 'cast'] = data.get('cast')
                df.at[idx, 'year'] = data.get('year')
                success_count += 1
            else:
                error_count += 1
        
        total_enriched += success_count
        
        # Save checkpoint
        save_checkpoint(df, enriched_file)
        
        # Progress report
        avg_time = batch_duration / len(batch_indices)
        remaining = len(indices_to_enrich) - end
        eta_minutes = (remaining * avg_time) / 60
        
        logger.info(f"Batch completed: {success_count} success, {error_count} errors")
        logger.info(f"Batch time: {batch_duration:.1f}s ({avg_time:.2f}s per movie)")
        logger.info(f"Progress: {total_enriched}/{len(indices_to_enrich)} enriched")
        logger.info(f"ETA: ~{eta_minutes:.1f} minutes remaining")
    
    # Final save
    # Final save
    if total_enriched % checkpoint_interval != 0:
        df.to_csv(enriched_file, index=False)
        
    logger.info("\n" + "="*60)
    logger.info("✅ ENRICHMENT COMPLETE!")
    logger.info(f"Total enriched: {total_enriched} movies")
    logger.info(f"Output: {enriched_file}")
    logger.info("="*60)
    
    # Log to Memory
    try:
        memory = MemoryManager()
        memory.log_external_api(
            trace_id=None,
            api_name="TMDB Enrichment",
            status="Complete",
            latency=time.time() - start_time
        )
    except Exception as e:
        logger.warning(f"Failed to log to memory: {e}")

    return df


if __name__ == "__main__":
    start_time = time.time()
    
    try:
        enrich_movies()
    except KeyboardInterrupt:
        logger.info("\n\n⚠️ Process interrupted by user")
    except Exception as e:
        logger.error(f"\n\n❌ Fatal error: {e}", exc_info=True)
    finally:
        elapsed = time.time() - start_time
        logger.info(f"\nTotal runtime: {elapsed/60:.1f} minutes")
