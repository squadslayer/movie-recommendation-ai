"""
Bollywood Movies Enrichment Script  
Enriches the Bollywood movies CSV with missing data from TMDB API

Missing columns to fetch:
- genre
- director  
- overview
- cast (top 5 actors)

Uses the existing parallel enrichment infrastructure from enrich_parallel.py
"""

import requests
import pandas as pd
import os
import time
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import threading

# Load environment variables
load_dotenv()
TMDB_API_KEY = os.getenv('TMDB_API_KEY')

if not TMDB_API_KEY:
    raise ValueError("❌ TMDB_API_KEY not found in .env file")

# Configuration
MAX_WORKERS = 4  # Number of parallel threads
BATCH_SIZE = 100  # Checkpoint interval
DEBUG_MODE = False

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [Worker-%(threadName)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enrichment_bollywood.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Thread-safe lock for checkpoint saving
checkpoint_lock = threading.Lock()

# Thread-local session
thread_local = threading.local()

def get_thread_session():
    """Get or create a session for the current thread"""
    if not hasattr(thread_local, 'session'):
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        thread_local.session = session
    return thread_local.session

def get_movie_details(movie_id, retries=3):
    """Get movie details and credits from TMDB"""
    session = get_thread_session()
    
    for attempt in range(retries):
        try:
            # Get movie details
            details_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
            params = {'api_key': TMDB_API_KEY}
            
            details_response = session.get(details_url, params=params, timeout=5)
            
            if details_response.status_code == 404:
                logger.warning(f"Movie ID {movie_id} not found in TMDB")
                return None
                
            if details_response.status_code != 200:
                logger.warning(f"Failed to fetch movie {movie_id}: {details_response.status_code}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None
            
            details = details_response.json()
            
            # Get credits
            credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits"
            credits_response = session.get(credits_url, params=params, timeout=5)
            
            if credits_response.status_code == 200:
                credits = credits_response.json()
            else:
                credits = {'crew': [], 'cast': []}
            
            # Rate limiting
            time.sleep(0.25)  # 4 requests per second
            
            return {'details': details, 'credits': credits}
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error for movie {movie_id} (attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
    
    return None

def extract_director(credits):
    """Extract director from credits"""
    for crew_member in credits.get('crew', []):
        if crew_member.get('job') == 'Director':
            return crew_member.get('name', 'Unknown')
    return 'Unknown'

def extract_cast(credits, max_actors=5):
    """Extract top N actors from credits"""
    cast_list = credits.get('cast', [])[:max_actors]
    actors = [actor.get('name') for actor in cast_list if actor.get('name')]
    return ', '.join(actors) if actors else 'Unknown'

def extract_genres(details):
    """Extract genres from movie details"""
    genres = details.get('genres', [])
    genre_names = [g.get('name') for g in genres if g.get('name')]
    return ', '.join(genre_names) if genre_names else 'Unknown'

def enrich_single_movie(movie_row):
    """
    Enrich a single movie (pure function, thread-safe)
    
    Args:
        movie_row: Tuple of (index, Series) from DataFrame.iterrows()
        
    Returns:
        Dictionary with enriched data
    """
    index, row = movie_row
    movie_id = str(row['id'])
    title = row['title']
    
    try:
        # Fetch from TMDB
        data = get_movie_details(movie_id)
        
        if data is None:
            logger.warning(f"[{index}] Failed to enrich: {title}")
            return {
                'index': index,
               'genre': 'Unknown',
                'director': 'Unknown',
                'overview': 'Unknown',
                'cast': 'Unknown'
            }
        
        details = data['details']
        credits = data['credits']
        
        # Extract fields
        genre = extract_genres(details)
        director = extract_director(credits)
        overview = details.get('overview', 'Unknown')
        cast = extract_cast(credits)
        
        if DEBUG_MODE:
            logger.debug(f"[{index}] Enriched: {title} -> Director: {director}, Cast: {cast[:50]}")
        
        return {
            'index': index,
            'genre': genre,
            'director': director,
            'overview': overview,
            'cast': cast
        }
        
    except Exception as e:
        logger.error(f"[{index}] Error enriching {title}: {e}")
        return {
            'index': index,
            'genre': 'Unknown',
            'director': 'Unknown',
            'overview': 'Unknown',
            'cast': 'Unknown'
        }

def save_checkpoint(df, filename="data/bollywood_movies_enriched_checkpoint.csv"):
    """Thread-safe checkpoint saving"""
    with checkpoint_lock:
        try:
            df.to_csv(filename, index=False)
            logger.info(f"💾 Checkpoint saved: {filename}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

def enrich_movies(input_file, output_file, checkpoint_interval=100):
    """
    Main enrichment function
    
    Args:
        input_file: Path to input CSV
        output_file: Path to output enriched CSV
        checkpoint_interval: Save checkpoint every N movies
    """
    logger.info("="*70)
    logger.info("BOLLYWOOD MOVIES ENRICHMENT")
    logger.info("="*70)
    
    # Load data
    logger.info(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    original_count = len(df)
    logger.info(f"✅ Loaded {original_count} movies")
    
    # Add missing columns if they don't exist
    for col in ['genre', 'director', 'overview', 'cast']:
        if col not in df.columns:
            df[col] = 'Unknown'
    
    # Find movies that need enrichment
    needs_enrichment = df[
        (df['genre'] == 'Unknown') | 
        (df['director'] == 'Unknown') |
        (df['overview'] == 'Unknown') |
        (df['cast'] == 'Unknown')
    ]
    
    total_to_enrich = len(needs_enrichment)
    logger.info(f"📊 Movies needing enrichment: {total_to_enrich}/{original_count}")
    
    if total_to_enrich == 0:
        logger.info("✅ All movies already enriched!")
        return
    
    # Prepare for parallel enrichment
    movie_rows = list(needs_enrichment.iterrows())
    processed_count = 0
    start_time = time.time()
    
    logger.info(f"🚀 Starting enrichment with {MAX_WORKERS} workers...")
    
    # Process in batches for checkpointing
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        batch_start = 0
        
        while batch_start < len(movie_rows):
            batch_end = min(batch_start + checkpoint_interval, len(movie_rows))
            batch_rows = movie_rows[batch_start:batch_end]
            
            # Submit batch futures
            future_to_row = {executor.submit(enrich_single_movie, row): row for row in batch_rows}
            
            # Collect results
            batch_results = []
            for future in as_completed(future_to_row):
                try:
                    result = future.result()
                    batch_results.append(result)
                    processed_count += 1
                    
                    if processed_count % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = processed_count / elapsed if elapsed > 0 else 0
                        eta = (total_to_enrich - processed_count) / rate if rate > 0 else 0
                        logger.info(f"Progress: {processed_count}/{total_to_enrich} ({processed_count/total_to_enrich*100:.1f}%) | Rate: {rate:.2f} movies/s | ETA: {eta/60:.1f} min")
                
                except Exception as e:
                    logger.error(f"Future failed: {e}")
            
            # Update DataFrame with batch results
            for result in batch_results:
                idx = result['index']
                df.at[idx, 'genre'] = result['genre']
                df.at[idx, 'director'] = result['director']
                df.at[idx, 'overview'] = result['overview']
                df.at[idx, 'cast'] = result['cast']
            
            # Save checkpoint
            if batch_end < len(movie_rows):
                save_checkpoint(df)
            
            batch_start = batch_end
    
    # Save final result
    logger.info(f"💾 Saving final enriched data to: {output_file}")
    df.to_csv(output_file, index=False)
    
    # Summary
    elapsed = time.time() - start_time
    logger.info("="*70)
    logger.info("ENRICHMENT COMPLETE")
    logger.info("="*70)
    logger.info(f"✅ Total movies enriched: {processed_count}")
    logger.info(f"⏱️  Total time: {elapsed/60:.1f} minutes")
    logger.info(f"⚡ Average rate: {processed_count/elapsed:.2f} movies/second")
    logger.info(f"📁 Output saved to: {output_file}")
    logger.info("="*70)

if __name__ == "__main__":
    INPUT_FILE = "data/Bollywood Movies Dataset export 2026-01-21 07-39-18.csv"
    OUTPUT_FILE = "data/bollywood_movies_enriched.csv"
    
    start_time = time.time()
    
    try:
        enrich_movies(INPUT_FILE, OUTPUT_FILE)
    except KeyboardInterrupt:
        logger.info("\n\n⚠️ Process interrupted by user")
    except Exception as e:
        logger.error(f"\n\n❌ Fatal error: {e}", exc_info=True)
    finally:
        elapsed = time.time() - start_time
        logger.info(f"\nTotal runtime: {elapsed/60:.1f} minutes")
