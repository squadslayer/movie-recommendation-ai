"""
Optimized async movie fetcher for TMDB
Fetches ~90K quality movies to expand dataset to 100K total
Uses asyncio for maximum throughput within API rate limits
"""

import asyncio
import aiohttp
import pandas as pd
import os
import json
from dotenv import load_dotenv
from datetime import datetime
import logging

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"

# Configuration
MAX_WORKERS = 8
TARGET_NEW_MOVIES = 90000
MIN_VOTE_COUNT = 5  # Quality filter: at least 5 votes
MIN_VOTE_AVERAGE = 3.0  # Quality filter: at least 3.0/10 rating
CHECKPOINT_INTERVAL = 1000
OUTPUT_FILE = "data/movies_enriched.csv"
STATE_FILE = "data/fetch_state.json"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class AsyncMovieFetcher:
    def __init__(self):
        self.session = None
        self.semaphore = asyncio.Semaphore(MAX_WORKERS)
        self.rate_limiter = asyncio.Semaphore(40)  # TMDB: 40 req/10s
        self.existing_ids = set()
        self.new_movies = []
        self.last_page = 0
        
    async def init_session(self):
        """Initialize aiohttp session with connection pooling"""
        connector = aiohttp.TCPConnector(limit=MAX_WORKERS, limit_per_host=MAX_WORKERS)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        
    async def close_session(self):
        if self.session:
            await self.session.close()
            
    async def rate_limit_sleep(self):
        """Enforce TMDB rate limit: 40 requests per 10 seconds"""
        await asyncio.sleep(0.25)  # 4 requests per second = safe margin
        
    async def fetch_json(self, url, params=None):
        """Fetch JSON with rate limiting"""
        async with self.rate_limiter:
            await self.rate_limit_sleep()
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                return None
                
    async def discover_movies(self, page=1):
        """Discover movies with quality filters"""
        params = {
            'api_key': API_KEY,
            'page': page,
            'sort_by': 'popularity.desc',
            'vote_count.gte': MIN_VOTE_COUNT,
            'vote_average.gte': MIN_VOTE_AVERAGE,
            'include_adult': 'false',
        }
        
        data = await self.fetch_json(f"{BASE_URL}/discover/movie", params)
        return data.get('results', []) if data else []
        
    async def get_movie_credits(self, movie_id):
        """Get cast and crew for a movie"""
        params = {'api_key': API_KEY}
        data = await self.fetch_json(f"{BASE_URL}/movie/{movie_id}/credits", params)
        
        if not data:
            return None, None
            
        # Extract director
        crew = data.get('crew', [])
        director = next((c['name'] for c in crew if c.get('job') == 'Director'), 'Unknown')
        
        # Extract top 5 cast
        cast = data.get('cast', [])[:5]
        cast_names = ', '.join([c['name'] for c in cast]) if cast else 'Unknown'
        
        return director, cast_names
        
    async def enrich_movie(self, movie):
        """Enrich a single movie with full details"""
        async with self.semaphore:
            movie_id = movie['id']
            
            # Skip if already in dataset
            if movie_id in self.existing_ids:
                return None
                
            try:
                # Get credits (director + cast)
                director, cast = await self.get_movie_credits(movie_id)
                
                if not director:
                    return None
                    
                # Extract year from release_date
                year = movie.get('release_date', '')[:4] if movie.get('release_date') else 'Unknown'
                
                # Get genres
                genre_ids = movie.get('genre_ids', [])
                genres = self.map_genres(genre_ids)
                
                enriched = {
                    'id': movie_id,
                    'original_language': movie.get('original_language', 'en'),
                    'overview': movie.get('overview', ''),
                    'release_date': movie.get('release_date', ''),
                    'title': movie.get('title', 'Unknown'),
                    'popularity': movie.get('popularity', 0),
                    'vote_average': movie.get('vote_average', 0),
                    'vote_count': movie.get('vote_count', 0),
                    'genre': genres,
                    'director': director,
                    'year': year,
                    'cast': cast,
                }
                
                return enriched
                
            except Exception as e:
                logger.error(f"Error enriching movie {movie_id}: {e}")
                return None
                
    def map_genres(self, genre_ids):
        """Map genre IDs to names"""
        genre_map = {
            28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy',
            80: 'Crime', 99: 'Documentary', 18: 'Drama', 10751: 'Family',
            14: 'Fantasy', 36: 'History', 27: 'Horror', 10402: 'Music',
            9648: 'Mystery', 10749: 'Romance', 878: 'Science Fiction',
            10770: 'TV Movie', 53: 'Thriller', 10752: 'War', 37: 'Western'
        }
        return ', '.join([genre_map.get(gid, 'Unknown') for gid in genre_ids[:3]])
        
    def load_existing_ids(self):
        """Load existing movie IDs to avoid duplicates"""
        if os.path.exists(OUTPUT_FILE):
            df = pd.read_csv(OUTPUT_FILE)
            self.existing_ids = set(df['id'].values)
            logger.info(f"Loaded {len(self.existing_ids)} existing movie IDs")
            
    def load_state(self):
        """Load last processed page from state file"""
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r') as f:
                    state = json.load(f)
                    self.last_page = state.get('last_page', 0)
                    logger.info(f"📍 Resuming from page {self.last_page + 1}")
            except:
                self.last_page = 0
                
    def save_state(self, page):
        """Save current page to state file"""
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, 'w') as f:
            json.dump({'last_page': page}, f)
            
    def save_checkpoint(self):
        """Save progress to CSV"""
        if not self.new_movies:
            return
            
        df_new = pd.DataFrame(self.new_movies)
        
        if os.path.exists(OUTPUT_FILE):
            df_existing = pd.read_csv(OUTPUT_FILE)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.drop_duplicates(subset=['id'], inplace=True)
            df_combined.to_csv(OUTPUT_FILE, index=False)
        else:
            df_new.to_csv(OUTPUT_FILE, index=False)
            
        logger.info(f"✅ Checkpoint: Saved {len(self.new_movies)} new movies")
        self.new_movies = []
        
    async def run(self):
        """Main execution"""
        await self.init_session()
        self.load_existing_ids()
        self.load_state()
        
        logger.info(f"🎬 Starting fetch: Target = {TARGET_NEW_MOVIES} new movies")
        logger.info(f"Quality filters: vote_count >= {MIN_VOTE_COUNT}, vote_average >= {MIN_VOTE_AVERAGE}")
        
        total_enriched = 0
        page = self.last_page + 1
        
        try:
            while total_enriched < TARGET_NEW_MOVIES:
                # Discover movies
                movies = await self.discover_movies(page)
                
                if not movies:
                    logger.warning(f"No more movies found at page {page}")
                    break
                    
                logger.info(f"Page {page}: Discovered {len(movies)} movies")
                
                # Enrich in parallel
                tasks = [self.enrich_movie(movie) for movie in movies]
                results = await asyncio.gather(*tasks)
                
                # Filter and add valid movies
                valid = [r for r in results if r is not None]
                self.new_movies.extend(valid)
                total_enriched += len(valid)
                
                logger.info(f"Progress: {total_enriched}/{TARGET_NEW_MOVIES} enriched")
                
                # Checkpoint
                if len(self.new_movies) >= CHECKPOINT_INTERVAL:
                    self.save_checkpoint()
                    self.save_state(page)
                    
                page += 1
                
        except KeyboardInterrupt:
            logger.info("⚠️ Interrupted by user")
        finally:
            # Final save
            self.save_checkpoint()
            self.save_state(page)
            await self.close_session()
            
        logger.info(f"🎉 Complete! Total new movies: {total_enriched}")

if __name__ == "__main__":
    print("🚀 Movie Fetcher Starting...")
    print(f"API Key: {'SET' if API_KEY else 'MISSING'}")
    fetcher = AsyncMovieFetcher()
    asyncio.run(fetcher.run())
