import os
import pickle
import sys
import pandas as pd
from typing import Dict, List, Any
from functools import lru_cache
import requests

# Add parent directory to path to import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommenders.enhanced_recommender import EnhancedRecommender

class InferenceController:
    """
    Manages the lifecycle of the ML model and handles inference requests.
    """
    def __init__(self, model_path: str = "models/trained_recommender.pkl"):
        self.model_path = model_path
        self.recommender = None
        self.model_version = "v1.0" # Placeholder versioning
        self.poster_cache = {}
        
    def load_model(self):
        """Load the trained model from disk."""
        print(f"Loading recommendation model from {self.model_path}...")
        if not os.path.exists(self.model_path):
            print("❌ Model file not found.")
            return False
            
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            self.recommender: EnhancedRecommender = model_data['recommender']
            
            size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
            print(f"✅ Model loaded successfully ({size_mb:.2f} MB)")
            if size_mb < 100:
                print("🚀 Optimized Model Detected (High Performance Mode)")
            
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    @lru_cache(maxsize=1000)
    def _get_poster_url(self, movie_id: str) -> str:
        """Fetch poster URL from TMDB with caching."""
        api_key = os.getenv('TMDB_API_KEY')
        if not api_key:
            return None
            
        try:
            url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}"
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                data = response.json()
                poster_path = data.get('poster_path')
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w200{poster_path}"
        except Exception:
            pass
        return None

    def predict(self, input_movies: List[str], limit: int = 10) -> Dict[str, Any]:
        """
        Run inference to generate recommendations.
        
        Args:
            input_movies: List of movie IDs or titles.
            limit: Number of recommendations per category.
            
        Returns:
            Dictionary containing categorized recommendations and metadata.
        """
        if not self.recommender:
            raise RuntimeError("Model not loaded")

        # For this demo, we assume single movie input for categorized recommendations
        # In a real scenario, we'd handle multiple inputs more gracefully
        movie_input = input_movies[0]
        
        # Try to resolve title to ID if needed
        movie_id = movie_input
        if movie_input not in self.recommender.movie_to_idx:
            # Simple title lookup (could be improved)
            found = False
            for mid, idx in self.recommender.movie_to_idx.items():
                title = self.recommender.movies_df.iloc[idx]['title']
                if str(title).lower() == str(movie_input).lower():
                    movie_id = mid
                    found = True
                    break
            if not found:
                return {"error": f"Movie '{movie_input}' not found", "recommendations": {}}

        # Get categorized recommendations
        raw_recs = self.recommender.get_categorized_recommendations(movie_id, n_per_category=limit)
        
        formatted_recs = {}
        for category, movies in raw_recs.items():
            formatted_recs[category] = []
            for rec_mid, score in movies:
                info = self.recommender.get_movie_info(rec_mid)
                if info:
                    # Enrich with poster
                    poster = self._get_poster_url(rec_mid)
                    
                    formatted_recs[category].append({
                        "id": str(rec_mid),
                        "title": info['title'],
                        "year": str(info.get('year', 'N/A')),
                        "rating": float(info.get('rating', 0) if pd.notna(info.get('rating')) else 0),
                        "genre": info.get('genre', 'N/A'),
                        "director": info.get('director', 'N/A'),
                        "score": float(score),
                        "poster_url": poster,
                        "overview": info.get('overview', '')[:200]
                    })
        
        # Explainability
        explanations = [
            f"Recommendations based on '{movie_id}'",
            f"Found {len(formatted_recs.get('similar_content', []))} content matches",
            f"Found {len(formatted_recs.get('same_director', []))} director matches"
        ]

        return {
            "model_version": self.model_version,
            "recommendations": formatted_recs,
            "explanations": explanations
        }

    def get_actor_movies(self, actor_name: str, limit: int = 20):
        """Get movies by actor."""
        if not self.recommender:
            return []
        return self.recommender.get_movies_by_actor(actor_name, n=limit)

    def get_actor_info(self, actor_name: str):
        """Get actor profile."""
        if not self.recommender:
            return None
        return self.recommender.get_actor_info(actor_name)
