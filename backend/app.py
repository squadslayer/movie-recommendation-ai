"""
Backend API endpoints for actor-based recommendations
Integrates with Wikipedia for actor images
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import pickle
import sys
import os
from dotenv import load_dotenv
import requests
from functools import lru_cache

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.wikipedia_images import get_actor_image_simple, get_actor_info_simple

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Load environment variables
load_dotenv()
TMDB_API_KEY = os.getenv('TMDB_API_KEY')

# Load model at startup
print("Loading recommendation model...")
try:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'trained_recommender.pkl')
    print(f"Looking for model at: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    recommender = model_data['recommender']
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    recommender = None

# Simple in-memory cache for posters to avoid hitting rate limits
@lru_cache(maxsize=1000)
def get_movie_poster(movie_id):
    """Fetch movie poster URL from TMDB"""
        
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w200{poster_path}"
    except Exception as e:
        print(f"Error fetching poster for {movie_id}: {e}")
        
    return None

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': recommender is not None
    })


@app.route('/api/actors/<actor_name>/movies', methods=['GET'])
def get_actor_movies(actor_name):
    """
    Get all movies featuring a specific actor
    
    Query params:
        limit: Maximum number of movies to return (default: 20)
    """
    if not recommender:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        limit = request.args.get('limit', 20, type=int)
        
        # Get movies by actor
        movies = recommender.get_movies_by_actor(actor_name, n=limit)
        
        # Format response
        results = []
        for movie_id, rating in movies:
            info = recommender.get_movie_info(movie_id)
            if info:
                poster_url = get_movie_poster(movie_id)
                results.append({
                    'id': movie_id,
                    'title': info['title'],
                    'year': info.get('year', 'N/A'),
                    'rating': float(rating),
                    'genre': info.get('genre', 'N/A'),
                    'director': info.get('director', 'N/A'),
                    'overview': info.get('overview', '')[:200],  # Truncated
                    'poster_url': poster_url
                })
        
        return jsonify({
            'actor': actor_name,
            'total_movies': len(results),
            'movies': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/actors/<actor_name>/profile', methods=['GET'])
def get_actor_profile(actor_name):
    """
    Get actor profile including statistics and top movies
    """
    if not recommender:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get actor info from recommender
        actor_data = recommender.get_actor_info(actor_name)
        
        return jsonify(actor_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/actors/<actor_name>/photo', methods=['GET'])
def get_actor_photo(actor_name):
    """
    Get actor photo URL from Wikipedia
    
    Query params:
        source: 'wikipedia' (default) or 'tmdb'
    """
    try:
        source = request.args.get('source', 'wikipedia')
        
        if source == 'wikipedia':
            # Use simple Wikipedia API (no pywikibot required)
            photo_url = get_actor_image_simple(actor_name)
            
            return jsonify({
                'actor': actor_name,
                'photo_url': photo_url,
                'source': 'wikipedia',
                'placeholder': '/assets/default-actor.png' if not photo_url else None
            })
        else:
            return jsonify({'error': 'Unsupported source'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/actors/<actor_name>/info', methods=['GET'])
def get_actor_wikipedia_info(actor_name):
    """
    Get comprehensive actor info from Wikipedia
    Including photo, biography, and Wikipedia link
    """
    try:
        # Get Wikipedia info
        wiki_info = get_actor_info_simple(actor_name)
        
        # Get filmography from recommender
        if recommender:
            actor_data = recommender.get_actor_info(actor_name)
            
            # Enrich top movies with posters
            top_movies = actor_data.get('top_movies', [])[:5]
            for movie in top_movies:
                movie['poster_url'] = get_movie_poster(movie['movie_id'])
                
            wiki_info['filmography'] = {
                'total_movies': actor_data.get('total_movies', 0),
                'average_rating': actor_data.get('average_rating', 0),
                'top_movies': top_movies
            }
        
        return jsonify(wiki_info)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/movies/<movie_id>/cast', methods=['GET'])
def get_movie_cast(movie_id):
    """
    Get cast list for a specific movie
    """
    if not recommender:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        movie_info = recommender.get_movie_info(movie_id)
        
        if not movie_info:
            return jsonify({'error': 'Movie not found'}), 404
        
        # Parse cast from comma-separated string
        # Parse cast from recommender DataFrame (since get_movie_info doesn't return cast)
        cast_list = []
        idx = recommender.movie_to_idx.get(movie_id)
        if idx is not None:
            row = recommender.movies_df.iloc[idx]
            # Try 'cast' column first, fallback to 'actors'
            cast_string = row.get('cast')
            if not cast_string or pd.isna(cast_string) or cast_string == 'Unknown':
                cast_string = row.get('actors', '')
                
            if cast_string and pd.notna(cast_string) and cast_string != 'Unknown':
                cast_list = [name.strip() for name in str(cast_string).split(',')]
        
        return jsonify({
            'movie_id': movie_id,
            'title': movie_info['title'],
            'cast': cast_list
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recommendations/categorized/<movie_id>', methods=['GET'])
def get_categorized_recommendations(movie_id):
    """
    Get categorized recommendations for a movie
    """
    if not recommender:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        n_per_category = request.args.get('n', 10, type=int)
        
        recommendations = recommender.get_categorized_recommendations(
            movie_id, 
            n_per_category=n_per_category
        )
        
        # Format each category
        formatted = {}
        for category, movies in recommendations.items():
            formatted[category] = []
            for rec_movie_id, score in movies:
                info = recommender.get_movie_info(rec_movie_id)
                if info:
                    formatted[category].append({
                        'id': rec_movie_id,
                        'title': info['title'],
                        'year': info.get('year'),
                        'rating': info.get('rating'),
                        'score': float(score)
                    })
        
        return jsonify(formatted)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    host = os.getenv('FLASK_HOST', '127.0.0.1')
    port = int(os.getenv('FLASK_PORT', '5000'))

    print("=" * 70)
    print("MOVIE RECOMMENDER API SERVER")
    print("=" * 70)
    print()
    print("Available endpoints:")
    print("  GET  /api/health")
    print("  GET  /api/actors/<name>/movies")
    print("  GET  /api/actors/<name>/profile")
    print("  GET  /api/actors/<name>/photo")
    print("  GET  /api/actors/<name>/info")
    print("  GET  /api/movies/<id>/cast")
    print("  GET  /api/recommendations/categorized/<id>")
    print()
    print(f"Starting server on http://{host}:{port}")
    print("=" * 70)
    
    app.run(debug=debug_mode, host=host, port=port)
