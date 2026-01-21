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
    
    # Check model size for optimization verification
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"✅ Model loaded successfully ({size_mb:.2f} MB)")
    if size_mb < 100:
        print("🚀 Optimized Model Detected (High Performance Mode)")
    else:
        print("⚠️ Warning: Large model detected. Consider retraining with optimization.")
    
    # Precompute title lookup for autocomplete (O(1) startup cost)
    print("Building autocomplete index...")
    MOVIE_TITLES = []
    TITLE_LOOKUP = {}
    
    for movie_id, idx in recommender.movie_to_idx.items():
        row = recommender.movies_df.iloc[idx]
        title = row['title']
        year = row.get('year', 'N/A')
        
        MOVIE_TITLES.append(title)
        TITLE_LOOKUP[title] = {
            'id': movie_id,
            'year': year
        }
    
    print(f"✅ Indexed {len(MOVIE_TITLES)} movie titles for autocomplete")
        
except Exception as e:
    print(f"❌ Error loading model: {e}")
    recommender = None
    size_mb = 0
    MOVIE_TITLES = []
    TITLE_LOOKUP = {}

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

@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'message': 'Movie Recommendation API is running',
        'endpoints': {
            'health': '/api/health',
            'autocomplete': '/api/search/autocomplete?q=query',
            'actor_movies': '/api/actors/<name>/movies',
            'recommendations': '/api/recommendations/categorized/<id>'
        },
        'status': 'active'
    })

@app.route('/api/search/autocomplete', methods=['GET'])
def autocomplete():
    """
    Autocomplete endpoint with hybrid prefix+fuzzy matching
    
    Query params:
        q: Search query (min 2 characters)
    """
    if not recommender:
        return jsonify({'suggestions': []}), 500
    
    try:
        query = request.args.get('q', '').strip().lower()
        
        # Minimum query length check
        if len(query) < 2:
            return jsonify({'suggestions': []})
        
        # 1. Prefix matches (fast, deterministic)
        prefix_matches = [
            title for title in MOVIE_TITLES
            if title.lower().startswith(query)
        ][:10]
        
        results = prefix_matches
        
        # 2. Fuzzy fallback if insufficient results
        if len(results) < 10:
            from difflib import get_close_matches
            fuzzy = get_close_matches(
                query,
                MOVIE_TITLES,
                n=10 - len(results),
                cutoff=0.4
            )
            results.extend(fuzzy)
        
        # 3. Format response (remove duplicates)
        suggestions = []
        for title in dict.fromkeys(results):
            meta = TITLE_LOOKUP[title]
            suggestions.append({
                'id': meta['id'],
                'title': title,
                'year': meta['year']
            })
        
        return jsonify({'suggestions': suggestions[:10]})
        
    except Exception as e:
        print(f"Error in autocomplete: {e}")
        return jsonify({'suggestions': []}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': recommender is not None,
        'model_size_mb': round(size_mb, 2)
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

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations_post():
    """
    Get recommendations based on input movies (POST)
    Handles title-to-ID lookup.
    """
    if not recommender:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        input_movies = data.get('input_movies', [])
        limit = data.get('limit', 10)
        
        if not input_movies:
            return jsonify({'error': 'No input movies provided'}), 400
            
        movie_input = input_movies[0]
        
        # 1. Resolve Title -> ID
        movie_id = movie_input
        # Check if input is a known ID
        if movie_input not in recommender.movie_to_idx:
            # Try finding by title
            found = False
            for mid, idx in recommender.movie_to_idx.items():
                title = recommender.movies_df.iloc[idx]['title']
                if str(title).lower() == str(movie_input).lower():
                    movie_id = mid
                    found = True
                    break
            
            if not found:
                 return jsonify({'error': f"Movie '{movie_input}' not found in database"}), 404

        # 2. Get Recommendations
        recommendations = recommender.get_categorized_recommendations(
            movie_id, 
            n_per_category=limit
        )
        
        # 3. Format Response
        formatted = {}
        for category, movies in recommendations.items():
            formatted[category] = []
            for rec_movie_id, score in movies:
                info = recommender.get_movie_info(rec_movie_id)
                if info:
                    # Enrich with poster
                    poster = get_movie_poster(rec_movie_id)
                    formatted[category].append({
                        'id': rec_movie_id,
                        'title': info['title'],
                        'year': info.get('year'),
                        'rating': info.get('rating'),
                        'score': float(score),
                        'poster_url': poster,
                        'overview': info.get('overview', '')[:200]
                    })
        
        # Add explanations
        explanations = [
            f"Recommendations based on '{movie_id}'",
            f"Found {len(formatted.get('similar_content', []))} matches based on plot & features"
        ]

        return jsonify({
            'recommendations': formatted,
            'trace_id': 'trace_' + str(movie_id), # Mock trace ID
            'explanations': explanations,
            'model_version': 'v2.1'
        })
        
    except Exception as e:
        print(f"Error in recommendations: {e}")
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
