"""
Enhanced Content-Based Recommender with Categorized Suggestions.
Provides recommendations based on genre, director, year, and content similarity.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import List, Tuple, Dict, Optional


class EnhancedRecommender:
    """Enhanced recommender with categorized recommendations."""
    
    def __init__(self, movies_df: pd.DataFrame, tfidf_matrix: Optional[np.ndarray] = None):
        """
        Initialize enhanced recommender.
        
        Args:
            movies_df: DataFrame with movie features (must include genre, director, year)
            tfidf_matrix: Pre-computed TF-IDF matrix (optional)
        """
        self.movies_df = movies_df
        self.tfidf_matrix = tfidf_matrix
        
        # Create movie ID to index mapping
        movie_id_col = movies_df.columns[0]
        self.movie_to_idx = {
            movie_id: idx for idx, movie_id in enumerate(movies_df[movie_id_col])
        }
        self.idx_to_movie = {idx: movie_id for movie_id, idx in self.movie_to_idx.items()}
        
        # Compute similarity matrix if TF-IDF is available
        # OPTIMIZATION: We do NOT compute the full NxN matrix here anymore to save memory (3GB+ -> <100MB).
        # We will calculate similarity on-the-fly using linear_kernel.
        self.similarity_matrix = None
    
    def get_movie_info(self, movie_id: str) -> Dict:
        """Get information about a movie."""
        if movie_id not in self.movie_to_idx:
            return {}
        
        idx = self.movie_to_idx[movie_id]
        row = self.movies_df.iloc[idx]
        
        return {
            'id': movie_id,
            'title': row.get('title', 'Unknown'),
            'genre': row.get('genre', 'Unknown'),
            'director': row.get('director', 'Unknown'),
            'year': row.get('year', 'Unknown'),
            'rating': row.get('vote_average', row.get('imdb_rating', 'N/A')),
            'overview': row.get('overview', '')
        }
    
    def get_similar_genre(self, movie_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get movies with similar genres.
        
        Args:
            movie_id: Movie ID
            n: Number of recommendations
            
        Returns:
            List of (movie_id, rating) tuples
        """
        movie_info = self.get_movie_info(movie_id)
        if not movie_info or movie_info['genre'] == 'Unknown':
            return []
        
        # Get genres for the input movie
        input_genres = set(g.strip() for g in str(movie_info['genre']).split(','))
        
        # Find movies with overlapping genres
        similar_movies = []
        for idx, row in self.movies_df.iterrows():
            other_id = row[self.movies_df.columns[0]]
            
            # Skip the input movie
            if other_id == movie_id:
                continue
            
            # Check genre overlap
            if pd.notna(row.get('genre')):
                other_genres = set(g.strip() for g in str(row['genre']).split(','))
                overlap = len(input_genres & other_genres)
                
                if overlap > 0:
                    rating = row.get('vote_average', row.get('imdb_rating', 0))
                    similar_movies.append((other_id, float(rating), overlap))
        
        # Sort by genre overlap (desc) then rating (desc)
        similar_movies.sort(key=lambda x: (x[2], x[1]), reverse=True)
        
        # Return top N with just id and rating
        return [(movie_id, rating) for movie_id, rating, _ in similar_movies[:n]]
    
    def get_same_director(self, movie_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get movies by the same director.
        
        Args:
            movie_id: Movie ID
            n: Number of recommendations
            
        Returns:
            List of (movie_id, rating) tuples
        """
        movie_info = self.get_movie_info(movie_id)
        if not movie_info or movie_info['director'] == 'Unknown':
            return []
        
        director = movie_info['director']
        
        # Find movies by same director
        same_director_movies = []
        for idx, row in self.movies_df.iterrows():
            other_id = row[self.movies_df.columns[0]]
            
            # Skip the input movie
            if other_id == movie_id:
                continue
            
            # Check if same director
            if pd.notna(row.get('director')) and str(row['director']) == director:
                rating = row.get('vote_average', row.get('imdb_rating', 0))
                same_director_movies.append((other_id, float(rating)))
        
        # Sort by rating (desc)
        same_director_movies.sort(key=lambda x: x[1], reverse=True)
        
        return same_director_movies[:n]
    
    def get_popular_that_year(self, movie_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get popular movies from the same year.
        
        Args:
            movie_id: Movie ID
            n: Number of recommendations
            
        Returns:
            List of (movie_id, rating) tuples
        """
        movie_info = self.get_movie_info(movie_id)
        if not movie_info or movie_info['year'] == 'Unknown':
            return []
        
        year = str(movie_info['year'])[:4]  # Get year as string
        
        # Find movies from same year
        same_year_movies = []
        for idx, row in self.movies_df.iterrows():
            other_id = row[self.movies_df.columns[0]]
            
            # Skip the input movie
            if other_id == movie_id:
                continue
            
            # Check if same year
            if pd.notna(row.get('year')) and str(row['year'])[:4] == year:
                rating = row.get('vote_average', row.get('imdb_rating', 0))
                popularity = row.get('popularity', 0)
                # Use rating as primary, popularity as secondary sort
                same_year_movies.append((other_id, float(rating), float(popularity)))
        
        # Sort by rating (desc) then popularity (desc)
        same_year_movies.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        # Return top N with just id and rating
        return [(movie_id, rating) for movie_id, rating, _ in same_year_movies[:n]]
    
    def get_similar_content(self, movie_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get movies with similar content (plot-based).
        
        Args:
            movie_id: Movie ID
            n: Number of recommendations
            
        Returns:
            List of (movie_id, similarity_score) tuples
        """
        if self.tfidf_matrix is None:
            return []
        
        if movie_id not in self.movie_to_idx:
            return []
        
        idx = self.movie_to_idx[movie_id]
        
        # Calculate similarity on-the-fly for just this movie vs all others
        # vector is 1xN, matrix is MxN -> result is 1xM cosine similarities (since vectors are normalized)
        from sklearn.metrics.pairwise import linear_kernel
        cosine_sim = linear_kernel(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        
        # Get top similar movies (excluding itself)
        # argsort is fast enough for 20k items (~2ms)
        similar_indices = np.argsort(cosine_sim)[::-1][1:n+1]
        
        recommendations = []
        for sim_idx in similar_indices:
            if sim_idx < len(self.idx_to_movie):
                sim_movie_id = self.idx_to_movie[sim_idx]
                score = cosine_sim[sim_idx]
                recommendations.append((sim_movie_id, float(score)))
        
        return recommendations
    
    def get_categorized_recommendations(self, movie_id: str, n_per_category: int = 10) -> Dict[str, List]:
        """
        Get categorized recommendations for a movie.
        
        Args:
            movie_id: Movie ID
            n_per_category: Number of recommendations per category
            
        Returns:
            Dictionary with categorized recommendations
        """
        return {
            'similar_genre': self.get_similar_genre(movie_id, n=n_per_category),
            'same_director': self.get_same_director(movie_id, n=n_per_category),
            'popular_that_year': self.get_popular_that_year(movie_id, n=n_per_category),
            'similar_content': self.get_similar_content(movie_id, n=n_per_category)
        }
    
    def get_movies_by_actor(self, actor_name: str, n: int = 20) -> List[Tuple[str, float]]:
        """
        Get all movies featuring a specific actor.
        
        Args:
            actor_name: Name of the actor (case-insensitive)
            n: Maximum number of movies to return
            
        Returns:
            List of (movie_id, rating) tuples sorted by rating
        """
        if 'cast' not in self.movies_df.columns:
            logging.warning("Cast information not available in dataset")
            return []
        
        # Normalize actor name for comparison
        actor_name_lower = actor_name.lower().strip()
        
        # Vectorized filtering
        df = self.movies_df.copy()
        df['cast_lower'] = df['cast'].fillna('').astype(str).str.lower()
        
        # Filter rows where actor appears in cast
        mask = df['cast_lower'].apply(
            lambda cast: actor_name_lower in [n.strip() for n in cast.split(',')]
        )
        filtered = df[mask]
        
        # Extract movie_id and rating
        movie_id_col = self.movies_df.columns[0]
        matching_movies = [
            (row[movie_id_col], float(row.get('vote_average', row.get('imdb_rating', 0)) or 0)) 
            for _, row in filtered.iterrows()
        ]
        
        # Sort by rating (descending)
        matching_movies.sort(key=lambda x: x[1], reverse=True)
        
        return matching_movies[:n]
    
    def get_actor_info(self, actor_name: str) -> Dict:
        """
        Get information about an actor based on their filmography.
        
        Args:
            actor_name: Name of the actor
            
        Returns:
            Dictionary with actor statistics and filmography
        """
        movies = self.get_movies_by_actor(actor_name, n=100)  # Get all movies
        
        if not movies:
            return {
                'actor_name': actor_name,
                'total_movies': 0,
                'average_rating': 0,
                'top_movies': []
            }
        
        # Calculate statistics
        ratings = [rating for _, rating in movies]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        # Get top 5 movies
        top_movies = []
        for movie_id, rating in movies[:5]:
            movie_info = self.get_movie_info(movie_id)
            if movie_info:
                top_movies.append({
                    'movie_id': str(movie_id),
                    'title': str(movie_info['title']),
                    'rating': float(rating),
                    'year': str(movie_info.get('year', 'N/A')),
                    'genre': str(movie_info.get('genre', 'N/A'))
                })
        
        return {
            'actor_name': actor_name,
            'total_movies': int(len(movies)),
            'average_rating': float(round(avg_rating, 2)),
            'top_movies': top_movies,
            'all_movie_ids': [str(movie_id) for movie_id, _ in movies]
        }
