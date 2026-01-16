"""
Content-Based Filtering Recommender System.
Recommends movies based on movie features and attributes.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, Dict, Optional


class ContentBasedRecommender:
    """Content-based recommender using movie features."""
    
    def __init__(self, movies_df: pd.DataFrame, tfidf_matrix: Optional[np.ndarray] = None):
        """
        Initialize content-based recommender.
        
        Args:
            movies_df: DataFrame with movie features
            tfidf_matrix: Pre-computed TF-IDF matrix (optional)
        """
        self.movies_df = movies_df
        self.tfidf_matrix = tfidf_matrix
        self.similarity_matrix = None
        
        # Create movie ID to index mapping
        if 'imdb_id' in movies_df.columns:
            self.movie_id_col = 'imdb_id'
        elif 'movie_id' in movies_df.columns:
            self.movie_id_col = 'movie_id'
        else:
            self.movie_id_col = movies_df.columns[0]
        
        self.movie_to_idx = {
            movie_id: idx for idx, movie_id in enumerate(movies_df[self.movie_id_col])
        }
        self.idx_to_movie = {
            idx: movie_id for movie_id, idx in self.movie_to_idx.items()
        }
    
    def compute_similarity(self, use_tfidf: bool = True):
        """
        Compute similarity matrix between movies.
        
        Args:
            use_tfidf: Whether to use TF-IDF matrix (if available)
        """
        if use_tfidf and self.tfidf_matrix is not None:
            # Use TF-IDF matrix for similarity
            self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        else:
            # Use numerical features
            feature_cols = [col for col in self.movies_df.columns 
                          if col.startswith('genre_') or col.endswith('_normalized')]
            
            if feature_cols:
                feature_matrix = self.movies_df[feature_cols].fillna(0).values
                self.similarity_matrix = cosine_similarity(feature_matrix)
            else:
                raise ValueError("No suitable features found for similarity computation")
    
    def get_similar_movies(
        self,
        movie_id: str,
        n: int = 10,
        min_similarity: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Get movies similar to a given movie.
        
        Args:
            movie_id: Movie ID
            n: Number of similar movies to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (movie_id, similarity_score) tuples
        """
        if self.similarity_matrix is None:
            self.compute_similarity()
        
        if movie_id not in self.movie_to_idx:
            return []
        
        movie_idx = self.movie_to_idx[movie_id]
        similarities = self.similarity_matrix[movie_idx]
        
        # Get indices of similar movies
        similar_indices = np.argsort(similarities)[::-1][1:]  # Exclude self
        
        # Filter by minimum similarity and get top-n
        similar_movies = []
        for idx in similar_indices:
            if similarities[idx] >= min_similarity:
                similar_movie_id = self.idx_to_movie[idx]
                similar_movies.append((similar_movie_id, similarities[idx]))
                
                if len(similar_movies) >= n:
                    break
        
        return similar_movies
    
    def recommend_for_user(
        self,
        user_liked_movies: List[str],
        n_recommendations: int = 10,
        exclude_movies: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Recommend movies based on user's liked movies.
        
        Args:
            user_liked_movies: List of movie IDs the user liked
            n_recommendations: Number of recommendations to return
            exclude_movies: Movies to exclude from recommendations
            
        Returns:
            List of (movie_id, score) tuples
        """
        if self.similarity_matrix is None:
            self.compute_similarity()
        
        if exclude_movies is None:
            exclude_movies = user_liked_movies
        else:
            exclude_movies = list(set(exclude_movies + user_liked_movies))
        
        # Aggregate similarity scores across all liked movies
        all_scores = {}
        
        for liked_movie in user_liked_movies:
            if liked_movie not in self.movie_to_idx:
                continue
            
            movie_idx = self.movie_to_idx[liked_movie]
            similarities = self.similarity_matrix[movie_idx]
            
            for idx, similarity in enumerate(similarities):
                candidate_movie = self.idx_to_movie[idx]
                
                # Skip if it's a liked movie or excluded
                if candidate_movie in exclude_movies:
                    continue
                
                # Aggregate scores (average)
                if candidate_movie not in all_scores:
                    all_scores[candidate_movie] = []
                all_scores[candidate_movie].append(similarity)
        
        # Calculate average similarity for each candidate
        recommendations = [
            (movie_id, np.mean(scores))
            for movie_id, scores in all_scores.items()
        ]
        
        # Sort by score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_recommendations]
    
    def recommend_by_genre(
        self,
        genres: List[str],
        n_recommendations: int = 10,
        min_rating: Optional[float] = None
    ) -> List[Tuple[str, Dict]]:
        """
        Recommend movies by genre.
        
        Args:
            genres: List of genres to filter by
            n_recommendations: Number of recommendations
            min_rating: Minimum IMDb rating
            
        Returns:
            List of (movie_id, movie_info) tuples
        """
        # Filter movies by genre
        filtered_movies = self.movies_df[
            self.movies_df['genre'].str.contains('|'.join(genres), case=False, na=False)
        ].copy()
        
        # Filter by rating if specified
        if min_rating is not None and 'imdb_rating' in filtered_movies.columns:
            filtered_movies = filtered_movies[filtered_movies['imdb_rating'] >= min_rating]
        
        # Sort by rating
        if 'imdb_rating' in filtered_movies.columns:
            filtered_movies = filtered_movies.sort_values('imdb_rating', ascending=False)
        
        # Get top-n
        recommendations = []
        for idx, row in filtered_movies.head(n_recommendations).iterrows():
            movie_id = row[self.movie_id_col]
            movie_info = {
                'title': row.get('title', 'Unknown'),
                'genre': row.get('genre', 'Unknown'),
                'rating': row.get('imdb_rating', 0),
                'year': row.get('year', 'Unknown')
            }
            recommendations.append((movie_id, movie_info))
        
        return recommendations
    
    def get_movie_info(self, movie_id: str) -> Optional[Dict]:
        """
        Get information about a movie.
        
        Args:
            movie_id: Movie ID
            
        Returns:
            Dictionary with movie information
        """
        if movie_id not in self.movie_to_idx:
            return None
        
        idx = self.movie_to_idx[movie_id]
        movie_row = self.movies_df.iloc[idx]
        
        return {
            'movie_id': movie_id,
            'title': movie_row.get('title', 'Unknown'),
            'genre': movie_row.get('genre', 'Unknown'),
            'year': movie_row.get('year', 'Unknown'),
            'rating': movie_row.get('imdb_rating', 0),
            'director': movie_row.get('director', 'Unknown'),
            'actors': movie_row.get('actors', 'Unknown'),
            'plot': movie_row.get('plot', 'No plot available')
        }
    
    def recommend_by_features(
        self,
        preferred_genres: Optional[List[str]] = None,
        preferred_directors: Optional[List[str]] = None,
        min_rating: Optional[float] = None,
        year_range: Optional[Tuple[int, int]] = None,
        n_recommendations: int = 10
    ) -> List[Tuple[str, Dict]]:
        """
        Recommend movies based on multiple feature preferences.
        
        Args:
            preferred_genres: List of preferred genres
            preferred_directors: List of preferred directors
            min_rating: Minimum IMDb rating
            year_range: Tuple of (min_year, max_year)
            n_recommendations: Number of recommendations
            
        Returns:
            List of (movie_id, movie_info) tuples
        """
        filtered_df = self.movies_df.copy()
        
        # Filter by genres
        if preferred_genres:
            genre_pattern = '|'.join(preferred_genres)
            filtered_df = filtered_df[
                filtered_df['genre'].str.contains(genre_pattern, case=False, na=False)
            ]
        
        # Filter by directors
        if preferred_directors:
            director_pattern = '|'.join(preferred_directors)
            filtered_df = filtered_df[
                filtered_df['director'].str.contains(director_pattern, case=False, na=False)
            ]
        
        # Filter by rating
        if min_rating is not None and 'imdb_rating' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['imdb_rating'] >= min_rating]
        
        # Filter by year range
        if year_range and 'year' in filtered_df.columns:
            min_year, max_year = year_range
            filtered_df = filtered_df[
                (filtered_df['year'] >= min_year) & (filtered_df['year'] <= max_year)
            ]
        
        # Sort by rating
        if 'imdb_rating' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('imdb_rating', ascending=False)
        
        # Get recommendations
        recommendations = []
        for idx, row in filtered_df.head(n_recommendations).iterrows():
            movie_id = row[self.movie_id_col]
            movie_info = {
                'title': row.get('title', 'Unknown'),
                'genre': row.get('genre', 'Unknown'),
                'rating': row.get('imdb_rating', 0),
                'year': row.get('year', 'Unknown'),
                'director': row.get('director', 'Unknown')
            }
            recommendations.append((movie_id, movie_info))
        
        return recommendations


if __name__ == "__main__":
    # Example usage
    print("Content-Based Recommender")
    print("=========================\n")
    
    # Create sample movie data
    movies_data = {
        'imdb_id': ['tt0111161', 'tt0068646', 'tt0468569', 'tt0137523', 'tt0109830'],
        'title': [
            'The Shawshank Redemption',
            'The Godfather',
            'The Dark Knight',
            'Fight Club',
            'Forrest Gump'
        ],
        'genre': [
            'Drama',
            'Crime, Drama',
            'Action, Crime, Drama',
            'Drama',
            'Drama, Romance'
        ],
        'year': [1994, 1972, 2008, 1999, 1994],
        'imdb_rating': [9.3, 9.2, 9.0, 8.8, 8.8],
        'director': [
            'Frank Darabont',
            'Francis Ford Coppola',
            'Christopher Nolan',
            'David Fincher',
            'Robert Zemeckis'
        ],
        'actors': [
            'Tim Robbins, Morgan Freeman',
            'Marlon Brando, Al Pacino',
            'Christian Bale, Heath Ledger',
            'Brad Pitt, Edward Norton',
            'Tom Hanks, Robin Wright'
        ],
        'plot': [
            'Two imprisoned men bond over years',
            'Aging patriarch of organized crime',
            'Batman fights the Joker',
            'Insomniac office worker forms fight club',
            'Life journey of Forrest Gump'
        ]
    }
    
    movies_df = pd.DataFrame(movies_data)
    
    # Add normalized features
    movies_df['imdb_rating_normalized'] = (movies_df['imdb_rating'] - movies_df['imdb_rating'].min()) / \
                                           (movies_df['imdb_rating'].max() - movies_df['imdb_rating'].min())
    
    # Create recommender
    recommender = ContentBasedRecommender(movies_df)
    
    # Get similar movies
    print("1. Similar Movies")
    print("-" * 40)
    movie_id = 'tt0111161'  # Shawshank Redemption
    movie_info = recommender.get_movie_info(movie_id)
    print(f"Finding movies similar to: {movie_info['title']}")
    
    similar_movies = recommender.get_similar_movies(movie_id, n=3)
    for similar_id, score in similar_movies:
        info = recommender.get_movie_info(similar_id)
        print(f"  {info['title']} (score: {score:.3f})")
    
    # Recommend based on user preferences
    print("\n2. Recommendations Based on Liked Movies")
    print("-" * 40)
    user_liked = ['tt0111161', 'tt0068646']  # Shawshank and Godfather
    print(f"User liked: {[recommender.get_movie_info(m)['title'] for m in user_liked]}")
    
    recommendations = recommender.recommend_for_user(user_liked, n_recommendations=2)
    print("Recommendations:")
    for movie_id, score in recommendations:
        info = recommender.get_movie_info(movie_id)
        print(f"  {info['title']} (score: {score:.3f})")
    
    # Recommend by genre
    print("\n3. Recommendations by Genre")
    print("-" * 40)
    genre_recs = recommender.recommend_by_genre(['Drama'], n_recommendations=3, min_rating=8.5)
    print("Top Drama movies:")
    for movie_id, info in genre_recs:
        print(f"  {info['title']} - Rating: {info['rating']}")
    
    # Recommend by multiple features
    print("\n4. Recommendations by Multiple Features")
    print("-" * 40)
    feature_recs = recommender.recommend_by_features(
        preferred_genres=['Drama'],
        min_rating=8.8,
        year_range=(1990, 2000),
        n_recommendations=3
    )
    print("Drama movies from 1990-2000 with rating >= 8.8:")
    for movie_id, info in feature_recs:
        print(f"  {info['title']} ({info['year']}) - Rating: {info['rating']}")
    
    print("\nContent-based filtering demo complete!")
