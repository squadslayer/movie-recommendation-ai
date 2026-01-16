"""
Collaborative Filtering Recommender System.
Implements user-based and item-based collaborative filtering algorithms.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from typing import List, Tuple, Dict, Optional


class CollaborativeFilteringRecommender:
    """Collaborative filtering recommender using user-item interactions."""
    
    def __init__(self, user_item_matrix: pd.DataFrame):
        """
        Initialize the collaborative filtering recommender.
        
        Args:
            user_item_matrix: User-item rating matrix (users as rows, items as columns)
        """
        self.user_item_matrix = user_item_matrix
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        
    def compute_user_similarity(self, metric: str = 'cosine') -> np.ndarray:
        """
        Compute similarity between users.
        
        Args:
            metric: Similarity metric ('cosine' by default)
            
        Returns:
            User similarity matrix
        """
        if metric == 'cosine':
            # Replace 0 with NaN for similarity calculation
            matrix = self.user_item_matrix.values
            self.user_similarity_matrix = cosine_similarity(matrix)
        
        return self.user_similarity_matrix
    
    def compute_item_similarity(self, metric: str = 'cosine') -> np.ndarray:
        """
        Compute similarity between items (movies).
        
        Args:
            metric: Similarity metric ('cosine' by default)
            
        Returns:
            Item similarity matrix
        """
        if metric == 'cosine':
            # Transpose to get items as rows
            matrix = self.user_item_matrix.T.values
            self.item_similarity_matrix = cosine_similarity(matrix)
        
        return self.item_similarity_matrix
    
    def predict_user_based(self, user_id: str, movie_id: str, k: int = 10) -> float:
        """
        Predict rating using user-based collaborative filtering.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            k: Number of similar users to consider
            
        Returns:
            Predicted rating
        """
        if self.user_similarity_matrix is None:
            self.compute_user_similarity()
        
        try:
            user_idx = self.user_item_matrix.index.get_loc(user_id)
            movie_idx = self.user_item_matrix.columns.get_loc(movie_id)
        except KeyError:
            return 0.0
        
        # Get similar users
        user_similarities = self.user_similarity_matrix[user_idx]
        
        # Get ratings for this movie from all users
        movie_ratings = self.user_item_matrix.iloc[:, movie_idx].values
        
        # Filter out users who haven't rated this movie
        rated_mask = movie_ratings > 0
        
        if not rated_mask.any():
            return 0.0
        
        # Get top-k similar users who have rated this movie
        similarities = user_similarities * rated_mask
        top_k_indices = np.argsort(similarities)[::-1][1:k+1]  # Exclude self
        
        # Calculate weighted average
        top_similarities = similarities[top_k_indices]
        top_ratings = movie_ratings[top_k_indices]
        
        if top_similarities.sum() == 0:
            return 0.0
        
        predicted_rating = np.sum(top_similarities * top_ratings) / np.sum(np.abs(top_similarities))
        
        return predicted_rating
    
    def predict_item_based(self, user_id: str, movie_id: str, k: int = 10) -> float:
        """
        Predict rating using item-based collaborative filtering.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            k: Number of similar items to consider
            
        Returns:
            Predicted rating
        """
        if self.item_similarity_matrix is None:
            self.compute_item_similarity()
        
        try:
            user_idx = self.user_item_matrix.index.get_loc(user_id)
            movie_idx = self.user_item_matrix.columns.get_loc(movie_id)
        except KeyError:
            return 0.0
        
        # Get similar movies
        movie_similarities = self.item_similarity_matrix[movie_idx]
        
        # Get user's ratings for all movies
        user_ratings = self.user_item_matrix.iloc[user_idx, :].values
        
        # Filter out movies user hasn't rated
        rated_mask = user_ratings > 0
        
        if not rated_mask.any():
            return 0.0
        
        # Get top-k similar movies that user has rated
        similarities = movie_similarities * rated_mask
        top_k_indices = np.argsort(similarities)[::-1][1:k+1]  # Exclude self
        
        # Calculate weighted average
        top_similarities = similarities[top_k_indices]
        top_ratings = user_ratings[top_k_indices]
        
        if top_similarities.sum() == 0:
            return 0.0
        
        predicted_rating = np.sum(top_similarities * top_ratings) / np.sum(np.abs(top_similarities))
        
        return predicted_rating
    
    def recommend_user_based(
        self,
        user_id: str,
        n_recommendations: int = 10,
        k_neighbors: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Recommend movies for a user using user-based collaborative filtering.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations to return
            k_neighbors: Number of similar users to consider
            
        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        if user_id not in self.user_item_matrix.index:
            return []
        
        # Get movies user hasn't rated
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index.tolist()
        
        # Predict ratings for unrated movies
        predictions = []
        for movie_id in unrated_movies:
            predicted_rating = self.predict_user_based(user_id, movie_id, k=k_neighbors)
            if predicted_rating > 0:
                predictions.append((movie_id, predicted_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n_recommendations]
    
    def recommend_item_based(
        self,
        user_id: str,
        n_recommendations: int = 10,
        k_neighbors: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Recommend movies for a user using item-based collaborative filtering.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations to return
            k_neighbors: Number of similar items to consider
            
        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        if user_id not in self.user_item_matrix.index:
            return []
        
        # Get movies user hasn't rated
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index.tolist()
        
        # Predict ratings for unrated movies
        predictions = []
        for movie_id in unrated_movies:
            predicted_rating = self.predict_item_based(user_id, movie_id, k=k_neighbors)
            if predicted_rating > 0:
                predictions.append((movie_id, predicted_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n_recommendations]
    
    def get_similar_users(self, user_id: str, n: int = 5) -> List[Tuple[str, float]]:
        """
        Get most similar users to a given user.
        
        Args:
            user_id: User ID
            n: Number of similar users to return
            
        Returns:
            List of (user_id, similarity_score) tuples
        """
        if self.user_similarity_matrix is None:
            self.compute_user_similarity()
        
        if user_id not in self.user_item_matrix.index:
            return []
        
        user_idx = self.user_item_matrix.index.get_loc(user_id)
        similarities = self.user_similarity_matrix[user_idx]
        
        # Get top-n similar users (excluding self)
        top_indices = np.argsort(similarities)[::-1][1:n+1]
        
        similar_users = [
            (self.user_item_matrix.index[idx], similarities[idx])
            for idx in top_indices
        ]
        
        return similar_users
    
    def get_similar_items(self, movie_id: str, n: int = 5) -> List[Tuple[str, float]]:
        """
        Get most similar movies to a given movie.
        
        Args:
            movie_id: Movie ID
            n: Number of similar movies to return
            
        Returns:
            List of (movie_id, similarity_score) tuples
        """
        if self.item_similarity_matrix is None:
            self.compute_item_similarity()
        
        if movie_id not in self.user_item_matrix.columns:
            return []
        
        movie_idx = self.user_item_matrix.columns.get_loc(movie_id)
        similarities = self.item_similarity_matrix[movie_idx]
        
        # Get top-n similar movies (excluding self)
        top_indices = np.argsort(similarities)[::-1][1:n+1]
        
        similar_movies = [
            (self.user_item_matrix.columns[idx], similarities[idx])
            for idx in top_indices
        ]
        
        return similar_movies


class MatrixFactorizationRecommender:
    """Matrix Factorization using SVD for collaborative filtering."""
    
    def __init__(self, n_factors: int = 20):
        """
        Initialize matrix factorization recommender.
        
        Args:
            n_factors: Number of latent factors
        """
        self.n_factors = n_factors
        self.svd = TruncatedSVD(n_components=n_factors, random_state=42)
        self.user_factors = None
        self.item_factors = None
        self.user_index = None
        self.item_index = None
        
    def fit(self, user_item_matrix: pd.DataFrame):
        """
        Fit the matrix factorization model.
        
        Args:
            user_item_matrix: User-item rating matrix
        """
        self.user_index = user_item_matrix.index
        self.item_index = user_item_matrix.columns
        
        # Convert to sparse matrix
        matrix = user_item_matrix.values
        
        # Perform SVD
        self.user_factors = self.svd.fit_transform(matrix)
        self.item_factors = self.svd.components_.T
        
    def predict(self, user_id: str, movie_id: str) -> float:
        """
        Predict rating for a user-movie pair.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Predicted rating
        """
        if user_id not in self.user_index or movie_id not in self.item_index:
            return 0.0
        
        user_idx = self.user_index.get_loc(user_id)
        movie_idx = self.item_index.get_loc(movie_id)
        
        # Dot product of user and item factors
        predicted_rating = np.dot(self.user_factors[user_idx], self.item_factors[movie_idx])
        
        return max(0, min(10, predicted_rating))  # Clip to valid rating range
    
    def recommend(self, user_id: str, user_item_matrix: pd.DataFrame, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """
        Recommend movies for a user.
        
        Args:
            user_id: User ID
            user_item_matrix: Original user-item matrix (to know what user hasn't rated)
            n_recommendations: Number of recommendations
            
        Returns:
            List of (movie_id, predicted_rating) tuples
        """
        if user_id not in self.user_index:
            return []
        
        # Get movies user hasn't rated
        user_ratings = user_item_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index.tolist()
        
        # Predict ratings
        predictions = []
        for movie_id in unrated_movies:
            predicted_rating = self.predict(user_id, movie_id)
            predictions.append((movie_id, predicted_rating))
        
        # Sort and return top-n
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]


if __name__ == "__main__":
    # Example usage
    print("Collaborative Filtering Recommender")
    print("====================================\n")
    
    # Create sample user-item matrix
    data = {
        'movie_1': [5, 3, 0, 1, 0],
        'movie_2': [4, 0, 0, 1, 0],
        'movie_3': [0, 1, 0, 5, 0],
        'movie_4': [0, 1, 0, 4, 0],
        'movie_5': [0, 0, 5, 0, 4],
    }
    
    user_item_matrix = pd.DataFrame(
        data,
        index=['user_1', 'user_2', 'user_3', 'user_4', 'user_5']
    )
    
    print("User-Item Matrix:")
    print(user_item_matrix)
    print()
    
    # User-based CF
    print("1. User-Based Collaborative Filtering")
    print("-" * 40)
    cf = CollaborativeFilteringRecommender(user_item_matrix)
    
    user_id = 'user_1'
    recommendations = cf.recommend_user_based(user_id, n_recommendations=3)
    print(f"Recommendations for {user_id}:")
    for movie_id, rating in recommendations:
        print(f"  {movie_id}: {rating:.2f}")
    
    # Similar users
    similar_users = cf.get_similar_users(user_id, n=2)
    print(f"\nSimilar users to {user_id}:")
    for similar_user, similarity in similar_users:
        print(f"  {similar_user}: {similarity:.3f}")
    
    # Item-based CF
    print("\n2. Item-Based Collaborative Filtering")
    print("-" * 40)
    recommendations = cf.recommend_item_based(user_id, n_recommendations=3)
    print(f"Recommendations for {user_id}:")
    for movie_id, rating in recommendations:
        print(f"  {movie_id}: {rating:.2f}")
    
    # Similar items
    movie_id = 'movie_1'
    similar_items = cf.get_similar_items(movie_id, n=2)
    print(f"\nSimilar movies to {movie_id}:")
    for similar_movie, similarity in similar_items:
        print(f"  {similar_movie}: {similarity:.3f}")
    
    # Matrix Factorization
    print("\n3. Matrix Factorization (SVD)")
    print("-" * 40)
    mf = MatrixFactorizationRecommender(n_factors=3)
    mf.fit(user_item_matrix)
    
    recommendations = mf.recommend(user_id, user_item_matrix, n_recommendations=3)
    print(f"Recommendations for {user_id}:")
    for movie_id, rating in recommendations:
        print(f"  {movie_id}: {rating:.2f}")
    
    print("\nCollaborative filtering demo complete!")
