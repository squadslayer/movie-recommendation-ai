"""
Hybrid Recommender System.
Combines collaborative filtering and content-based approaches for better recommendations.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import sys
sys.path.append('..')

from .collaborative_filtering import CollaborativeFilteringRecommender, MatrixFactorizationRecommender
from .content_based import ContentBasedRecommender


class HybridRecommender:
    """Hybrid recommender combining collaborative and content-based approaches."""
    
    def __init__(
        self,
        cf_recommender: Optional[CollaborativeFilteringRecommender] = None,
        cb_recommender: Optional[ContentBasedRecommender] = None,
        mf_recommender: Optional[MatrixFactorizationRecommender] = None
    ):
        """
        Initialize hybrid recommender.
        
        Args:
            cf_recommender: Collaborative filtering recommender
            cb_recommender: Content-based recommender
            mf_recommender: Matrix factorization recommender
        """
        self.cf_recommender = cf_recommender
        self.cb_recommender = cb_recommender
        self.mf_recommender = mf_recommender
    
    def weighted_hybrid(
        self,
        user_id: str,
        n_recommendations: int = 10,
        cf_weight: float = 0.5,
        cb_weight: float = 0.5,
        user_item_matrix: Optional[pd.DataFrame] = None,
        user_liked_movies: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Combine recommendations using weighted average.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            cf_weight: Weight for collaborative filtering (0-1)
            cb_weight: Weight for content-based filtering (0-1)
            user_item_matrix: User-item matrix (for MF recommender)
            user_liked_movies: List of movies user liked (for CB recommender)
            
        Returns:
            List of (movie_id, score) tuples
        """
        # Normalize weights
        total_weight = cf_weight + cb_weight
        cf_weight /= total_weight
        cb_weight /= total_weight
        
        combined_scores = {}
        
        # Get collaborative filtering recommendations
        if self.cf_recommender is not None:
            cf_recs = self.cf_recommender.recommend_user_based(
                user_id, 
                n_recommendations=n_recommendations * 2
            )
            
            for movie_id, score in cf_recs:
                if movie_id not in combined_scores:
                    combined_scores[movie_id] = 0
                combined_scores[movie_id] += score * cf_weight
        
        # Get content-based recommendations
        if self.cb_recommender is not None and user_liked_movies:
            cb_recs = self.cb_recommender.recommend_for_user(
                user_liked_movies,
                n_recommendations=n_recommendations * 2
            )
            
            for movie_id, score in cb_recs:
                if movie_id not in combined_scores:
                    combined_scores[movie_id] = 0
                combined_scores[movie_id] += score * cb_weight
        
        # Sort by combined score
        recommendations = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return recommendations[:n_recommendations]
    
    def switching_hybrid(
        self,
        user_id: str,
        n_recommendations: int = 10,
        min_user_ratings: int = 5,
        user_item_matrix: Optional[pd.DataFrame] = None,
        user_liked_movies: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Switch between CF and CB based on data availability.
        Uses CF if user has enough ratings, otherwise uses CB.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            min_user_ratings: Minimum ratings needed to use CF
            user_item_matrix: User-item matrix
            user_liked_movies: List of movies user liked
            
        Returns:
            List of (movie_id, score) tuples
        """
        # Check if user has enough ratings for CF
        use_cf = False
        if user_item_matrix is not None and user_id in user_item_matrix.index:
            user_ratings = user_item_matrix.loc[user_id]
            num_ratings = (user_ratings > 0).sum()
            use_cf = num_ratings >= min_user_ratings
        
        if use_cf and self.cf_recommender is not None:
            # Use collaborative filtering
            return self.cf_recommender.recommend_user_based(
                user_id,
                n_recommendations=n_recommendations
            )
        elif self.cb_recommender is not None and user_liked_movies:
            # Use content-based filtering
            return self.cb_recommender.recommend_for_user(
                user_liked_movies,
                n_recommendations=n_recommendations
            )
        else:
            return []
    
    def cascade_hybrid(
        self,
        user_id: str,
        n_recommendations: int = 10,
        cf_threshold: float = 0.5,
        user_item_matrix: Optional[pd.DataFrame] = None,
        user_liked_movies: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Use CF first, then refine with CB.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            cf_threshold: Minimum CF score to keep recommendation
            user_item_matrix: User-item matrix
            user_liked_movies: List of movies user liked
            
        Returns:
            List of (movie_id, score) tuples
        """
        recommendations = []
        
        # Get CF recommendations
        if self.cf_recommender is not None:
            cf_recs = self.cf_recommender.recommend_user_based(
                user_id,
                n_recommendations=n_recommendations * 2
            )
            
            # Keep only high-confidence CF recommendations
            for movie_id, score in cf_recs:
                if score >= cf_threshold:
                    recommendations.append((movie_id, score))
        
        # If we don't have enough, supplement with CB
        if len(recommendations) < n_recommendations:
            if self.cb_recommender is not None and user_liked_movies:
                # Get CB recommendations, excluding already recommended movies
                exclude_movies = [movie_id for movie_id, _ in recommendations]
                
                cb_recs = self.cb_recommender.recommend_for_user(
                    user_liked_movies,
                    n_recommendations=n_recommendations - len(recommendations),
                    exclude_movies=exclude_movies
                )
                
                recommendations.extend(cb_recs)
        
        return recommendations[:n_recommendations]
    
    def feature_combination_hybrid(
        self,
        user_id: str,
        n_recommendations: int = 10,
        user_item_matrix: Optional[pd.DataFrame] = None,
        user_liked_movies: Optional[List[str]] = None
    ) -> List[Tuple[str, float, Dict]]:
        """
        Combine features from both CF and CB for a more nuanced recommendation.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            user_item_matrix: User-item matrix
            user_liked_movies: List of movies user liked
            
        Returns:
            List of (movie_id, combined_score, metadata) tuples
        """
        cf_scores = {}
        cb_scores = {}
        
        # Get CF scores
        if self.cf_recommender is not None:
            cf_recs = self.cf_recommender.recommend_user_based(
                user_id,
                n_recommendations=n_recommendations * 2
            )
            cf_scores = {movie_id: score for movie_id, score in cf_recs}
        
        # Get CB scores
        if self.cb_recommender is not None and user_liked_movies:
            cb_recs = self.cb_recommender.recommend_for_user(
                user_liked_movies,
                n_recommendations=n_recommendations * 2
            )
            cb_scores = {movie_id: score for movie_id, score in cb_recs}
        
        # Combine all movies
        all_movies = set(cf_scores.keys()) | set(cb_scores.keys())
        
        recommendations = []
        for movie_id in all_movies:
            cf_score = cf_scores.get(movie_id, 0)
            cb_score = cb_scores.get(movie_id, 0)
            
            # Combined score (can be more sophisticated)
            combined_score = (cf_score + cb_score) / 2
            
            # Boost if movie appears in both
            if cf_score > 0 and cb_score > 0:
                combined_score *= 1.2
            
            metadata = {
                'cf_score': cf_score,
                'cb_score': cb_score,
                'in_both': cf_score > 0 and cb_score > 0
            }
            
            recommendations.append((movie_id, combined_score, metadata))
        
        # Sort by combined score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_recommendations]
    
    def matrix_factorization_hybrid(
        self,
        user_id: str,
        n_recommendations: int = 10,
        user_item_matrix: Optional[pd.DataFrame] = None,
        user_liked_movies: Optional[List[str]] = None,
        mf_weight: float = 0.6,
        cb_weight: float = 0.4
    ) -> List[Tuple[str, float]]:
        """
        Combine matrix factorization with content-based filtering.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            user_item_matrix: User-item matrix
            user_liked_movies: List of movies user liked
            mf_weight: Weight for MF recommendations
            cb_weight: Weight for CB recommendations
            
        Returns:
            List of (movie_id, score) tuples
        """
        # Normalize weights
        total_weight = mf_weight + cb_weight
        mf_weight /= total_weight
        cb_weight /= total_weight
        
        combined_scores = {}
        
        # Get MF recommendations
        if self.mf_recommender is not None and user_item_matrix is not None:
            mf_recs = self.mf_recommender.recommend(
                user_id,
                user_item_matrix,
                n_recommendations=n_recommendations * 2
            )
            
            for movie_id, score in mf_recs:
                if movie_id not in combined_scores:
                    combined_scores[movie_id] = 0
                # Normalize MF scores to 0-1 range
                normalized_score = score / 10.0  # Assuming ratings are 0-10
                combined_scores[movie_id] += normalized_score * mf_weight
        
        # Get CB recommendations
        if self.cb_recommender is not None and user_liked_movies:
            cb_recs = self.cb_recommender.recommend_for_user(
                user_liked_movies,
                n_recommendations=n_recommendations * 2
            )
            
            for movie_id, score in cb_recs:
                if movie_id not in combined_scores:
                    combined_scores[movie_id] = 0
                combined_scores[movie_id] += score * cb_weight
        
        # Sort by combined score
        recommendations = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return recommendations[:n_recommendations]


if __name__ == "__main__":
    # Example usage
    print("Hybrid Recommender System")
    print("=========================\n")
    
    # Create sample data
    user_item_data = {
        'movie_1': [5, 3, 0, 1, 0],
        'movie_2': [4, 0, 0, 1, 0],
        'movie_3': [0, 1, 0, 5, 0],
        'movie_4': [0, 1, 0, 4, 0],
        'movie_5': [0, 0, 5, 0, 4],
    }
    
    user_item_matrix = pd.DataFrame(
        user_item_data,
        index=['user_1', 'user_2', 'user_3', 'user_4', 'user_5']
    )
    
    movies_data = {
        'imdb_id': ['movie_1', 'movie_2', 'movie_3', 'movie_4', 'movie_5'],
        'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
        'genre': ['Action', 'Action', 'Drama', 'Drama', 'Comedy'],
        'year': [2020, 2019, 2021, 2020, 2018],
        'imdb_rating': [8.5, 8.0, 7.5, 7.8, 8.2],
        'imdb_rating_normalized': [0.9, 0.8, 0.5, 0.6, 0.7]
    }
    
    movies_df = pd.DataFrame(movies_data)
    
    # Initialize recommenders
    cf_recommender = CollaborativeFilteringRecommender(user_item_matrix)
    cb_recommender = ContentBasedRecommender(movies_df)
    
    # Create hybrid recommender
    hybrid = HybridRecommender(
        cf_recommender=cf_recommender,
        cb_recommender=cb_recommender
    )
    
    user_id = 'user_1'
    user_liked_movies = ['movie_1', 'movie_2']
    
    # Weighted hybrid
    print("1. Weighted Hybrid Recommendations")
    print("-" * 40)
    recs = hybrid.weighted_hybrid(
        user_id,
        n_recommendations=3,
        cf_weight=0.6,
        cb_weight=0.4,
        user_liked_movies=user_liked_movies
    )
    print(f"Recommendations for {user_id}:")
    for movie_id, score in recs:
        print(f"  {movie_id}: {score:.3f}")
    
    # Switching hybrid
    print("\n2. Switching Hybrid Recommendations")
    print("-" * 40)
    recs = hybrid.switching_hybrid(
        user_id,
        n_recommendations=3,
        min_user_ratings=2,
        user_item_matrix=user_item_matrix,
        user_liked_movies=user_liked_movies
    )
    print(f"Recommendations for {user_id}:")
    for movie_id, score in recs:
        print(f"  {movie_id}: {score:.3f}")
    
    # Feature combination hybrid
    print("\n3. Feature Combination Hybrid")
    print("-" * 40)
    recs = hybrid.feature_combination_hybrid(
        user_id,
        n_recommendations=3,
        user_item_matrix=user_item_matrix,
        user_liked_movies=user_liked_movies
    )
    print(f"Detailed recommendations for {user_id}:")
    for movie_id, score, metadata in recs:
        print(f"  {movie_id}: {score:.3f}")
        print(f"    CF score: {metadata['cf_score']:.3f}, CB score: {metadata['cb_score']:.3f}")
        print(f"    In both: {metadata['in_both']}")
    
    print("\nHybrid recommender demo complete!")
