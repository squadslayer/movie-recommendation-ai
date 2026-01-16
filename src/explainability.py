"""
Explainability module for recommendation systems.
Provides explanations for why certain movies were recommended.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class RecommendationExplainer:
    """Generate explanations for movie recommendations."""
    
    def __init__(self, movies_df: pd.DataFrame):
        """
        Initialize explainer.
        
        Args:
            movies_df: DataFrame containing movie information
        """
        self.movies_df = movies_df
        
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
    
    def explain_collaborative_filtering(
        self,
        user_id: str,
        recommended_movie_id: str,
        similar_users: List[Tuple[str, float]],
        user_item_matrix: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Explain collaborative filtering recommendation.
        
        Args:
            user_id: User ID
            recommended_movie_id: Recommended movie ID
            similar_users: List of (user_id, similarity_score) tuples
            user_item_matrix: User-item rating matrix
            
        Returns:
            Dictionary with explanation details
        """
        explanation = {
            'method': 'Collaborative Filtering',
            'reason': 'Similar users also liked this movie',
            'similar_users': [],
            'average_rating': 0.0
        }
        
        # Get ratings from similar users who rated this movie
        ratings = []
        for similar_user_id, similarity in similar_users[:5]:
            if similar_user_id in user_item_matrix.index and \
               recommended_movie_id in user_item_matrix.columns:
                rating = user_item_matrix.loc[similar_user_id, recommended_movie_id]
                if rating > 0:
                    ratings.append(rating)
                    explanation['similar_users'].append({
                        'user_id': similar_user_id,
                        'similarity': similarity,
                        'rating': rating
                    })
        
        if ratings:
            explanation['average_rating'] = np.mean(ratings)
        
        # Generate human-readable explanation
        if explanation['similar_users']:
            num_users = len(explanation['similar_users'])
            avg_rating = explanation['average_rating']
            explanation['text'] = (
                f"This movie is recommended because {num_users} users similar to you "
                f"rated it highly (average rating: {avg_rating:.1f}/10). "
                f"These users have similar taste to yours based on their past ratings."
            )
        else:
            explanation['text'] = "This movie is recommended based on collaborative filtering."
        
        return explanation
    
    def explain_item_based_cf(
        self,
        user_id: str,
        recommended_movie_id: str,
        similar_movies: List[Tuple[str, float]],
        user_item_matrix: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Explain item-based collaborative filtering recommendation.
        
        Args:
            user_id: User ID
            recommended_movie_id: Recommended movie ID
            similar_movies: List of (movie_id, similarity_score) tuples
            user_item_matrix: User-item rating matrix
            
        Returns:
            Dictionary with explanation details
        """
        explanation = {
            'method': 'Item-Based Collaborative Filtering',
            'reason': 'Similar to movies you liked',
            'similar_movies': []
        }
        
        # Get movies user has rated highly
        if user_id in user_item_matrix.index:
            user_ratings = user_item_matrix.loc[user_id]
            
            # Find which similar movies the user rated
            for movie_id, similarity in similar_movies[:5]:
                if movie_id in user_ratings.index and user_ratings[movie_id] > 0:
                    movie_title = self._get_movie_title(movie_id)
                    explanation['similar_movies'].append({
                        'movie_id': movie_id,
                        'title': movie_title,
                        'similarity': similarity,
                        'your_rating': user_ratings[movie_id]
                    })
        
        # Generate human-readable explanation
        if explanation['similar_movies']:
            movie_titles = [m['title'] for m in explanation['similar_movies'][:3]]
            explanation['text'] = (
                f"This movie is recommended because it's similar to movies you liked, "
                f"such as: {', '.join(movie_titles)}."
            )
        else:
            explanation['text'] = "This movie is recommended based on similar movies."
        
        return explanation
    
    def explain_content_based(
        self,
        recommended_movie_id: str,
        user_liked_movies: List[str],
        similarity_scores: Optional[List[Tuple[str, float]]] = None
    ) -> Dict[str, any]:
        """
        Explain content-based recommendation.
        
        Args:
            recommended_movie_id: Recommended movie ID
            user_liked_movies: List of movie IDs user liked
            similarity_scores: Optional list of (movie_id, score) for liked movies
            
        Returns:
            Dictionary with explanation details
        """
        recommended_movie = self._get_movie_info(recommended_movie_id)
        
        explanation = {
            'method': 'Content-Based Filtering',
            'reason': 'Shares features with movies you liked',
            'recommended_movie': recommended_movie,
            'matching_features': {},
            'similar_to': []
        }
        
        if not recommended_movie:
            explanation['text'] = "This movie is recommended based on content similarity."
            return explanation
        
        # Extract features
        rec_genres = set(recommended_movie.get('genre', '').split(', '))
        rec_director = recommended_movie.get('director', '')
        rec_actors = set(recommended_movie.get('actors', '').split(', '))
        
        # Compare with liked movies
        matching_genres = set()
        matching_directors = []
        matching_actors = set()
        
        for liked_movie_id in user_liked_movies[:5]:
            liked_movie = self._get_movie_info(liked_movie_id)
            if not liked_movie:
                continue
            
            # Check genres
            liked_genres = set(liked_movie.get('genre', '').split(', '))
            common_genres = rec_genres & liked_genres
            matching_genres.update(common_genres)
            
            # Check director
            if rec_director and rec_director == liked_movie.get('director', ''):
                matching_directors.append(rec_director)
            
            # Check actors
            liked_actors = set(liked_movie.get('actors', '').split(', '))
            common_actors = rec_actors & liked_actors
            matching_actors.update(common_actors)
            
            # Track which movie it's similar to
            if similarity_scores:
                for movie_id, score in similarity_scores:
                    if movie_id == liked_movie_id:
                        explanation['similar_to'].append({
                            'movie_id': liked_movie_id,
                            'title': liked_movie.get('title', 'Unknown'),
                            'similarity': score
                        })
        
        # Store matching features
        if matching_genres:
            explanation['matching_features']['genres'] = list(matching_genres)
        if matching_directors:
            explanation['matching_features']['director'] = matching_directors[0]
        if matching_actors:
            explanation['matching_features']['actors'] = list(matching_actors)[:3]
        
        # Generate human-readable explanation
        reasons = []
        if matching_genres:
            genres_str = ', '.join(list(matching_genres)[:2])
            reasons.append(f"shares genres ({genres_str})")
        if matching_directors:
            reasons.append(f"same director ({matching_directors[0]})")
        if matching_actors:
            actors_str = ', '.join(list(matching_actors)[:2])
            reasons.append(f"common actors ({actors_str})")
        
        if reasons:
            explanation['text'] = (
                f"This movie is recommended because it {' and '.join(reasons)} "
                f"with movies you enjoyed."
            )
        else:
            explanation['text'] = "This movie has similar content to movies you liked."
        
        return explanation
    
    def explain_hybrid(
        self,
        recommended_movie_id: str,
        cf_score: float,
        cb_score: float,
        cf_explanation: Optional[Dict] = None,
        cb_explanation: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Explain hybrid recommendation.
        
        Args:
            recommended_movie_id: Recommended movie ID
            cf_score: Collaborative filtering score
            cb_score: Content-based score
            cf_explanation: Optional CF explanation
            cb_explanation: Optional CB explanation
            
        Returns:
            Dictionary with explanation details
        """
        explanation = {
            'method': 'Hybrid Recommendation',
            'cf_score': cf_score,
            'cb_score': cb_score,
            'primary_reason': ''
        }
        
        # Determine primary reason
        if cf_score > cb_score:
            explanation['primary_reason'] = 'collaborative_filtering'
            primary_text = "similar users also enjoyed it"
            if cf_explanation:
                primary_text = cf_explanation.get('text', primary_text)
        else:
            explanation['primary_reason'] = 'content_based'
            primary_text = "it matches your content preferences"
            if cb_explanation:
                primary_text = cb_explanation.get('text', primary_text)
        
        # Generate combined explanation
        if cf_score > 0 and cb_score > 0:
            explanation['text'] = (
                f"This movie is recommended for multiple reasons: "
                f"{primary_text}. Additionally, it scores well on both "
                f"user similarity ({cf_score:.2f}) and content similarity ({cb_score:.2f})."
            )
        else:
            explanation['text'] = f"This movie is recommended because {primary_text}."
        
        return explanation
    
    def get_feature_importance(
        self,
        recommended_movie_id: str,
        user_liked_movies: List[str]
    ) -> Dict[str, float]:
        """
        Calculate feature importance for recommendation.
        
        Args:
            recommended_movie_id: Recommended movie ID
            user_liked_movies: List of movie IDs user liked
            
        Returns:
            Dictionary of feature importance scores
        """
        recommended_movie = self._get_movie_info(recommended_movie_id)
        
        if not recommended_movie:
            return {}
        
        importance = {
            'genre': 0.0,
            'director': 0.0,
            'actors': 0.0,
            'year': 0.0,
            'rating': 0.0
        }
        
        rec_genres = set(recommended_movie.get('genre', '').split(', '))
        rec_director = recommended_movie.get('director', '')
        rec_actors = set(recommended_movie.get('actors', '').split(', '))
        rec_year = recommended_movie.get('year', 0)
        
        num_liked = len(user_liked_movies)
        
        for liked_movie_id in user_liked_movies:
            liked_movie = self._get_movie_info(liked_movie_id)
            if not liked_movie:
                continue
            
            # Genre overlap
            liked_genres = set(liked_movie.get('genre', '').split(', '))
            if rec_genres & liked_genres:
                importance['genre'] += 1.0 / num_liked
            
            # Director match
            if rec_director and rec_director == liked_movie.get('director', ''):
                importance['director'] += 1.0 / num_liked
            
            # Actor overlap
            liked_actors = set(liked_movie.get('actors', '').split(', '))
            if rec_actors & liked_actors:
                importance['actors'] += 1.0 / num_liked
            
            # Year similarity (within 5 years)
            liked_year = liked_movie.get('year', 0)
            if rec_year and liked_year and abs(rec_year - liked_year) <= 5:
                importance['year'] += 1.0 / num_liked
        
        return importance
    
    def visualize_explanation(
        self,
        recommended_movie_id: str,
        user_liked_movies: List[str]
    ):
        """
        Visualize recommendation explanation.
        
        Args:
            recommended_movie_id: Recommended movie ID
            user_liked_movies: List of movie IDs user liked
        """
        importance = self.get_feature_importance(recommended_movie_id, user_liked_movies)
        
        if not importance:
            print("No feature importance data available")
            return
        
        plt.figure(figsize=(10, 6))
        
        features = list(importance.keys())
        scores = list(importance.values())
        
        plt.barh(features, scores, color='steelblue', edgecolor='navy')
        plt.xlabel('Importance Score')
        plt.title(f'Feature Importance for Recommendation\n{self._get_movie_title(recommended_movie_id)}')
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.show()
    
    def _get_movie_info(self, movie_id: str) -> Optional[Dict]:
        """Get movie information by ID."""
        if movie_id not in self.movie_to_idx:
            return None
        
        idx = self.movie_to_idx[movie_id]
        movie_row = self.movies_df.iloc[idx]
        
        return {
            'movie_id': movie_id,
            'title': movie_row.get('title', 'Unknown'),
            'genre': movie_row.get('genre', ''),
            'year': movie_row.get('year', 0),
            'director': movie_row.get('director', ''),
            'actors': movie_row.get('actors', ''),
            'rating': movie_row.get('imdb_rating', 0)
        }
    
    def _get_movie_title(self, movie_id: str) -> str:
        """Get movie title by ID."""
        info = self._get_movie_info(movie_id)
        return info['title'] if info else 'Unknown Movie'


if __name__ == "__main__":
    # Example usage
    print("Recommendation Explainability")
    print("=============================\n")
    
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
        ]
    }
    
    movies_df = pd.DataFrame(movies_data)
    explainer = RecommendationExplainer(movies_df)
    
    # Example 1: Content-based explanation
    print("1. Content-Based Recommendation Explanation")
    print("-" * 40)
    recommended_id = 'tt0137523'  # Fight Club
    user_liked = ['tt0111161', 'tt0109830']  # Shawshank, Forrest Gump
    
    explanation = explainer.explain_content_based(recommended_id, user_liked)
    print(f"Recommended: {explanation['recommended_movie']['title']}")
    print(f"Explanation: {explanation['text']}")
    if explanation['matching_features']:
        print(f"Matching features: {explanation['matching_features']}")
    
    # Example 2: Feature importance
    print("\n2. Feature Importance")
    print("-" * 40)
    importance = explainer.get_feature_importance(recommended_id, user_liked)
    print(f"Feature importance for '{explainer._get_movie_title(recommended_id)}':")
    for feature, score in importance.items():
        print(f"  {feature}: {score:.2f}")
    
    # Example 3: Hybrid explanation
    print("\n3. Hybrid Recommendation Explanation")
    print("-" * 40)
    hybrid_explanation = explainer.explain_hybrid(
        recommended_id,
        cf_score=0.75,
        cb_score=0.82
    )
    print(f"Method: {hybrid_explanation['method']}")
    print(f"CF Score: {hybrid_explanation['cf_score']:.2f}")
    print(f"CB Score: {hybrid_explanation['cb_score']:.2f}")
    print(f"Explanation: {hybrid_explanation['text']}")
    
    print("\nExplainability demo complete!")
