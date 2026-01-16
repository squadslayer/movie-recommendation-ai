"""
End-to-end pipeline for movie recommendation system using local datasets.
This script demonstrates the complete workflow without API dependencies.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import DatasetLoader
from src.preprocessing import MovieDataProcessor, create_sample_ratings
from src.recommenders.collaborative_filtering import CollaborativeFilteringRecommender, MatrixFactorizationRecommender
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.hybrid import HybridRecommender
from src.evaluation import RecommenderEvaluator
from src.explainability import RecommendationExplainer


class RecommendationPipeline:
    """Complete recommendation pipeline using local datasets."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize pipeline.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = data_dir
        self.loader = DatasetLoader(data_dir)
        self.processor = MovieDataProcessor()
        self.evaluator = RecommenderEvaluator()
        
        # Data containers
        self.movies_df = None
        self.ratings_df = None
        self.processed_movies = None
        self.tfidf_matrix = None
        self.user_item_matrix = None
        
        # Recommenders
        self.cf_recommender = None
        self.cb_recommender = None
        self.mf_recommender = None
        self.hybrid_recommender = None
        self.explainer = None
    
    def load_data(self, movies_file: str, ratings_file: str = None):
        """
        Load movie and rating data.
        
        Args:
            movies_file: Path to movies CSV file
            ratings_file: Optional path to ratings CSV file
        """
        print("=" * 60)
        print("STEP 1: Loading Data")
        print("=" * 60)
        
        # Load movies
        self.movies_df = self.loader.load_movies(movies_file)
        print(f"✓ Loaded {len(self.movies_df)} movies from {movies_file}")
        
        # Load or create ratings
        if ratings_file:
            try:
                self.ratings_df = self.loader.load_ratings(ratings_file)
                print(f"✓ Loaded {len(self.ratings_df)} ratings from {ratings_file}")
            except FileNotFoundError:
                print(f"⚠ Ratings file not found. Creating sample ratings...")
                self._create_sample_ratings()
        else:
            print("⚠ No ratings file specified. Creating sample ratings...")
            self._create_sample_ratings()
        
        print(f"\nDataset Summary:")
        print(f"  Movies: {len(self.movies_df)}")
        if self.ratings_df is not None:
            print(f"  Ratings: {len(self.ratings_df)}")
            print(f"  Users: {self.ratings_df['user_id'].nunique()}")
            print(f"  Avg ratings/user: {self.ratings_df.groupby('user_id').size().mean():.1f}")
        print()
    
    def _create_sample_ratings(self):
        """Create sample ratings if none exist."""
        movie_ids = self.movies_df.iloc[:, 0].tolist()  # First column should be movie ID
        self.ratings_df = create_sample_ratings(movie_ids, num_users=50)
        print(f"✓ Created {len(self.ratings_df)} sample ratings for 50 users")
    
    def preprocess_data(self):
        """Preprocess movie data and create features."""
        print("=" * 60)
        print("STEP 2: Preprocessing Data")
        print("=" * 60)
        
        # Clean movie data
        print("→ Cleaning movie data...")
        self.processed_movies = self.processor.clean_data(self.movies_df)
        print(f"✓ Cleaned {len(self.processed_movies)} movies")
        
        # Create content features
        print("→ Extracting content features...")
        self.processed_movies = self.processor.create_content_features(self.processed_movies)
        print(f"✓ Created content features")
        
        # Create TF-IDF matrix
        print("→ Creating TF-IDF matrix...")
        self.tfidf_matrix, _ = self.processor.create_tfidf_matrix(self.processed_movies)
        print(f"✓ TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        
        # Create user-item matrix
        if self.ratings_df is not None:
            print("→ Creating user-item matrix...")
            # Map movie IDs in ratings to match processed movies
            movie_id_col = self.processed_movies.columns[0]
            self.user_item_matrix = self.processor.create_user_item_matrix(
                self.ratings_df,
                item_col='movie_id'
            )
            print(f"✓ User-item matrix shape: {self.user_item_matrix.shape}")
        print()
    
    def build_recommenders(self):
        """Build all recommendation models."""
        print("=" * 60)
        print("STEP 3: Building Recommenders")
        print("=" * 60)
        
        # Content-based recommender
        print("→ Building content-based recommender...")
        self.cb_recommender = ContentBasedRecommender(
            self.processed_movies,
            self.tfidf_matrix
        )
        print("✓ Content-based recommender ready")
        
        # Collaborative filtering recommender (if we have ratings)
        if self.user_item_matrix is not None and len(self.user_item_matrix) > 0:
            print("→ Building collaborative filtering recommender...")
            self.cf_recommender = CollaborativeFilteringRecommender(self.user_item_matrix)
            self.cf_recommender.compute_user_similarity()
            self.cf_recommender.compute_item_similarity()
            print("✓ Collaborative filtering recommender ready")
            
            # Matrix factorization recommender
            print("→ Building matrix factorization recommender...")
            self.mf_recommender = MatrixFactorizationRecommender(n_factors=20)
            self.mf_recommender.fit(self.user_item_matrix)
            print("✓ Matrix factorization recommender ready")
            
            # Hybrid recommender
            print("→ Building hybrid recommender...")
            self.hybrid_recommender = HybridRecommender(
                cf_recommender=self.cf_recommender,
                cb_recommender=self.cb_recommender,
                mf_recommender=self.mf_recommender
            )
            print("✓ Hybrid recommender ready")
        
        # Explainer
        print("→ Building explainer...")
        self.explainer = RecommendationExplainer(self.processed_movies)
        print("✓ Explainer ready")
        print()
    
    def get_recommendations(
        self,
        user_id: str = None,
        movie_id: str = None,
        method: str = 'hybrid',
        n: int = 10
    ):
        """
        Get recommendations.
        
        Args:
            user_id: User ID for personalized recommendations
            movie_id: Movie ID for similar movie recommendations
            method: Method to use ('cf', 'cb', 'mf', 'hybrid')
            n: Number of recommendations
            
        Returns:
            List of recommendations with explanations
        """
        print("=" * 60)
        print(f"STEP 4: Getting Recommendations ({method.upper()})")
        print("=" * 60)
        
        if movie_id:
            # Get similar movies
            print(f"Finding movies similar to: {movie_id}\n")
            similar = self.cb_recommender.get_similar_movies(movie_id, n=n)
            
            print(f"Top {len(similar)} similar movies:")
            for i, (mid, score) in enumerate(similar, 1):
                info = self.cb_recommender.get_movie_info(mid)
                if info:
                    print(f"{i}. {info['title']} ({info['year']}) - Score: {score:.3f}")
            
            return similar
        
        elif user_id:
            # Get user recommendations
            if method == 'cf' and self.cf_recommender:
                recs = self.cf_recommender.recommend_user_based(user_id, n_recommendations=n)
            elif method == 'cb' and self.cb_recommender:
                # Get user's liked movies
                liked_movies = self._get_user_liked_movies(user_id)
                recs = self.cb_recommender.recommend_for_user(liked_movies, n_recommendations=n)
            elif method == 'mf' and self.mf_recommender:
                recs = self.mf_recommender.recommend(user_id, self.user_item_matrix, n_recommendations=n)
            elif method == 'hybrid' and self.hybrid_recommender:
                liked_movies = self._get_user_liked_movies(user_id)
                recs = self.hybrid_recommender.weighted_hybrid(
                    user_id,
                    n_recommendations=n,
                    user_liked_movies=liked_movies
                )
            else:
                print(f"⚠ Method '{method}' not available or no data for user recommendations")
                return []
            
            print(f"Top {len(recs)} recommendations for user '{user_id}':\n")
            for i, (movie_id, score) in enumerate(recs, 1):
                info = self.cb_recommender.get_movie_info(movie_id)
                if info:
                    print(f"{i}. {info['title']} ({info['year']}) - Score: {score:.3f}")
                    print(f"   Genre: {info['genre']} | Rating: {info['rating']}")
            
            return recs
        
        else:
            print("⚠ Please specify either user_id or movie_id")
            return []
    
    def _get_user_liked_movies(self, user_id: str, min_rating: float = 7.0):
        """Get movies user has rated highly."""
        if self.ratings_df is None or user_id not in self.user_item_matrix.index:
            return []
        
        user_ratings = self.user_item_matrix.loc[user_id]
        liked = user_ratings[user_ratings >= min_rating].index.tolist()
        return liked
    
    def explain_recommendation(self, user_id: str, movie_id: str, method: str = 'content'):
        """
        Explain why a movie was recommended.
        
        Args:
            user_id: User ID
            movie_id: Recommended movie ID
            method: Method used ('content', 'cf', 'hybrid')
        """
        print("=" * 60)
        print("STEP 5: Explaining Recommendation")
        print("=" * 60)
        
        movie_info = self.cb_recommender.get_movie_info(movie_id)
        if movie_info:
            print(f"\nMovie: {movie_info['title']} ({movie_info['year']})")
            print(f"Genre: {movie_info['genre']}")
            print(f"Rating: {movie_info['rating']}\n")
        
        if method == 'content':
            liked_movies = self._get_user_liked_movies(user_id)
            explanation = self.explainer.explain_content_based(movie_id, liked_movies)
            print(f"Explanation: {explanation['text']}")
            
            if explanation.get('matching_features'):
                print(f"\nMatching Features:")
                for feature, values in explanation['matching_features'].items():
                    print(f"  {feature}: {values}")
        
        elif method == 'cf' and self.cf_recommender:
            similar_users = self.cf_recommender.get_similar_users(user_id, n=5)
            explanation = self.explainer.explain_collaborative_filtering(
                user_id, movie_id, similar_users, self.user_item_matrix
            )
            print(f"Explanation: {explanation['text']}")
        
        print()
    
    def evaluate_recommender(self, user_id: str, method: str = 'hybrid', k: int = 10):
        """
        Evaluate recommendation quality.
        
        Args:
            user_id: User ID to evaluate for
            method: Method to evaluate
            k: Number of recommendations to evaluate
        """
        print("=" * 60)
        print(f"STEP 6: Evaluating Recommender")
        print("=" * 60)
        
        # Get recommendations
        recs = self.get_recommendations(user_id, method=method, n=k)
        recommended_ids = [movie_id for movie_id, _ in recs]
        
        # Get ground truth (highly rated movies)
        if user_id in self.user_item_matrix.index:
            user_ratings = self.user_item_matrix.loc[user_id]
            relevant_ids = user_ratings[user_ratings >= 8].index.tolist()
            
            # Evaluate
            results = self.evaluator.evaluate_recommendations(
                recommended_ids,
                relevant_ids,
                k_values=[5, 10]
            )
            
            print(f"\nEvaluation Results:")
            for metric, score in results.items():
                print(f"  {metric}: {score:.4f}")
        else:
            print(f"⚠ User '{user_id}' not found in ratings data")
        
        print()


def main():
    """Run the complete pipeline."""
    print("\n" + "=" * 60)
    print("MOVIE RECOMMENDATION SYSTEM - END-TO-END PIPELINE")
    print("=" * 60 + "\n")
    
    # Initialize pipeline
    pipeline = RecommendationPipeline(data_dir="data")
    
    # Load data (adjust filenames based on your actual data files)
    try:
        pipeline.load_data(
            movies_file="movies.csv",      # Your movie data file
            ratings_file="ratings.csv"     # Your ratings data file (optional)
        )
    except FileNotFoundError as e:
        print(f"\n⚠ Data file not found: {e}")
        print("\nCreating sample dataset for demonstration...\n")
        
        # Create sample data
        pipeline.loader.create_sample_dataset(n_movies=100)
        pipeline.loader.create_sample_ratings(
            movie_ids=[f'tt{str(i).zfill(7)}' for i in range(1, 101)],
            n_users=50
        )
        
        # Load the sample data
        pipeline.load_data(
            movies_file="sample_movies.csv",
            ratings_file="sample_ratings.csv"
        )
    
    # Preprocess
    pipeline.preprocess_data()
    
    # Build recommenders
    pipeline.build_recommenders()
    
    # Example recommendations
    if pipeline.user_item_matrix is not None and len(pipeline.user_item_matrix) > 0:
        user_id = pipeline.user_item_matrix.index[0]
        
        # Get hybrid recommendations
        pipeline.get_recommendations(user_id=user_id, method='hybrid', n=5)
        
        # Get recommendations and explain first one
        recs = pipeline.get_recommendations(user_id=user_id, method='content', n=5)
        if recs:
            pipeline.explain_recommendation(user_id, recs[0][0], method='content')
        
        # Evaluate
        pipeline.evaluate_recommender(user_id, method='hybrid', k=10)
    
    print("=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print("\nYou can now use the pipeline object to:")
    print("  - Get recommendations: pipeline.get_recommendations(user_id='user_1')")
    print("  - Find similar movies: pipeline.get_recommendations(movie_id='tt0000001')")
    print("  - Explain recommendations: pipeline.explain_recommendation(user_id, movie_id)")
    print()


if __name__ == "__main__":
    main()
