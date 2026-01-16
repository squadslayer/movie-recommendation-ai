"""
Data loader utility for working with local movie datasets.
This module provides utilities to load and prepare movie data from CSV files
without requiring API calls.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import os


class DatasetLoader:
    """Load and manage movie datasets from local files."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize dataset loader.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
    
    def load_movies(self, filepath: str, required_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load movie data from CSV file.
        
        Args:
            filepath: Path to CSV file (relative to data_dir or absolute)
            required_columns: List of required column names to validate
            
        Returns:
            DataFrame with movie data
        """
        # Handle both relative and absolute paths
        if not Path(filepath).is_absolute():
            filepath = self.data_dir / filepath
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Movie data file not found: {filepath}")
        
        # Load data
        df = pd.read_csv(filepath)
        
        # Validate required columns
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    
    def load_ratings(self, filepath: str) -> pd.DataFrame:
        """
        Load user ratings from CSV file.
        
        Args:
            filepath: Path to ratings CSV file
            
        Returns:
            DataFrame with user ratings
        """
        if not Path(filepath).is_absolute():
            filepath = self.data_dir / filepath
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Ratings file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Validate required columns
        required = ['user_id', 'movie_id', 'rating']
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Ratings file missing required columns: {missing}")
        
        return df
    
    def create_sample_dataset(self, output_file: str = "sample_movies.csv", n_movies: int = 50):
        """
        Create a sample movie dataset for testing.
        
        Args:
            output_file: Output filename
            n_movies: Number of sample movies to create
        """
        # Sample movie data
        sample_movies = {
            'imdb_id': [f'tt{str(i).zfill(7)}' for i in range(1, n_movies + 1)],
            'title': [f'Sample Movie {i}' for i in range(1, n_movies + 1)],
            'year': np.random.randint(1970, 2024, n_movies),
            'genre': np.random.choice(
                ['Action', 'Comedy', 'Drama', 'Thriller', 'Sci-Fi', 'Romance', 
                 'Action, Drama', 'Comedy, Romance', 'Thriller, Crime'],
                n_movies
            ),
            'imdb_rating': np.round(np.random.uniform(5.0, 9.5, n_movies), 1),
            'metascore': np.random.randint(40, 100, n_movies),
            'runtime': [f'{np.random.randint(80, 180)} min' for _ in range(n_movies)],
            'director': [f'Director {np.random.randint(1, 20)}' for _ in range(n_movies)],
            'actors': [f'Actor {np.random.randint(1, 50)}, Actor {np.random.randint(1, 50)}' 
                      for _ in range(n_movies)],
            'plot': [f'A {genre} movie about interesting events.' 
                    for genre in np.random.choice(['thrilling', 'dramatic', 'funny', 'action-packed'], n_movies)],
            'language': ['English'] * n_movies,
            'country': ['USA'] * n_movies,
            'imdb_votes': np.random.randint(1000, 100000, n_movies)
        }
        
        df = pd.DataFrame(sample_movies)
        output_path = self.data_dir / output_file
        df.to_csv(output_path, index=False)
        print(f"Created sample dataset with {n_movies} movies: {output_path}")
        
        return df
    
    def create_sample_ratings(
        self, 
        movie_ids: List[str], 
        output_file: str = "sample_ratings.csv",
        n_users: int = 100,
        min_ratings: int = 5,
        max_ratings: int = 30
    ):
        """
        Create sample user ratings for testing.
        
        Args:
            movie_ids: List of movie IDs
            output_file: Output filename
            n_users: Number of users to simulate
            min_ratings: Minimum ratings per user
            max_ratings: Maximum ratings per user
        """
        np.random.seed(42)
        
        ratings = []
        for user_id in range(1, n_users + 1):
            n_ratings = np.random.randint(min_ratings, min(max_ratings, len(movie_ids)))
            rated_movies = np.random.choice(movie_ids, size=n_ratings, replace=False)
            
            for movie_id in rated_movies:
                rating = np.random.randint(1, 11)  # 1-10 scale
                ratings.append({
                    'user_id': f'user_{user_id}',
                    'movie_id': movie_id,
                    'rating': rating
                })
        
        df = pd.DataFrame(ratings)
        output_path = self.data_dir / output_file
        df.to_csv(output_path, index=False)
        print(f"Created sample ratings with {len(ratings)} ratings for {n_users} users: {output_path}")
        
        return df
    
    def prepare_movieslens_format(self, movies_file: str, ratings_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare MovieLens-format datasets.
        
        Args:
            movies_file: Path to movies.csv
            ratings_file: Path to ratings.csv
            
        Returns:
            Tuple of (movies_df, ratings_df)
        """
        movies_df = self.load_movies(movies_file)
        ratings_df = self.load_ratings(ratings_file)
        
        print(f"Loaded {len(movies_df)} movies and {len(ratings_df)} ratings")
        print(f"Users: {ratings_df['user_id'].nunique()}")
        print(f"Average ratings per user: {ratings_df.groupby('user_id').size().mean():.1f}")
        
        return movies_df, ratings_df
    
    def download_movielens_small(self):
        """
        Download and extract MovieLens small dataset.
        Note: This requires internet connection.
        """
        try:
            import requests
            import zipfile
            from io import BytesIO
            
            print("Downloading MovieLens 100K dataset...")
            url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
            
            response = requests.get(url)
            response.raise_for_status()
            
            # Extract zip file
            with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
                zip_file.extractall(self.data_dir)
            
            print(f"Downloaded and extracted MovieLens dataset to {self.data_dir}")
            print("Files available: movies.csv, ratings.csv, tags.csv, links.csv")
            
        except ImportError:
            print("Error: 'requests' library not found. Install with: pip install requests")
        except Exception as e:
            print(f"Error downloading dataset: {e}")


def load_dataset_pipeline(
    movies_file: str = "movies.csv",
    ratings_file: str = "ratings.csv",
    data_dir: str = "data"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load movies and ratings.
    
    Args:
        movies_file: Movies CSV filename
        ratings_file: Ratings CSV filename
        data_dir: Data directory
        
    Returns:
        Tuple of (movies_df, ratings_df)
    """
    loader = DatasetLoader(data_dir)
    return loader.prepare_movieslens_format(movies_file, ratings_file)


if __name__ == "__main__":
    # Example usage
    print("Dataset Loader Utility")
    print("=====================\n")
    
    loader = DatasetLoader("data")
    
    # Create sample datasets for testing
    print("1. Creating sample movie dataset...")
    movies_df = loader.create_sample_dataset(n_movies=100)
    print(f"   Created {len(movies_df)} movies\n")
    
    print("2. Creating sample ratings dataset...")
    ratings_df = loader.create_sample_ratings(
        movie_ids=movies_df['imdb_id'].tolist(),
        n_users=50
    )
    print(f"   Created {len(ratings_df)} ratings\n")
    
    print("3. Loading datasets...")
    movies_df = loader.load_movies("sample_movies.csv")
    ratings_df = loader.load_ratings("sample_ratings.csv")
    
    print(f"\nMovies dataset shape: {movies_df.shape}")
    print(f"Ratings dataset shape: {ratings_df.shape}")
    print(f"\nSample movies:")
    print(movies_df[['title', 'year', 'genre', 'imdb_rating']].head())
    
    print("\nDataset loader ready!")
