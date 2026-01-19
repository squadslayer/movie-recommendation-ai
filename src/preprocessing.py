"""
Data preprocessing module for movie recommendation system.
Handles data cleaning, feature extraction, and preparation for recommendation algorithms.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional
import re


class MovieDataProcessor:
    """Handles preprocessing of movie data for recommendation algorithms."""
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the movie dataset.
        
        Args:
            df: Raw movie DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Detect ID column (handle both 'id' and 'imdb_id')
        id_column = None
        if 'imdb_id' in df_clean.columns:
            id_column = 'imdb_id'
        elif 'id' in df_clean.columns:
            id_column = 'id'
        elif 'movie_id' in df_clean.columns:
            id_column = 'movie_id'
        
        # Remove duplicates based on ID column if found
        if id_column:
            df_clean = df_clean.drop_duplicates(subset=[id_column], keep='first')
        
        # Handle missing values in critical columns
        if 'title' in df_clean.columns:
            df_clean['title'] = df_clean['title'].fillna('Unknown')
        if 'genre' in df_clean.columns:
            df_clean['genre'] = df_clean['genre'].fillna('Unknown')
        elif 'genres' in df_clean.columns:
            # TMDB uses 'genres' instead of 'genre'
            df_clean['genre'] = df_clean['genres'].fillna('Unknown')
        
        # Handle plot/overview
        if 'plot' in df_clean.columns:
            df_clean['plot'] = df_clean['plot'].fillna('')
        elif 'overview' in df_clean.columns:
            # TMDB uses 'overview' instead of 'plot'
            df_clean['plot'] = df_clean['overview'].fillna('')
        else:
            df_clean['plot'] = ''
        
        if 'director' in df_clean.columns:
            df_clean['director'] = df_clean['director'].fillna('Unknown')
        
        # Handle actors (map from 'cast' if present)
        if 'cast' in df_clean.columns:
            df_clean['actors'] = df_clean['cast'].fillna('Unknown')
        elif 'actors' in df_clean.columns:
            df_clean['actors'] = df_clean['actors'].fillna('Unknown')
        
        # Clean and convert year
        if 'year' in df_clean.columns:
            df_clean['year'] = df_clean['year'].apply(self._extract_year)
        elif 'release_date' in df_clean.columns:
            # TMDB uses 'release_date' instead of 'year'
            df_clean['year'] = df_clean['release_date'].apply(self._extract_year)
        
        # Clean and convert ratings
        if 'imdb_rating' in df_clean.columns:
            df_clean['imdb_rating'] = pd.to_numeric(df_clean['imdb_rating'], errors='coerce')
        elif 'vote_average' in df_clean.columns:
            # TMDB uses 'vote_average' instead of 'imdb_rating'
            df_clean['imdb_rating'] = pd.to_numeric(df_clean['vote_average'], errors='coerce')
        
        if 'metascore' in df_clean.columns:
            df_clean['metascore'] = pd.to_numeric(df_clean['metascore'], errors='coerce')
        
        # Fill missing ratings with median (safely)
        if 'imdb_rating' in df_clean.columns and not df_clean['imdb_rating'].isna().all():
            df_clean['imdb_rating'] = df_clean['imdb_rating'].fillna(df_clean['imdb_rating'].median())
        elif 'imdb_rating' in df_clean.columns:
             df_clean['imdb_rating'] = df_clean['imdb_rating'].fillna(0) # Fallback for all-NaN

        if 'metascore' in df_clean.columns and not df_clean['metascore'].isna().all():
            df_clean['metascore'] = df_clean['metascore'].fillna(df_clean['metascore'].median())
        
        # Clean runtime (optional column)
        if 'runtime' in df_clean.columns:
            df_clean['runtime_minutes'] = df_clean['runtime'].apply(self._extract_runtime)
        else:
            df_clean['runtime_minutes'] = None # Will be handled safely in normalization
        
        # Clean votes (optional column)
        if 'imdb_votes' in df_clean.columns:
            df_clean['imdb_votes'] = df_clean['imdb_votes'].apply(self._clean_votes)
        elif 'vote_count' in df_clean.columns:
            # TMDB uses 'vote_count'
            df_clean['imdb_votes'] = pd.to_numeric(df_clean['vote_count'], errors='coerce').fillna(0).astype(int)
        
        return df_clean
    
    def _extract_year(self, year_str) -> Optional[int]:
        """Extract year from year string (handles ranges like '2010–2015')."""
        if pd.isna(year_str):
            return None
        
        year_str = str(year_str)
        match = re.search(r'\d{4}', year_str)
        if match:
            return int(match.group())
        return None
    
    def _extract_runtime(self, runtime_str) -> Optional[int]:
        """Extract runtime in minutes from runtime string."""
        if pd.isna(runtime_str):
            return None
        
        runtime_str = str(runtime_str)
        match = re.search(r'(\d+)', runtime_str)
        if match:
            return int(match.group(1))
        return None
    
    def _clean_votes(self, votes_str) -> int:
        """Clean vote count (remove commas)."""
        if pd.isna(votes_str):
            return 0
        
        votes_str = str(votes_str).replace(',', '')
        try:
            return int(votes_str)
        except ValueError:
            return 0
    
    def extract_genres(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and one-hot encode genres.
        
        Args:
            df: DataFrame with 'genre' column
            
        Returns:
            DataFrame with one-hot encoded genre columns
        """
        # Check if genre column exists
        if 'genre' not in df.columns:
            return df
        
        # Split genres and create one-hot encoding
        all_genres = set()
        for genres in df['genre'].dropna():
            all_genres.update([g.strip() for g in str(genres).split(',')])
        
        # Remove 'Unknown' and 'N/A'
        all_genres = {g for g in all_genres if g not in ['Unknown', 'N/A', '']}
        
        # Create binary columns for each genre
        for genre in sorted(all_genres):
            df[f'genre_{genre}'] = df['genre'].apply(
                lambda x: 1 if genre in str(x) else 0
            )
        
        return df
    
    def create_content_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for content-based filtering.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with additional content features
        """
        df_features = df.copy()
        
        # Combine text features for TF-IDF
        text_parts = []
        
        if 'plot' in df_features.columns:
            text_parts.append(df_features['plot'].fillna(''))
        if 'genre' in df_features.columns:
            text_parts.append(df_features['genre'].fillna(''))
        if 'director' in df_features.columns:
            text_parts.append(df_features['director'].fillna(''))
        if 'actors' in df_features.columns:
            text_parts.append(df_features['actors'].fillna(''))
        
        # Combine available text fields
        if text_parts:
            df_features['combined_text'] = text_parts[0]
            for part in text_parts[1:]:
                df_features['combined_text'] = df_features['combined_text'] + ' ' + part
        else:
            df_features['combined_text'] = ''
        
        # Extract genre features (if genre column exists)
        df_features = self.extract_genres(df_features)
        
        # Normalize numerical features
        numerical_features = ['imdb_rating', 'metascore', 'runtime_minutes', 'year']
        available_features = []
        
        # Verify columns exist and are not all-NaN before attempting normalization
        for f in numerical_features:
            if f in df_features.columns:
                # Check if column has valid data (not all NaN)
                if not df_features[f].isna().all():
                     available_features.append(f)
        
        if available_features:
            # Fill NaNs with median for available features
            df_features[available_features] = df_features[available_features].fillna(
                df_features[available_features].median()
            )
            
            # Normalize
            df_features[[f'{f}_normalized' for f in available_features]] = self.scaler.fit_transform(
                df_features[available_features]
            )
        
        return df_features
    
    def create_tfidf_matrix(self, df: pd.DataFrame, text_column: str = 'combined_text'):
        """
        Create TF-IDF matrix from text features.
        
        Args:
            df: DataFrame with text column
            text_column: Name of the column containing text
            
        Returns:
            TF-IDF matrix and feature names
        """
        texts = df[text_column].fillna('')
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        return tfidf_matrix, feature_names
    
    def create_user_item_matrix(
        self,
        ratings_df: pd.DataFrame,
        user_col: str = 'user_id',
        item_col: str = 'movie_id',
        rating_col: str = 'rating'
    ) -> pd.DataFrame:
        """
        Create user-item matrix for collaborative filtering.
        
        Args:
            ratings_df: DataFrame with user ratings
            user_col: Name of user ID column
            item_col: Name of movie ID column
            rating_col: Name of rating column
            
        Returns:
            User-item matrix (users as rows, movies as columns)
        """
        user_item_matrix = ratings_df.pivot_table(
            index=user_col,
            columns=item_col,
            values=rating_col,
            fill_value=0
        )
        
        return user_item_matrix
    
    def normalize_ratings(self, user_item_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize ratings by subtracting user mean.
        
        Args:
            user_item_matrix: User-item matrix
            
        Returns:
            Normalized user-item matrix
        """
        # Calculate user means (only for non-zero ratings)
        user_means = user_item_matrix.replace(0, np.nan).mean(axis=1)
        
        # Normalize
        normalized_matrix = user_item_matrix.sub(user_means, axis=0)
        normalized_matrix = normalized_matrix.fillna(0)
        
        return normalized_matrix
    
    def split_train_test(
        self,
        ratings_df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split ratings data into train and test sets.
        
        Args:
            ratings_df: DataFrame with user ratings
            test_size: Proportion of test data
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, test_df)
        """
        train_df, test_df = train_test_split(
            ratings_df,
            test_size=test_size,
            random_state=random_state
        )
        
        return train_df, test_df


def create_sample_ratings(movie_ids: List[str], num_users: int = 100) -> pd.DataFrame:
    """
    Create sample user ratings for testing.
    
    Args:
        movie_ids: List of movie IDs
        num_users: Number of users to simulate
        
    Returns:
        DataFrame with sample ratings
    """
    np.random.seed(42)
    
    ratings = []
    for user_id in range(num_users):
        # Each user rates 10-30 random movies
        num_ratings = np.random.randint(10, min(30, len(movie_ids)))
        rated_movies = np.random.choice(movie_ids, size=num_ratings, replace=False)
        
        for movie_id in rated_movies:
            # Ratings between 1-10 (following IMDb scale)
            rating = np.random.randint(1, 11)
            ratings.append({
                'user_id': f'user_{user_id}',
                'movie_id': movie_id,
                'rating': rating
            })
    
    return pd.DataFrame(ratings)


def load_and_preprocess_movies(csv_path: str) -> Tuple[pd.DataFrame, any, any]:
    """
    Load and preprocess movie data from CSV.
    
    Args:
        csv_path: Path to movie CSV file
        
    Returns:
        Tuple of (processed_df, tfidf_matrix, processor)
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Initialize processor
    processor = MovieDataProcessor()
    
    # Clean data
    df_clean = processor.clean_data(df)
    
    # Create content features
    df_processed = processor.create_content_features(df_clean)
    
    # Create TF-IDF matrix
    tfidf_matrix, _ = processor.create_tfidf_matrix(df_processed)
    
    return df_processed, tfidf_matrix, processor


if __name__ == "__main__":
    # Example usage
    print("Movie Data Preprocessing Module")
    print("================================")
    
    # Create sample data
    sample_data = {
        'imdb_id': ['tt0111161', 'tt0068646', 'tt0468569'],
        'title': ['The Shawshank Redemption', 'The Godfather', 'The Dark Knight'],
        'year': ['1994', '1972', '2008'],
        'genre': ['Drama', 'Crime, Drama', 'Action, Crime, Drama'],
        'imdb_rating': ['9.3', '9.2', '9.0'],
        'metascore': ['80', '100', '84'],
        'runtime': ['142 min', '175 min', '152 min'],
        'plot': [
            'Two imprisoned men bond over a number of years',
            'The aging patriarch of an organized crime dynasty',
            'When the menace known as the Joker wreaks havoc'
        ],
        'director': ['Frank Darabont', 'Francis Ford Coppola', 'Christopher Nolan'],
        'actors': ['Tim Robbins, Morgan Freeman', 'Marlon Brando, Al Pacino', 'Christian Bale, Heath Ledger'],
        'imdb_votes': ['2,500,000', '1,750,000', '2,400,000']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize processor
    processor = MovieDataProcessor()
    
    # Clean data
    print("\n1. Cleaning data...")
    df_clean = processor.clean_data(df)
    print(f"Cleaned {len(df_clean)} movies")
    
    # Create features
    print("\n2. Creating content features...")
    df_features = processor.create_content_features(df_clean)
    print(f"Created features: {df_features.columns.tolist()}")
    
    # Create TF-IDF
    print("\n3. Creating TF-IDF matrix...")
    tfidf_matrix, feature_names = processor.create_tfidf_matrix(df_features)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # Create sample ratings
    print("\n4. Creating sample ratings...")
    ratings_df = create_sample_ratings(df_clean['imdb_id'].tolist(), num_users=20)
    print(f"Created {len(ratings_df)} ratings for {ratings_df['user_id'].nunique()} users")
    
    # Create user-item matrix
    print("\n5. Creating user-item matrix...")
    user_item_matrix = processor.create_user_item_matrix(ratings_df, item_col='movie_id')
    print(f"User-item matrix shape: {user_item_matrix.shape}")
    
    print("\nPreprocessing complete!")
