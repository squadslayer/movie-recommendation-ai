"""
Evaluation metrics for recommendation systems.
Implements RMSE, MAE, precision, recall, F1-score, hit rate, and coverage metrics.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class RecommenderEvaluator:
    """Evaluator for recommendation systems."""
    
    def __init__(self):
        self.metrics = {}
    
    def rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error.
        
        Args:
            y_true: Actual ratings
            y_pred: Predicted ratings
            
        Returns:
            RMSE score
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error.
        
        Args:
            y_true: Actual ratings
            y_pred: Predicted ratings
            
        Returns:
            MAE score
        """
        return mean_absolute_error(y_true, y_pred)
    
    def precision_at_k(
        self,
        recommended_items: List[str],
        relevant_items: List[str],
        k: Optional[int] = None
    ) -> float:
        """
        Calculate precision@k.
        
        Args:
            recommended_items: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top items to consider (None = all)
            
        Returns:
            Precision score
        """
        if k is not None:
            recommended_items = recommended_items[:k]
        
        if len(recommended_items) == 0:
            return 0.0
        
        relevant_set = set(relevant_items)
        recommended_set = set(recommended_items)
        
        num_relevant_recommended = len(relevant_set & recommended_set)
        
        return num_relevant_recommended / len(recommended_items)
    
    def recall_at_k(
        self,
        recommended_items: List[str],
        relevant_items: List[str],
        k: Optional[int] = None
    ) -> float:
        """
        Calculate recall@k.
        
        Args:
            recommended_items: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top items to consider (None = all)
            
        Returns:
            Recall score
        """
        if k is not None:
            recommended_items = recommended_items[:k]
        
        if len(relevant_items) == 0:
            return 0.0
        
        relevant_set = set(relevant_items)
        recommended_set = set(recommended_items)
        
        num_relevant_recommended = len(relevant_set & recommended_set)
        
        return num_relevant_recommended / len(relevant_items)
    
    def f1_score_at_k(
        self,
        recommended_items: List[str],
        relevant_items: List[str],
        k: Optional[int] = None
    ) -> float:
        """
        Calculate F1-score@k.
        
        Args:
            recommended_items: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k: Number of top items to consider (None = all)
            
        Returns:
            F1 score
        """
        precision = self.precision_at_k(recommended_items, relevant_items, k)
        recall = self.recall_at_k(recommended_items, relevant_items, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def hit_rate(
        self,
        recommended_items: List[str],
        relevant_items: List[str]
    ) -> float:
        """
        Calculate hit rate (whether any recommended item is relevant).
        
        Args:
            recommended_items: List of recommended item IDs
            relevant_items: List of relevant item IDs
            
        Returns:
            1.0 if hit, 0.0 otherwise
        """
        relevant_set = set(relevant_items)
        recommended_set = set(recommended_items)
        
        return 1.0 if len(relevant_set & recommended_set) > 0 else 0.0
    
    def average_precision(
        self,
        recommended_items: List[str],
        relevant_items: List[str]
    ) -> float:
        """
        Calculate average precision.
        
        Args:
            recommended_items: List of recommended item IDs (ordered)
            relevant_items: List of relevant item IDs
            
        Returns:
            Average precision score
        """
        relevant_set = set(relevant_items)
        
        num_relevant = 0
        sum_precision = 0.0
        
        for i, item in enumerate(recommended_items):
            if item in relevant_set:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                sum_precision += precision_at_i
        
        if num_relevant == 0:
            return 0.0
        
        return sum_precision / len(relevant_items)
    
    def ndcg_at_k(
        self,
        recommended_items: List[str],
        relevant_items: Dict[str, float],
        k: Optional[int] = None
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@k.
        
        Args:
            recommended_items: List of recommended item IDs (ordered)
            relevant_items: Dict mapping item IDs to relevance scores
            k: Number of top items to consider (None = all)
            
        Returns:
            NDCG score
        """
        if k is not None:
            recommended_items = recommended_items[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(recommended_items):
            relevance = relevant_items.get(item, 0.0)
            dcg += relevance / np.log2(i + 2)  # i+2 because i starts at 0
        
        # Calculate ideal DCG
        ideal_items = sorted(relevant_items.items(), key=lambda x: x[1], reverse=True)
        if k is not None:
            ideal_items = ideal_items[:k]
        
        idcg = 0.0
        for i, (_, relevance) in enumerate(ideal_items):
            idcg += relevance / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def catalog_coverage(
        self,
        all_recommendations: List[List[str]],
        all_items: List[str]
    ) -> float:
        """
        Calculate catalog coverage (percentage of items that were recommended).
        
        Args:
            all_recommendations: List of recommendation lists for different users
            all_items: List of all available item IDs
            
        Returns:
            Coverage percentage (0-1)
        """
        recommended_items = set()
        for recs in all_recommendations:
            recommended_items.update(recs)
        
        return len(recommended_items) / len(all_items)
    
    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate rating predictions with multiple metrics.
        
        Args:
            y_true: Actual ratings
            y_pred: Predicted ratings
            metrics: List of metrics to compute (default: ['rmse', 'mae'])
            
        Returns:
            Dictionary of metric scores
        """
        if metrics is None:
            metrics = ['rmse', 'mae']
        
        results = {}
        
        if 'rmse' in metrics:
            results['rmse'] = self.rmse(y_true, y_pred)
        
        if 'mae' in metrics:
            results['mae'] = self.mae(y_true, y_pred)
        
        return results
    
    def evaluate_recommendations(
        self,
        recommended_items: List[str],
        relevant_items: List[str],
        k_values: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Evaluate recommendations with multiple metrics.
        
        Args:
            recommended_items: List of recommended item IDs
            relevant_items: List of relevant item IDs
            k_values: List of k values to evaluate at (default: [5, 10])
            
        Returns:
            Dictionary of metric scores
        """
        if k_values is None:
            k_values = [5, 10]
        
        results = {}
        
        for k in k_values:
            results[f'precision@{k}'] = self.precision_at_k(recommended_items, relevant_items, k)
            results[f'recall@{k}'] = self.recall_at_k(recommended_items, relevant_items, k)
            results[f'f1@{k}'] = self.f1_score_at_k(recommended_items, relevant_items, k)
        
        results['hit_rate'] = self.hit_rate(recommended_items, relevant_items)
        results['average_precision'] = self.average_precision(recommended_items, relevant_items)
        
        return results
    
    def cross_validate(
        self,
        user_item_matrix: pd.DataFrame,
        recommender,
        n_folds: int = 5,
        test_size: float = 0.2
    ) -> Dict[str, List[float]]:
        """
        Perform cross-validation on a recommender.
        
        Args:
            user_item_matrix: User-item rating matrix
            recommender: Recommender object with fit and predict methods
            n_folds: Number of folds
            test_size: Proportion of test data
            
        Returns:
            Dictionary of metric scores for each fold
        """
        # Simple implementation - can be enhanced
        results = {'rmse': [], 'mae': []}
        
        for fold in range(n_folds):
            # Split data (simple random split for demonstration)
            np.random.seed(fold)
            
            # This is a simplified version - in practice, you'd want more sophisticated splitting
            print(f"Fold {fold + 1}/{n_folds} - Cross-validation not fully implemented in this demo")
        
        return results
    
    def plot_metrics(self, metrics_dict: Dict[str, float], title: str = "Evaluation Metrics"):
        """
        Plot evaluation metrics.
        
        Args:
            metrics_dict: Dictionary of metric names and values
            title: Plot title
        """
        plt.figure(figsize=(10, 6))
        
        metric_names = list(metrics_dict.keys())
        metric_values = list(metrics_dict.values())
        
        plt.bar(metric_names, metric_values, color='skyblue', edgecolor='navy')
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    def plot_precision_recall_curve(
        self,
        recommended_items: List[str],
        relevant_items: List[str],
        max_k: int = 20
    ):
        """
        Plot precision-recall curve for different k values.
        
        Args:
            recommended_items: List of recommended item IDs
            relevant_items: List of relevant item IDs
            max_k: Maximum k value to plot
        """
        k_values = list(range(1, min(max_k + 1, len(recommended_items) + 1)))
        precisions = []
        recalls = []
        
        for k in k_values:
            precisions.append(self.precision_at_k(recommended_items, relevant_items, k))
            recalls.append(self.recall_at_k(recommended_items, relevant_items, k))
        
        plt.figure(figsize=(10, 6))
        plt.plot(recalls, precisions, marker='o', linestyle='-', color='blue')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Example usage
    print("Recommender Evaluation Metrics")
    print("===============================\n")
    
    evaluator = RecommenderEvaluator()
    
    # Example 1: Rating prediction evaluation
    print("1. Rating Prediction Evaluation")
    print("-" * 40)
    y_true = np.array([5, 4, 3, 5, 4, 2, 3, 4, 5, 3])
    y_pred = np.array([4.5, 4.2, 3.1, 4.8, 3.9, 2.5, 3.2, 4.1, 4.7, 3.3])
    
    rmse = evaluator.rmse(y_true, y_pred)
    mae = evaluator.mae(y_true, y_pred)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Example 2: Recommendation evaluation
    print("\n2. Recommendation Evaluation")
    print("-" * 40)
    recommended = ['movie_1', 'movie_3', 'movie_5', 'movie_7', 'movie_9', 'movie_2']
    relevant = ['movie_1', 'movie_2', 'movie_4', 'movie_6', 'movie_8']
    
    results = evaluator.evaluate_recommendations(recommended, relevant, k_values=[3, 5])
    
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")
    
    # Example 3: NDCG
    print("\n3. NDCG Evaluation")
    print("-" * 40)
    recommended_ndcg = ['movie_1', 'movie_3', 'movie_2', 'movie_5']
    relevant_scores = {
        'movie_1': 5,
        'movie_2': 3,
        'movie_3': 4,
        'movie_4': 2
    }
    
    ndcg = evaluator.ndcg_at_k(recommended_ndcg, relevant_scores, k=4)
    print(f"NDCG@4: {ndcg:.4f}")
    
    # Example 4: Coverage
    print("\n4. Catalog Coverage")
    print("-" * 40)
    all_recs = [
        ['movie_1', 'movie_2', 'movie_3'],
        ['movie_2', 'movie_4', 'movie_5'],
        ['movie_1', 'movie_3', 'movie_6']
    ]
    all_movies = [f'movie_{i}' for i in range(1, 11)]
    
    coverage = evaluator.catalog_coverage(all_recs, all_movies)
    print(f"Catalog Coverage: {coverage:.2%}")
    
    print("\nEvaluation demo complete!")
