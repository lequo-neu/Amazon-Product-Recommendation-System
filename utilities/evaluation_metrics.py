import numpy as np
from sklearn.metrics import mean_squared_error
from typing import List, Set, Dict, Optional

def compute_rmse_accuracy(predictions: np.ndarray, 
                         actuals: np.ndarray, 
                         threshold: float = 3.5) -> tuple[float, float]:
    mask = ~np.isnan(actuals)
    
    if mask.sum() == 0:
        return np.nan, np.nan
    
    pred_filtered = predictions[mask]
    actual_filtered = actuals[mask]
    
    rmse_raw = np.sqrt(mean_squared_error(actual_filtered, pred_filtered))
    
    max_possible_rmse = 4.5
    rmse_normalized = 1 - (rmse_raw / max_possible_rmse)
    rmse_normalized = np.clip(rmse_normalized, 0, 1)
    
    accuracy = np.mean(
        (pred_filtered >= threshold) == (actual_filtered >= threshold)
    )
    
    return rmse_normalized, accuracy

# ==============================================================================
# RANKING METRICS
# ==============================================================================

def recall_at_k(recommended: List[str], 
                relevant: Set[str], 
                k: int) -> float:
    if len(relevant) == 0:
        return 0.0
    
    recommended_k = set(recommended[:k])
    hits = len(recommended_k & relevant)
    
    return hits / len(relevant)


def ndcg_at_k(recommended: List[str], 
              relevant: Set[str], 
              k: int) -> float:
    if len(relevant) == 0:
        return 0.0
    
    recommended_k = recommended[:k]
    
    # DCG: Sum of discounted gains for relevant items
    dcg = sum(
        1.0 / np.log2(i + 2) 
        for i, item in enumerate(recommended_k) 
        if item in relevant
    )
    
    # IDCG: Best possible DCG (all relevant items ranked first)
    idcg = sum(
        1.0 / np.log2(i + 2) 
        for i in range(min(len(relevant), k))
    )
    
    return dcg / idcg if idcg > 0 else 0.0


def map_at_k(recommended: List[str], 
             relevant: Set[str], 
             k: int) -> float:
    if len(relevant) == 0:
        return 0.0
    
    recommended_k = recommended[:k]
    
    score = 0.0
    num_hits = 0.0
    
    for i, item in enumerate(recommended_k):
        if item in relevant:
            num_hits += 1.0
            # Precision at position i+1
            precision_at_i = num_hits / (i + 1.0)
            score += precision_at_i
    
    # Normalize by min(|relevant|, k)
    return score / min(len(relevant), k)
