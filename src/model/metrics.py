import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

def compute_ranking_metrics(y_true, y_pred, draft_years, k_list=[10, 30]):
    """
    Computes per-draft-class ranking metrics and averages them across cohorts.
    """
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'draft_year': draft_years})
    
    spearman_scores = []
    ndcg_10_scores = []
    ndcg_30_scores = []
    hit_rates_10_in_30 = []

    for year, group in df.groupby('draft_year'):
        if len(group) < 10:
            continue
            
        true_vals = group['y_true'].values
        pred_vals = group['y_pred'].values
        
        # 1. Spearman Rank Correlation
        rho, _ = spearmanr(true_vals, pred_vals)
        if not np.isnan(rho):
            spearman_scores.append(rho)
            
        # 2. NDCG@K
        # Shift VORP so scores are strictly positive for NDCG calculations
        min_v = min(true_vals.min(), 0)
        pos_true = np.array([true_vals - min_v + 0.01])
        pos_pred = np.array([pred_vals])
        
        ndcg_10_scores.append(ndcg_score(pos_true, pos_pred, k=10))
        ndcg_30_scores.append(ndcg_score(pos_true, pos_pred, k=min(30, len(group))))
        
        # 3. Hit Rate: Top 10 predicted -> Actual Top 30 VORP producers
        top_10_pred_idx = np.argsort(pred_vals)[::-1][:10]
        top_30_true_idx = set(np.argsort(true_vals)[::-1][:30])
        hits = sum([1 for idx in top_10_pred_idx if idx in top_30_true_idx])
        hit_rates_10_in_30.append(hits / 10.0)

    return {
        "spearman_rho": float(np.mean(spearman_scores)),
        "ndcg_10": float(np.mean(ndcg_10_scores)),
        "ndcg_30": float(np.mean(ndcg_30_scores)),
        "hit_rate_top10_in_30": float(np.mean(hit_rates_10_in_30))
    }