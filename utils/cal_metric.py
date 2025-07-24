import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
# pearson_corr, _ = pearsonr(t_true, p_score)

def SIM(map1, map2, eps=1e-12):
    map1, map2 = map1/(map1.sum()+eps), map2/(map2.sum() + eps)
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)

def cosine_similarity(map1, map2, eps=1e-12):
    """
    Compute the cosine similarity between two maps.
    """
    map1 = map1.flatten()
    map2 = map2.flatten()
    
    dot_product = np.dot(map1, map2)
    norm_map1 = np.linalg.norm(map1) + eps
    norm_map2 = np.linalg.norm(map2) + eps
    
    return dot_product / (norm_map1 * norm_map2)


def cal_metric_new(preds: list, gts: list) -> dict:
    """
    Calculate metrics
    Args:
        preds: (np.ndarray) model predictions
        gts: (np.ndarray) ground truth values
    Returns:
        (dict) metrics
    """
    num = len(preds)
    log_mse = 0
    pearson_corr_sum = 0
    SIM_matrix = np.zeros(num)
    
    for i in range(num):        
        log_mse += np.mean((np.log1p(gts[i]) - np.log1p(preds[i])) ** 2)
        
        gts_row = gts[i].flatten()
        preds_row = preds[i].flatten()
        
        valid_idx = (~np.isnan(gts_row)) & (~np.isnan(preds_row)) & (~np.isinf(gts_row)) & (~np.isinf(preds_row))
        gts_clean = gts_row[valid_idx]
        preds_clean = preds_row[valid_idx]
        
        if len(gts_clean) > 1 and len(preds_clean) > 1:
            pearson_corr, _ = pearsonr(gts_clean, preds_clean)
        else:
            pearson_corr = 0
        
        pearson_corr_sum += pearson_corr
        
        # Compute cosine similarity
        SIM_matrix[i] = cosine_similarity(preds[i], gts[i])
    
    sim = np.mean(SIM_matrix)
    
    metrics = {
        'rmse': np.sqrt(mse / num),
        'log_mse': log_mse / num,
        'pearson_corr': pearson_corr_sum / num,
        'sim': sim,
    }
    
    return metrics