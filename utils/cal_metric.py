import numpy as np
# auc
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr
# pearson_corr, _ = pearsonr(t_true, p_score)

def SIM(map1, map2, eps=1e-12):
    map1, map2 = map1/(map1.sum()+eps), map2/(map2.sum() + eps)
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)


def cal_metric(preds: list, gts: list) -> dict:
    """
    Calculate metrics
    Args:
        preds: (np.ndarray) model predictions
        gts: (np.ndarray) ground truth values
    Returns:
        (dict) metrics
    """
    # all_num = 24656
    # add auc and sim and iou
    num = len(preds)
    mae = 0
    mse = 0
    log_mse = 0
    pearson_corr_sum = 0
    SIM_matrix = np.zeros(num)
    cIOU = 0
    for i in range(len(preds)):
        # mae
        mae += np.mean(np.abs(preds[i] - gts[i]))
        mse += np.mean((preds[i] - gts[i]) ** 2)

        non_zero_idx = gts[i] != 0
        log_mse += np.mean((np.log1p(gts[i]) - np.log1p(preds[i])) ** 2)

        gts_row = gts[i].flatten()  # 或 gts[i] 如果是二维列表也适用
        preds_row = preds[i].flatten()

        valid_idx = (~np.isnan(gts_row)) & (~np.isnan(preds_row)) & (~np.isinf(gts_row)) & (~np.isinf(preds_row))
        gts_clean = gts_row[valid_idx]
        preds_clean = preds_row[valid_idx]
        pearson_corr, _ = pearsonr(gts_clean, preds_clean)
        pearson_corr_sum += pearson_corr

        # sim
        SIM_matrix[i] = SIM(preds[i], gts[i])

        # cIoU
        intersect = np.sum(np.minimum(preds[i], gts[i]))
        union = np.sum(np.maximum(preds[i], gts[i]))
        iou_continuous = intersect / union if union > 0 else 0
        cIOU += iou_continuous

    sim = np.mean(SIM_matrix)
    metrics = {}
    metrics['mae'] = mae / num
    metrics['mse'] = mse / num
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['log_mse'] = log_mse / num
    metrics['pearson_corr'] = pearson_corr_sum / num
    metrics['sim'] = sim
    metrics['ciou'] = cIOU / num

    return metrics