# ==========================================
# losses_metrics.py
# 损失与评估函数
# ==========================================
from config_setup import *

def masked_mse(pred, target, mask=None):
    if mask is None: return F.mse_loss(pred, target)
    if mask.dim() == pred.dim() - 1:
        mask = mask.unsqueeze(-1).expand_as(pred)
    diff = (pred - target) ** 2 * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom

def mse_rmse(pred, target, mask=None):
    mse = masked_mse(pred, target, mask).item()
    return mse, math.sqrt(mse)
