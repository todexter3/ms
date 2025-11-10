# ==========================================
# train_eval.py
# 通用训练 / 验证 / 测试流程
# ==========================================
from config_setup import *
from losses_metrics import masked_mse, mse_rmse

@dataclass
class BatchT:
    Xw: torch.Tensor
    Yw: torch.Tensor
    M: torch.Tensor

def to_device(batch, device=DEVICE):
    return BatchT(
        Xw=batch["Xw"].to(device),
        Yw=batch["Yw"].to(device),
        M=batch["mask"].to(device)
    )

def train_one_epoch(model, loader, optimizer):
    model.train()
    total = 0
    for batch in loader:
        b = to_device(batch)
        preds = model(b.Xw)
        loss = masked_mse(preds, b.Yw)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    return {"train_loss": total / len(loader)}

@torch.no_grad()
def evaluate_val(model, loader):
    model.eval()
    mse_sum, rmse_sum = 0.0, 0.0
    for batch in loader:
        b = to_device(batch)
        preds = model(b.Xw)
        mse, rmse = mse_rmse(preds, b.Yw)
        mse_sum += mse
        rmse_sum += rmse
    return {"val_mse": mse_sum/len(loader), "val_rmse": rmse_sum/len(loader)}
