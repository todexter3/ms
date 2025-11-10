# ==========================================
# baseline_runners.py
# 不同模型的训练封装
# ==========================================
from config_setup import *
from models import SimpleMLP, GRURegressor, TCNRegressor
from losses_metrics import mse_rmse
import torch.nn.functional as F
from xgboost import XGBRegressor
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import math, gc, os
def run_mlp_baseline(X_train, Y_train, X_test, Y_test, epochs=EPOCHS, lr=LR, ticker_n=10):
    """
    MLP baseline with NaN/Inf protection and per-dimension (y36/y320) metrics.
    Adds FINAL evaluation per ticker (MSE/RMSE for y36/y320).
    """
    result_file = "results/mlp_result.txt"
    os.makedirs("results", exist_ok=True)

    # === Input sanitization ===
    def sanitize(arr, name):
        nan_count = np.isnan(arr).sum()
        inf_count = np.isinf(arr).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"⚠️ [{name}] contains {nan_count} NaN / {inf_count} Inf → replaced with 0.")
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        return arr

    X_train = sanitize(X_train, "X_train")
    Y_train = sanitize(Y_train, "Y_train")
    X_test  = sanitize(X_test,  "X_test")
    Y_test  = sanitize(Y_test,  "Y_test")

    mask_valid = ~np.isnan(Y_train).any(axis=1)
    X_train, Y_train = X_train[mask_valid], Y_train[mask_valid]

    # === Model ===
    model = SimpleMLP().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xtr, Ytr = torch.from_numpy(X_train).float().to(DEVICE), torch.from_numpy(Y_train).float().to(DEVICE)
    Xte, Yte = torch.from_numpy(X_test).float().to(DEVICE), torch.from_numpy(Y_test).float().to(DEVICE)

    with open(result_file, "a") as f:
        f.write("Epoch | TrainLoss | TotalMSE | TotalRMSE | MSE_y36 | RMSE_y36 | MSE_y320 | RMSE_y320\n")

    # === Training loop ===
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred = model(Xtr)
        loss = loss_fn(pred, Ytr)

        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=-1e6)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # === Evaluation ===
        model.eval()
        with torch.no_grad():
            preds = model(Xte)
            preds = torch.nan_to_num(preds, nan=0.0, posinf=1e6, neginf=-1e6)
            mse_total = F.mse_loss(preds, Yte).item()
            rmse_total = math.sqrt(mse_total)
            mse_y36 = F.mse_loss(preds[:, 0], Yte[:, 0]).item()
            rmse_y36 = math.sqrt(mse_y36)
            mse_y320 = F.mse_loss(preds[:, 1], Yte[:, 1]).item()
            rmse_y320 = math.sqrt(mse_y320)

        msg = (f"Epoch {ep:03d} | Train={loss.item():.6f} | "
               f"TotalMSE={mse_total:.6f}, RMSE={rmse_total:.6f} | "
               f"y36 MSE={mse_y36:.6f}, RMSE={rmse_y36:.6f} | "
               f"y320 MSE={mse_y320:.6f}, RMSE={rmse_y320:.6f}")
        print(msg)
        with open(result_file, "a") as f:
            f.write(msg + "\n")

        torch.cuda.empty_cache()

    # === Final evaluation per ticker ===
    print("\n[FINAL] Evaluating per-ticker performance...")
    with torch.no_grad():
        preds = model(Xte).cpu().numpy()
        Yte_np = Yte.cpu().numpy()

    with open(result_file, "a") as f:
        f.write("\n[FINAL RESULTS]\n")

    all_y36_pred, all_y36_true = [], []
    all_y320_pred, all_y320_true = [], []

    for a in range(ticker_n):
        start = a * len(preds) // ticker_n
        end = (a + 1) * len(preds) // ticker_n
        p_a, y_a = preds[start:end], Yte_np[start:end]

        mse_total = np.mean((p_a - y_a) ** 2)
        rmse_total = math.sqrt(mse_total)
        mse_y36 = np.mean((p_a[:, 0] - y_a[:, 0]) ** 2)
        rmse_y36 = math.sqrt(mse_y36)
        mse_y320 = np.mean((p_a[:, 1] - y_a[:, 1]) ** 2)
        rmse_y320 = math.sqrt(mse_y320)

        print(f"Ticker {a:02d} | Total MSE={mse_total:.6f}, RMSE={rmse_total:.6f} | "
              f"y36 MSE={mse_y36:.6f}, RMSE={rmse_y36:.6f} | "
              f"y320 MSE={mse_y320:.6f}, RMSE={rmse_y320:.6f}")

        with open(result_file, "a") as f:
            f.write(f"Ticker {a:02d} | "
                    f"Total MSE={mse_total:.6f}, RMSE={rmse_total:.6f} | "
                    f"y36 MSE={mse_y36:.6f}, RMSE={rmse_y36:.6f} | "
                    f"y320 MSE={mse_y320:.6f}, RMSE={rmse_y320:.6f}\n")

        # === 收集全局统计用的数据 ===
        all_y36_pred.append(p_a[:, 0])
        all_y36_true.append(y_a[:, 0])
        all_y320_pred.append(p_a[:, 1])
        all_y320_true.append(y_a[:, 1])

    # === 计算全局 MSE / RMSE ===
    all_y36_pred = np.concatenate(all_y36_pred, axis=0)
    all_y36_true = np.concatenate(all_y36_true, axis=0)
    all_y320_pred = np.concatenate(all_y320_pred, axis=0)
    all_y320_true = np.concatenate(all_y320_true, axis=0)

    global_mse_y36 = np.mean((all_y36_pred - all_y36_true) ** 2)
    global_rmse_y36 = math.sqrt(global_mse_y36)
    global_mse_y320 = np.mean((all_y320_pred - all_y320_true) ** 2)
    global_rmse_y320 = math.sqrt(global_mse_y320)

    print("\n[GLOBAL RESULTS]")
    print(f"y36  | MSE={global_mse_y36:.6f}, RMSE={global_rmse_y36:.6f}")
    print(f"y320 | MSE={global_mse_y320:.6f}, RMSE={global_rmse_y320:.6f}")

    with open(result_file, "a") as f:
        f.write("\n[GLOBAL RESULTS]\n")
        f.write(f"y36  | MSE={global_mse_y36:.6f}, RMSE={global_rmse_y36:.6f}\n")
        f.write(f"y320 | MSE={global_mse_y320:.6f}, RMSE={global_rmse_y320:.6f}\n")

    gc.collect()
    torch.cuda.empty_cache()
    return model

def run_xgboost_baseline(X_new, Y_new, train_indices, test_indices,
                         window=190, stride=5, epochs=200):
    """
    XGBoost baseline with per-ticker and per-label (y36/y320) evaluation.
    Now logs both MSE and RMSE for total and per-ticker results.
    """
    ticker_n, T, feature_n = X_new.shape[0], X_new.shape[1], 9
    Xs, Ys, tids, tstep = [], [], [], []
    result_file = "results/xgboost_result.txt"
    os.makedirs("results", exist_ok=True)

    # === Build sliding windows ===
    for a in range(ticker_n):
        Xa, Ya = X_new[a], Y_new[a]
        for s in range(0, T - window, stride):
            e = s + window
            if np.isnan(Ya[e - 1]).any():
                continue
            Xs.append(Xa[s:e, :feature_n].flatten())
            Ys.append(Ya[e - 1])
            tids.append(a)
            tstep.append(e - 1)

    Xs, Ys, tids, tstep = np.array(Xs), np.array(Ys), np.array(tids), np.array(tstep)
    train_mask = tstep <= train_indices[-1]
    test_mask = (tstep >= test_indices[0]) & (tstep <= test_indices[-1])

    Xtr, Ytr = Xs[train_mask], Ys[train_mask]
    Xte, Yte = Xs[test_mask], Ys[test_mask]
    tids_te = tids[test_mask]

    # === Train 2 separate regressors for y36 and y320 ===
    models = [
        XGBRegressor(
            n_estimators=epochs,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method="hist"
        )
        for _ in range(2)
    ]
    for i in range(2):
        models[i].fit(Xtr, Ytr[:, i])

    # === Predictions ===
    preds = np.stack([m.predict(Xte) for m in models], axis=1)

    # === Global Metrics ===
    mse_total = np.mean((preds - Yte) ** 2)
    rmse_total = math.sqrt(mse_total)
    mse_y36 = np.mean((preds[:, 0] - Yte[:, 0]) ** 2)
    rmse_y36 = math.sqrt(mse_y36)
    mse_y320 = np.mean((preds[:, 1] - Yte[:, 1]) ** 2)
    rmse_y320 = math.sqrt(mse_y320)

    print(f"✅ XGBoost Total MSE={mse_total:.6f}, RMSE={rmse_total:.6f}")
    print(f"   y36  MSE={mse_y36:.6f}, RMSE={rmse_y36:.6f}")
    print(f"   y320 MSE={mse_y320:.6f}, RMSE={rmse_y320:.6f}")

    with open(result_file, "a") as f:
        f.write("\n--- XGBoost Results ---\n")
        f.write(f"Total MSE={mse_total:.6f}, RMSE={rmse_total:.6f}\n")
        f.write(f"y36  MSE={mse_y36:.6f}, RMSE={rmse_y36:.6f}\n")
        f.write(f"y320 MSE={mse_y320:.6f}, RMSE={rmse_y320:.6f}\n\n")
        f.write("Per-ticker breakdown (MSE / RMSE | y36 / y320):\n")

    # === Per-ticker breakdown ===
    for a in np.unique(tids_te):
        mask = tids_te == a
        mse_total_a = np.mean((preds[mask] - Yte[mask]) ** 2)
        rmse_total_a = math.sqrt(mse_total_a)

        mse_y36_a = np.mean((preds[mask, 0] - Yte[mask, 0]) ** 2)
        rmse_y36_a = math.sqrt(mse_y36_a)

        mse_y320_a = np.mean((preds[mask, 1] - Yte[mask, 1]) ** 2)
        rmse_y320_a = math.sqrt(mse_y320_a)

        print(
            f"Asset {a:02d} | "
            f"MSE_total={mse_total_a:.6f}, RMSE_total={rmse_total_a:.6f} | "
            f"y36 MSE={mse_y36_a:.6f}, RMSE={rmse_y36_a:.6f} | "
            f"y320 MSE={mse_y320_a:.6f}, RMSE={rmse_y320_a:.6f}"
        )

        with open(result_file, "a") as f:
            f.write(
                f"Asset {a:02d} | "
                f"MSE_total={mse_total_a:.6f}, RMSE_total={rmse_total_a:.6f} | "
                f"y36 MSE={mse_y36_a:.6f}, RMSE={rmse_y36_a:.6f} | "
                f"y320 MSE={mse_y320_a:.6f}, RMSE={rmse_y320_a:.6f}\n"
            )

    return models



# ===== 辅助函数 =====
def evaluate_in_batches(model, X_test, Y_test, device, batch_size=128):
    """分批推理防爆显存"""
    model.eval()
    
    ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    preds_list, targets_list = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds_list.append(pred)
            targets_list.append(yb.numpy())
            del xb, yb, pred
            # torch.cuda.empty_cache()
    preds = np.concatenate(preds_list, axis=0)
    Yte_np = np.concatenate(targets_list, axis=0)
    return preds, Yte_np

def safe_mse(a, b):
    """忽略NaN安全计算MSE"""
    mask = ~np.isnan(a) & ~np.isnan(b)
    if np.any(mask):
        return np.mean((a[mask] - b[mask]) ** 2)
    return np.nan


# ===== 主函数 =====
def train_model(model, X_train, Y_train, X_test, Y_test, result_file, 
                epochs=50, lr=3e-4, batch_size=32, use_amp=True):

    def sanitize(arr, name):
        nan_count = np.isnan(arr).sum()
        inf_count = np.isinf(arr).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"⚠️ [{name}] contains {nan_count} NaN / {inf_count} Inf → replaced with 0.")
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        return arr

    # === Step 1: 数据清洗 ===
    X_train = sanitize(X_train, "X_train")
    Y_train = sanitize(Y_train, "Y_train")
    X_test  = sanitize(X_test,  "X_test")
    Y_test  = sanitize(Y_test,  "Y_test")
    mask_valid = ~np.isnan(Y_train).any(axis=1)
    X_train, Y_train = X_train[mask_valid], Y_train[mask_valid]

    # === Step 2: 初始化 ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # === Step 3: DataLoader ===
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float())
    test_ds  = TensorDataset(torch.from_numpy(X_test).float(),  torch.from_numpy(Y_test).float())
    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dl_test  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    os.makedirs("results", exist_ok=True)
    with open(result_file, "a") as f:
        f.write("Epoch | TrainLoss | TestMSE | TestRMSE | MSE_y36 | RMSE_y36 | MSE_y320 | RMSE_y320\n")

    # === Step 4: 训练循环 ===
    for ep in range(1, epochs + 1):
        model.train()
        train_loss, nan_flag = 0, False
        for xb, yb in tqdm(dl_train, desc=f"Epoch {ep}/{epochs}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if loss.ndim > 0:
                loss = loss.mean()

            # 防止 NaN / Inf
            # if not torch.isfinite(loss):
            #     print(f"⚠️ Non-finite loss detected, replacing with 0 at this batch.")
            #     loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))

            # backward + optimizer
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(dl_train.dataset)

        # === Step 5: 验证 ===
        model.eval()
        with torch.no_grad():
            all_preds, all_true = [], []
            for xb, yb in dl_test:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                preds = torch.nan_to_num(preds, nan=0.0, posinf=1e6, neginf=-1e6)
                all_preds.append(preds.cpu())
                all_true.append(yb.cpu())
            preds = torch.cat(all_preds)
            Yte = torch.cat(all_true)
            mse_total = F.mse_loss(preds, Yte).item()
            rmse_total = math.sqrt(mse_total)
            mse_y36 = F.mse_loss(preds[:, 0], Yte[:, 0]).item()
            rmse_y36 = math.sqrt(mse_y36)
            mse_y320 = F.mse_loss(preds[:, 1], Yte[:, 1]).item()
            rmse_y320 = math.sqrt(mse_y320)

        msg = (f"Epoch {ep:03d} | Train={train_loss:.6f} | "
               f"Total MSE={mse_total:.6f}, RMSE={rmse_total:.6f} | "
               f"y36 MSE={mse_y36:.6f}, RMSE={rmse_y36:.6f} | "
               f"y320 MSE={mse_y320:.6f}, RMSE={rmse_y320:.6f}")
        if nan_flag:
            msg += " ⚠️ NaN encountered"
        print(msg)
        with open(result_file, "a") as f:
            f.write(msg + "\n")
        torch.cuda.empty_cache()

    # === Step 6: 分批最终评估（显存安全 + NaN安全） ===
        # === Step 6: 分批最终评估（显存安全 + NaN安全 + 诊断） ===
    print("\n[FINAL] Evaluating per-ticker performance...")
    preds, Yte_np = evaluate_in_batches(model, X_test, Y_test, device, batch_size=128)

    # --- 显式打印诊断信息 ---
    print(f"[Diag] preds shape={preds.shape}, Yte shape={Yte_np.shape}")
    print(f"[Diag] NaN in preds: {np.isnan(preds).sum()} / {preds.size}")
    print(f"[Diag] NaN in Yte_np: {np.isnan(Yte_np).sum()} / {Yte_np.size}")

    # --- 若全NaN则强制替换 ---
    preds = np.nan_to_num(preds, nan=0.0, posinf=1e6, neginf=-1e6)
    Yte_np = np.nan_to_num(Yte_np, nan=0.0, posinf=1e6, neginf=-1e6)

    with open(result_file, "a") as f:
        f.write("\n[FINAL RESULTS]\n")

    ticker_n = 10
    n_per_ticker = len(preds) // ticker_n

    # === 先初始化全局列表 ===
    all_y36_pred, all_y36_true = [], []
    all_y320_pred, all_y320_true = [], []

    for a in range(ticker_n):
        start, end = a * n_per_ticker, (a + 1) * n_per_ticker
        p_a, y_a = preds[start:end], Yte_np[start:end]

        # --- 打印每个 ticker 的有效样本数 ---
        valid_mask = ~np.isnan(p_a) & ~np.isnan(y_a)
        valid_ratio = np.mean(valid_mask)
        print(f"[Diag] Ticker {a:02d}: valid ratio = {valid_ratio:.2%}")

        # --- 若仍全 NaN 则跳过 ---
        if np.all(~valid_mask):
            print(f"⚠️ Ticker {a:02d}: no valid data, skipping.")
            continue

        # --- 安全计算 ---
        mse_total = safe_mse(p_a, y_a)
        rmse_total = math.sqrt(mse_total) if not np.isnan(mse_total) else np.nan
        mse_y36 = safe_mse(p_a[:, 0], y_a[:, 0])
        rmse_y36 = math.sqrt(mse_y36) if not np.isnan(mse_y36) else np.nan
        mse_y320 = safe_mse(p_a[:, 1], y_a[:, 1])
        rmse_y320 = math.sqrt(mse_y320) if not np.isnan(mse_y320) else np.nan

        print(f"Ticker {a:02d} | Total MSE={mse_total:.6f}, RMSE={rmse_total:.6f} | "
              f"y36 MSE={mse_y36:.6f}, RMSE={rmse_y36:.6f} | "
              f"y320 MSE={mse_y320:.6f}, RMSE={rmse_y320:.6f}")
        with open(result_file, "a") as f:
            f.write(f"Ticker {a:02d} | "
                    f"Total MSE={mse_total:.6f}, RMSE={rmse_total:.6f} | "
                    f"y36 MSE={mse_y36:.6f}, RMSE={rmse_y36:.6f} | "
                    f"y320 MSE={mse_y320:.6f}, RMSE={rmse_y320:.6f}\n")

        # === 累积全局数据 ===
        all_y36_pred.append(p_a[:, 0])
        all_y36_true.append(y_a[:, 0])
        all_y320_pred.append(p_a[:, 1])
        all_y320_true.append(y_a[:, 1])

    # === 拼接所有 ticker 的样本 ===
    all_y36_pred = np.concatenate(all_y36_pred, axis=0)
    all_y36_true = np.concatenate(all_y36_true, axis=0)
    all_y320_pred = np.concatenate(all_y320_pred, axis=0)
    all_y320_true = np.concatenate(all_y320_true, axis=0)

    # === 计算全局 MSE/RMSE ===
    global_mse_y36 = safe_mse(all_y36_pred, all_y36_true)
    global_rmse_y36 = math.sqrt(global_mse_y36) if not np.isnan(global_mse_y36) else np.nan
    global_mse_y320 = safe_mse(all_y320_pred, all_y320_true)
    global_rmse_y320 = math.sqrt(global_mse_y320) if not np.isnan(global_mse_y320) else np.nan

    print("\n[GLOBAL RESULTS]")
    print(f"y36  | MSE={global_mse_y36:.6f}, RMSE={global_rmse_y36:.6f}")
    print(f"y320 | MSE={global_mse_y320:.6f}, RMSE={global_rmse_y320:.6f}")

    with open(result_file, "a") as f:
        f.write("\n[GLOBAL RESULTS]\n")
        f.write(f"y36  | MSE={global_mse_y36:.6f}, RMSE={global_rmse_y36:.6f}\n")
        f.write(f"y320 | MSE={global_mse_y320:.6f}, RMSE={global_rmse_y320:.6f}\n")
    # gc.collect()
    # torch.cuda.empty_cache()
    return model

def run_gru_baseline(X_train, Y_train, X_test, Y_test, epochs=EPOCHS, lr=3e-4):
    from models import GRURegressor
    model = GRURegressor()
    return train_model(model, X_train, Y_train, X_test, Y_test,
                       result_file= "results/gru_result.txt", epochs=2, lr=lr, batch_size=16)

def run_tcn_baseline(X_train, Y_train, X_test, Y_test, epochs=EPOCHS, lr=3e-4):
    from models import TCNRegressor
    model = TCNRegressor()
    return train_model(model, X_train, Y_train, X_test, Y_test,
                       result_file= "results/tcn_result.txt", epochs=5, lr=lr, batch_size=16)

def run_shared_transformer_baseline(X_train, Y_train, X_test, Y_test,
                                    epochs=EPOCHS, lr=3e-4, batch_size=16):
    """
    Shared Transformer baseline:
      - 自适应 in_dim = X_new.shape[-1]
      - 使用共享 Transformer encoder + 每 ticker 独立 head + 注意力聚合
      - 训练和评估逻辑与 GRU/TCN 一致
    """
    from models import SharedTransf  # 确保模型在 models.py 中定义

    # === Step 1: 自动推断维度 ===
    in_dim = X_train.shape[-1]
    n_tickers = X_train.shape[1] if X_train.ndim == 4 else 10  # 默认10个ticker

    print(f"🚀 Running Shared Transformer baseline: in_dim={in_dim}, n_tickers={n_tickers}")
    print(f"📏 Shapes — X_train: {X_train.shape}, Y_train: {Y_train.shape} | "
          f"X_test: {X_test.shape}, Y_test: {Y_test.shape}")

    # === Step 2: 实例化模型 ===
    model = SharedTransf(in_dim=in_dim, emb_dim=128, n_tickers=n_tickers)

    # === Step 3: 调用通用训练函数 ===
    return train_model(
        model,
        X_train,
        Y_train,
        X_test,
        Y_test,
        result_file="results/shared_transformer_result.txt",
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        use_amp=False,     # Transformer 通常可稳定开启 AMP，但为安全起见先关
    )

