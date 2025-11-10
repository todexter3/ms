# ==========================================
# main.py
# 主入口: 调用预处理 + 运行baseline
# ==========================================
from preprocess_data import load_and_preprocess_data
from dataset_utils import make_sliding_window_dataset
from baseline_runners import run_mlp_baseline, run_gru_baseline, run_tcn_baseline, run_xgboost_baseline, run_shared_transformer_baseline
from config_setup import *
import itertools

def build_mt_loaders(
    X, Y, train_indices, test_indices,
    window=190, stride=5, batch_size=16, num_workers=0
):
    """为 multi_ticker_transformer 构建 Train/Val/Test DataLoader"""
    T = X.shape[1]
    all_starts = np.arange(0, T - window + 1, stride, dtype=np.int64)

    train_end = train_indices[-1]
    test_start, test_end = test_indices[0], test_indices[-1]
    w_end = all_starts + window - 1

    train_starts = all_starts[w_end <= train_end]
    test_starts  = all_starts[(w_end >= test_start) & (w_end <= test_end)]

    # 取训练末尾 5% 做验证
    n_val = max(1, int(len(train_starts) * 0.05))
    val_starts = train_starts[-n_val:]
    train_starts = train_starts[:-n_val] if len(train_starts) > n_val else train_starts[:0]

    ds_train = WindowDataset(X, Y, train_starts, window)
    ds_val   = WindowDataset(X, Y, val_starts, window)
    ds_test  = WindowDataset(X, Y, test_starts, window)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    info = {
        "n_train": len(ds_train),
        "n_val": len(ds_val),
        "n_test": len(ds_test),
        "window": window, "stride": stride,
        "train_starts": train_starts, "val_starts": val_starts, "test_starts": test_starts
    }
    print(f"[Windows] Train={info['n_train']}, Val={info['n_val']}, Test={info['n_test']} | "
          f"Window={window}, Stride={stride}")
    return dl_train, dl_val, dl_test, info

def train_mt_one_epoch(model, loader, optimizer, device="cuda"):
    model.train()
    loss_meter, n_batches = 0.0, 0
    for batch in loader:
        Xw = batch["Xw"].to(device)     # (B,10,L,17)
        M   = batch["mask"].to(device)  # (B,10,L,17)
        Yw  = batch["Yw"].to(device)    # (B,10,2)

        # 只训练前9个资产
        X9, M9, Y9 = Xw[:, :9], M[:, :9], Yw[:, :9]
        preds9 = model.forward_train9(X9, M9)                 # (B,9,2)

        ymask = ~torch.isnan(Y9).any(dim=-1)                  # (B,9)
        yvalid = torch.nan_to_num(Y9, nan=0.0)
        loss = masked_mse(preds9, yvalid, ymask)              # 你已有的 masked_mse

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_meter += loss.item()
        n_batches += 1
    return {"train_loss": loss_meter / max(n_batches, 1)}

@torch.no_grad()
def eval_mt_val(model, loader, device="cuda"):
    model.eval()
    mse_y36_sum, rmse_y36_sum = 0.0, 0.0
    mse_y320_sum, rmse_y320_sum = 0.0, 0.0
    n = 0

    for batch in loader:
        Xw = batch["Xw"].to(device)
        M = batch["mask"].to(device)
        Yw = batch["Yw"].to(device)

        # 取前9个asset
        X9, M9, Y9 = Xw[:, :9], M[:, :9], Yw[:, :9]
        preds9 = model.forward_train9(X9, M9)  # (B, 9, 2)

        # 掩码处理
        mask9 = ~torch.isnan(Y9).any(dim=-1)  # (B, 9)
        Y9v = torch.nan_to_num(Y9, nan=0.0)

        # === 分别计算 y36 和 y320 ===
        Y36, Y320 = Y9v[..., 0], Y9v[..., 1]
        P36, P320 = preds9[..., 0], preds9[..., 1]

        mask36 = mask9 & ~torch.isnan(Y36)
        mask320 = mask9 & ~torch.isnan(Y320)

        mse36, rmse36 = mse_rmse(P36, Y36, mask36)
        mse320, rmse320 = mse_rmse(P320, Y320, mask320)

        mse_y36_sum += mse36
        rmse_y36_sum += rmse36
        mse_y320_sum += mse320
        rmse_y320_sum += rmse320
        n += 1

    return {
        "val_y36_mse": mse_y36_sum / max(n, 1),
        "val_y36_rmse": rmse_y36_sum / max(n, 1),
        "val_y320_mse": mse_y320_sum / max(n, 1),
        "val_y320_rmse": rmse_y320_sum / max(n, 1),
    }

@torch.no_grad()
def eval_mt_test(model, loader, device="cuda", verbose=True):
    """
    Evaluate on 10 tickers:
    - Overall (10 assets combined)
    - Per-ticker overall MSE/RMSE
    - Per-ticker per-label (y36/y320) MSE/RMSE
    - NEW: Overall y36/y320 MSE/RMSE (aggregated over all tickers)
    """
    model.eval()
    mse_sum, rmse_sum, n = 0.0, 0.0, 0
    mse_y36_sum, mse_y320_sum = 0.0, 0.0
    rmse_y36_sum, rmse_y320_sum = 0.0, 0.0

    ticker_mse = torch.zeros(10, device=device)
    ticker_rmse = torch.zeros(10, device=device)
    ticker_cnt = torch.zeros(10, device=device)

    ticker_mse_y36 = torch.zeros(10, device=device)
    ticker_rmse_y36 = torch.zeros(10, device=device)
    ticker_mse_y320 = torch.zeros(10, device=device)
    ticker_rmse_y320 = torch.zeros(10, device=device)

    for batch in loader:
        Xw = batch["Xw"].to(device)
        M = batch["mask"].to(device)
        Yw = batch["Yw"].to(device)

        preds10 = model.infer_with_unseen(Xw, M)  # (B,10,2)
        mask10 = ~torch.isnan(Yw).any(dim=-1)     # (B,10)
        Y10v = torch.nan_to_num(Yw, nan=0.0)

        mse, rmse = mse_rmse(preds10, Y10v, mask10)
        mse_sum += mse
        rmse_sum += rmse
        n += 1

        # === 分label统计（整体） ===
        mask_y36 = mask10
        mask_y320 = mask10
        Y_y36 = Y10v[..., 0:1]
        Y_y320 = Y10v[..., 1:2]
        P_y36 = preds10[..., 0:1]
        P_y320 = preds10[..., 1:2]

        mse_y36, rmse_y36 = mse_rmse(P_y36, Y_y36, mask_y36)
        mse_y320, rmse_y320 = mse_rmse(P_y320, Y_y320, mask_y320)

        mse_y36_sum += mse_y36
        mse_y320_sum += mse_y320
        rmse_y36_sum += rmse_y36
        rmse_y320_sum += rmse_y320

        # === Per-ticker统计 ===
        for a in range(10):
            vmask = mask10[:, a]
            if vmask.sum() == 0:
                continue
            Yv_a = Y10v[:, a, :]
            P_a = preds10[:, a, :]

            mse_a = masked_mse(P_a, Yv_a, vmask).item()
            mse_y36_a = masked_mse(P_a[:, 0:1], Yv_a[:, 0:1], vmask).item()
            mse_y320_a = masked_mse(P_a[:, 1:2], Yv_a[:, 1:2], vmask).item()

            ticker_mse[a] += mse_a
            ticker_rmse[a] += math.sqrt(max(mse_a, 0.0))
            ticker_mse_y36[a] += mse_y36_a
            ticker_rmse_y36[a] += math.sqrt(max(mse_y36_a, 0.0))
            ticker_mse_y320[a] += mse_y320_a
            ticker_rmse_y320[a] += math.sqrt(max(mse_y320_a, 0.0))
            ticker_cnt[a] += 1

    out = {
        "test_mse": mse_sum / max(n, 1),
        "test_rmse": rmse_sum / max(n, 1),
        "test_mse_y36": mse_y36_sum / max(n, 1),
        "test_rmse_y36": rmse_y36_sum / max(n, 1),
        "test_mse_y320": mse_y320_sum / max(n, 1),
        "test_rmse_y320": rmse_y320_sum / max(n, 1),
        "ticker_mse": (ticker_mse / ticker_cnt.clamp_min(1)).cpu().numpy(),
        "ticker_rmse": (ticker_rmse / ticker_cnt.clamp_min(1)).cpu().numpy(),
        "ticker_mse_y36": (ticker_mse_y36 / ticker_cnt.clamp_min(1)).cpu().numpy(),
        "ticker_rmse_y36": (ticker_rmse_y36 / ticker_cnt.clamp_min(1)).cpu().numpy(),
        "ticker_mse_y320": (ticker_mse_y320 / ticker_cnt.clamp_min(1)).cpu().numpy(),
        "ticker_rmse_y320": (ticker_rmse_y320 / ticker_cnt.clamp_min(1)).cpu().numpy(),
    }

    if verbose:
        print(f"\n[TEST] Overall MSE={out['test_mse']:.6f}, RMSE={out['test_rmse']:.6f}")
        print(f"       y36  MSE={out['test_mse_y36']:.6f}, RMSE={out['test_rmse_y36']:.6f}")
        print(f"       y320 MSE={out['test_mse_y320']:.6f}, RMSE={out['test_rmse_y320']:.6f}")
        print("📊 Per-ticker breakdown:")
        for a in range(10):
            print(f"  Ticker {a:02d} | MSE={out['ticker_mse'][a]:.6f}, RMSE={out['ticker_rmse'][a]:.6f} "
                  f"| y36={out['ticker_mse_y36'][a]:.6f}, y320={out['ticker_mse_y320'][a]:.6f}")
    return out

def fit_multi_ticker(
    model, dl_train, dl_val, dl_test,
    epochs=50, lr=3e-4, device="cuda",
    early_stop_patience=10, model_name="multi_ticker_full"
):
    result_file = f"results/{model_name}_result.txt"
    os.makedirs("results", exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_y36_rmse, best_y320_rmse = float("inf"), float("inf")
    best_y36_state, best_y320_state = None, None
    patience_y36, patience_y320 = 0, 0
    logs = []

    for ep in range(1, epochs + 1):
        # === training & evaluation ===
        tr = train_mt_one_epoch(model, dl_train, optimizer, device=device)
        va = eval_mt_val(model, dl_val, device=device)  # returns 4 metrics
        te = eval_mt_test(model, dl_test, device=device, verbose=False)

        logs.append({"epoch": ep, **tr, **va, **te})

        # === logging to console ===
        print(
            f"Epoch {ep:02d} | "
            f"train_loss={tr['train_loss']:.6f} | "
            f"val_y36_rmse={va['val_y36_rmse']:.6f}, "
            f"val_y320_rmse={va['val_y320_rmse']:.6f} | "
            f"test_y36_rmse={te['test_rmse_y36']:.6f}, "
            f"test_y320_rmse={te['test_rmse_y320']:.6f}"
        )

        # === append to file ===
        with open(result_file, "a") as f:
            f.write(f"\nEpoch {ep:02d}\n")
            f.write(f"TrainLoss={tr['train_loss']:.6f}\n")
            f.write(f"Val y36  MSE={va['val_y36_mse']:.6f}, RMSE={va['val_y36_rmse']:.6f}\n")
            f.write(f"Val y320 MSE={va['val_y320_mse']:.6f}, RMSE={va['val_y320_rmse']:.6f}\n")
            f.write(f"Test y36  MSE={te['test_mse_y36']:.6f}, RMSE={te['test_rmse_y36']:.6f}\n")
            f.write(f"Test y320 MSE={te['test_mse_y320']:.6f}, RMSE={te['test_rmse_y320']:.6f}\n")
            f.write("Per-ticker breakdown (MSE / RMSE | y36 / y320):\n")
            for a in range(10):
                f.write(
                    f"  Ticker {a:02d} | "
                    f"MSE={te['ticker_mse'][a]:.6f}, RMSE={te['ticker_rmse'][a]:.6f} | "
                    f"y36_mse={te['ticker_mse_y36'][a]:.6f}, y36_rmse={te['ticker_rmse_y36'][a]:.6f} | "
                    f"y320_mse={te['ticker_mse_y320'][a]:.6f}, y320_rmse={te['ticker_rmse_y320'][a]:.6f}\n"
                )
            f.write("-" * 80 + "\n")

        # === early stopping & best checkpoint ===
        improved_y36 = va["val_y36_rmse"] < best_y36_rmse - 1e-8
        improved_y320 = va["val_y320_rmse"] < best_y320_rmse - 1e-8

        if improved_y36:
            best_y36_rmse = va["val_y36_rmse"]
            best_y36_state = {k: v.detach().cpu() if hasattr(v, "detach") else v
                              for k, v in model.state_dict().items()}
            patience_y36 = 0
        else:
            patience_y36 += 1

        if improved_y320:
            best_y320_rmse = va["val_y320_rmse"]
            best_y320_state = {k: v.detach().cpu() if hasattr(v, "detach") else v
                               for k, v in model.state_dict().items()}
            patience_y320 = 0
        else:
            patience_y320 += 1

        if early_stop_patience and max(patience_y36, patience_y320) >= early_stop_patience:
            print(f"⏹ Early stopped at epoch {ep}")
            break

    # === evaluate both best checkpoints ===
    with open(result_file, "a") as f:
        f.write("\n[FINAL EVALUATIONS]\n")

    # --- Best y36 ---
    if best_y36_state is not None:
        model.load_state_dict(best_y36_state)
        final_y36 = eval_mt_test(model, dl_test, device=device, verbose=True)
        print(f"[BEST y36 MODEL] val_y36_rmse={best_y36_rmse:.6f} | test_y36_rmse={final_y36['test_rmse_y36']:.6f}")
        with open(result_file, "a") as f:
            f.write(f"\n[BEST y36 MODEL] Val_y36_RMSE={best_y36_rmse:.6f}\n")
            f.write(f"Test y36 MSE={final_y36['test_mse_y36']:.6f}, RMSE={final_y36['test_rmse_y36']:.6f}\n")
            f.write(f"Test y320 MSE={final_y36['test_mse_y320']:.6f}, RMSE={final_y36['test_rmse_y320']:.6f}\n")

    # --- Best y320 ---
    if best_y320_state is not None:
        model.load_state_dict(best_y320_state)
        final_y320 = eval_mt_test(model, dl_test, device=device, verbose=True)
        print(f"[BEST y320 MODEL] val_y320_rmse={best_y320_rmse:.6f} | test_y320_rmse={final_y320['test_rmse_y320']:.6f}")
        with open(result_file, "a") as f:
            f.write(f"\n[BEST y320 MODEL] Val_y320_RMSE={best_y320_rmse:.6f}\n")
            f.write(f"Test y36 MSE={final_y320['test_mse_y36']:.6f}, RMSE={final_y320['test_rmse_y36']:.6f}\n")
            f.write(f"Test y320 MSE={final_y320['test_mse_y320']:.6f}, RMSE={final_y320['test_rmse_y320']:.6f}\n")

    return model, logs




class WindowDataset(Dataset):
    """
    X: (A=10, T, D=17)  Y: (A=10, T, 2)
    Returns windows:
      Xw: (A=10, L, D), Yw: (A=10, 2), mask: (A=10, L, D) where True=valid(~NaN)
    NOTE:
      - For training model we will exclude A=last ticker outside this Dataset
        (we do that in collate or training step).
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, starts: np.ndarray, window: int):
        super().__init__()
        assert X.ndim == 3 and Y.ndim == 3
        self.X = X
        self.Y = Y
        self.starts = starts
        self.window = int(window)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        e = s + self.window
        Xw = self.X[:, s:e, :]                # (10, L, 17)
        Yw = self.Y[:, e-1, :]                # (10, 2) label at window end
        mask = ~np.isnan(Xw)                  # (10, L, 17)
        Xw = np.nan_to_num(Xw, nan=0.0)
        sample = {
            "Xw": torch.from_numpy(Xw).float(),
            "Yw": torch.from_numpy(Yw).float(),
            "mask": torch.from_numpy(mask).float()
        }
        return sample

def masked_mse(pred, target, mask=None):
    """
    pred, target: (..., 2)
    mask: (...,) or (...,2)  1=valid
    If mask is None, use all.
    """
    if mask is None:
        return F.mse_loss(pred, target)
    if mask.dim() == pred.dim() - 1:
        mask = mask.unsqueeze(-1).expand_as(pred)
    diff = (pred - target) ** 2
    diff = diff * mask
    denom = mask.sum().clamp_min(1.0)
    return diff.sum() / denom

def mse_rmse(pred, target, mask=None) -> Tuple[float, float]:
    with torch.no_grad():
        if mask is None:
            mse = F.mse_loss(pred, target).item()
        else:
            mse = masked_mse(pred, target, mask).item()
        rmse = math.sqrt(max(mse, 0.0))
    return mse, rmse



if __name__ == "__main__":
    # Ablation 参数控制
    # === Ablation switches ===
    # add_pos_embedding = False       # 时间编码
    # use_winsorize = True           # 分位截断
    # model_name = "multi_ticker_full"             # 可选: mlp, gru, tcn, xgboost, multi_ticker_transformer, multi_ticker_full
    ablations = list(itertools.product([False, True], [False, True]))  # (pos_emb, winsorize)

    for add_pos_embedding, use_winsorize in ablations:
        # === 数据加载 ===
        X_new, Y_new, train_indices, test_indices = load_and_preprocess_data(
            add_pos_embedding=add_pos_embedding, use_winsorize=use_winsorize
        )
        for model_name in [
            # "tcn",
            # "gru",
            # "xgboost",
            "multi_ticker_full",
            # "mlp",
            # "shared_transformer",
        ]:

            with open(f"results/{model_name}_result.txt", "a") as f:
                print(f"\n{'='*80}", file=f)
                print(f"🚀 Running {model_name} | pos_emb={add_pos_embedding} | winsorize={use_winsorize}", file=f)
                print(f"{'='*80}", file=f)
            # === 序列任务准备 (TCN/GRU/Transformer) ===
            if model_name in ["gru", "tcn", "shared_transformer", "multi_ticker_full"]:
                X_train_seq, Y_train_seq, X_test_seq, Y_test_seq, ticker_idx_test = \
                    make_sliding_window_dataset(X_new, Y_new, train_indices, test_indices,
                                                window=WINDOW, stride=STRIDE, feature_dim=9)
                print(f"🚀 Prepared sliding window dataset for {model_name.upper()} baseline.")
                print(f"📏 Shapes — X_train: {X_train_seq.shape}, Y_train: {Y_train_seq.shape} | ")
            if model_name in ['mlp']:
                print(f"🚀 Preparing data for {model_name.upper()} baseline...")

                # === Step 1: 截取前9维有效特征 ===
                X_feat = X_new[:, :, :9]
                Y_feat = Y_new

                # === Step 2: 按时间索引划分训练 / 测试 ===
                X_train = X_feat[:, train_indices, :].reshape(-1, 9)
                Y_train = Y_feat[:, train_indices, :].reshape(-1, 2)
                X_test  = X_feat[:, test_indices, :].reshape(-1, 9)
                Y_test  = Y_feat[:, test_indices, :].reshape(-1, 2)

                print(f"📏 Raw shapes — X_train: {X_train.shape}, X_test: {X_test.shape}")

                # === Step 3: 检查 NaN / Inf ===
                def check_nan_inf(name, arr):
                    n_nan = np.isnan(arr).sum()
                    n_inf = np.isinf(arr).sum()
                    if n_nan > 0 or n_inf > 0:
                        print(f"⚠️ {name} contains {n_nan} NaN / {n_inf} Inf → will be cleaned.")
                        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
                    return arr

                X_train = check_nan_inf("X_train", X_train)
                Y_train = check_nan_inf("Y_train", Y_train)
                X_test  = check_nan_inf("X_test",  X_test)
                Y_test  = check_nan_inf("Y_test",  Y_test)

                # === Step 4: 移除 NaN 样本（仅保留标签有效的）===
                mask_train = ~np.isnan(Y_train).any(axis=1)
                mask_test  = ~np.isnan(Y_test).any(axis=1)
                X_train, Y_train = X_train[mask_train], Y_train[mask_train]
                X_test, Y_test   = X_test[mask_test], Y_test[mask_test]

                print(f"✅ After filtering — X_train: {X_train.shape}, X_test: {X_test.shape}")
                run_mlp_baseline(X_train, Y_train, X_test, Y_test)



            elif model_name == 'shared_transformer':
                run_shared_transformer_baseline(X_train_seq, Y_train_seq, X_test_seq, Y_test_seq)



            elif model_name == "gru":
                run_gru_baseline(X_train_seq, Y_train_seq, X_test_seq, Y_test_seq)

            elif model_name == "tcn":
                run_tcn_baseline(X_train_seq, Y_train_seq, X_test_seq, Y_test_seq)

            elif model_name == "xgboost":
                run_xgboost_baseline(X_new, Y_new, train_indices, test_indices)

            elif model_name == "shared_transformer":
                run_shared_transformer_baseline(X_train_seq, Y_train_seq, X_test_seq, Y_test_seq)


            elif model_name == "multi_ticker_full":
                from models_multiticker_full import MultiTickerModel
                model = MultiTickerModel(in_dim=X_new.shape[-1]).to(DEVICE)
                dl_train, dl_val, dl_test, info = build_mt_loaders(
                    X_new, Y_new, train_indices, test_indices,
                    window=WINDOW, stride=STRIDE,
                    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
                )
                model, logs = fit_multi_ticker(
                    model, dl_train, dl_val, dl_test,
                    epochs=EPOCHS, lr=LR, device=DEVICE,
                    early_stop_patience=10, model_name="multi_ticker_full"
                )

            else:
                raise ValueError(f"❌ 未知模型名: {model_name}")
            gc.collect()
            torch.cuda.empty_cache()