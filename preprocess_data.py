# ==========================================
# preprocess_data.py
# 数据加载、时间补全、周期特征、ablation接口
# ==========================================
from config_setup import *
from datetime import time

def winsorize_per_ticker(X, lower_q=0.01, upper_q=0.99):
    X_proc = X.copy()
    num_A, T, D = X.shape
    num_features = min(9, D)
    for f in range(num_A):
        for d in range(num_features):
            q_low = np.nanquantile(X[f, :, d], lower_q)
            q_high = np.nanquantile(X[f, :, d], upper_q)
            X_proc[f, :, d] = np.clip(X_proc[f, :, d], q_low, q_high)
    print(f"✅ Winsorized between {lower_q:.2%}–{upper_q:.2%}")
    return X_proc


def load_and_preprocess_data(add_pos_embedding=True, use_winsorize=True):
    """
    主入口：加载与预处理数据。
    参数：
        add_pos_embedding: 是否添加时间周期嵌入 (ablation 用)
    返回：
        X_new, Y_new, train_indices, test_indices
    """
    df = pd.read_feather(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    tickers = sorted(df["ticker"].unique())
    print(f"✅ Found {len(tickers)} tickers")

    # -------------------------------
    # 全局有效时间（仅工作日 × 38 时段）
    # -------------------------------
    valid_times = pd.to_datetime([
        "03:45","04:00","04:15","04:30","04:45","05:00",
        "05:30","05:45","06:00","06:15",
        "08:30","08:45","09:00","09:15","09:30","09:45",
        "16:00","16:15","16:30","16:45",
        "17:00","17:15","17:30","17:45",
        "18:00","18:15","18:30","18:45",
        "19:00","19:15","19:30","19:45",
        "20:00","20:15","20:30","20:45","21:00","21:15"
    ]).time

    GLOBAL_START = pd.Timestamp("2012-01-04 03:45:00", tz="UTC")
    GLOBAL_END   = pd.Timestamp("2014-12-30 21:15:00", tz="UTC")
    all_days = pd.date_range(GLOBAL_START.normalize(), GLOBAL_END.normalize(), freq="D", tz="UTC")
    valid_days = all_days[all_days.dayofweek < 5]

    valid_ts = pd.to_datetime([datetime.combine(d.date(), t) for d in valid_days for t in valid_times], utc=True)
    print(f"✅ 全局时间点数: {len(valid_ts)}")

    # -------------------------------
    # 对每个资产补全
    # -------------------------------
    def complete_single_ticker_global(ticker_df, valid_ts):
        ticker = ticker_df["ticker"].iloc[0]
        multi_index = pd.MultiIndex.from_product([valid_ts, [ticker]], names=["time", "ticker"])
        ticker_df_idxed = ticker_df.set_index(["time", "ticker"]).sort_index()
        ticker_full = ticker_df_idxed.reindex(multi_index).reset_index()
        ticker_full["time_idx"] = np.arange(len(valid_ts))
        return ticker_full

    dfs_full = []
    for ticker in tqdm(tickers, desc="补全资产"):
        dfs_full.append(complete_single_ticker_global(df[df["ticker"] == ticker], valid_ts))
    df_interp = pd.concat(dfs_full, ignore_index=True).sort_values(["ticker","time"]).reset_index(drop=True)
    print("🎯 全局补全完成")

    # -------------------------------
    # 构造特征与标签
    # -------------------------------
    feature_cols = ["main_ret_slp", "ret", "close_adj", "high_adj", "low_adj", "open_adj", "tr", "capvol0", "volume"]
    label_cols = ["y36","y320"]
    tickers = sorted(df_interp["ticker"].unique())
    time_index = sorted(df_interp["time_idx"].unique())

    multi_index = pd.MultiIndex.from_product([time_index, tickers], names=["time_idx", "ticker"])
    df_full = df_interp.set_index(["time_idx","ticker"]).reindex(multi_index).sort_index()
    X = df_full[feature_cols].to_numpy().reshape(len(time_index), len(tickers), len(feature_cols))
    Y = df_full[label_cols].to_numpy().reshape(len(time_index), len(tickers), len(label_cols))
    X_new, Y_new = np.transpose(X, (1,0,2)), np.transpose(Y, (1,0,2))
    print(f"✅ X={X_new.shape}, Y={Y_new.shape}")

    # -------------------------------
    # Ablation: 添加时间编码
    # -------------------------------
    if add_pos_embedding:
        print("🕒 添加周期时间编码 + 全局位置编码 ...")
        num_A, T, _ = X_new.shape
        t_idx = np.arange(T)[None, :]
        t_idx = np.repeat(t_idx, num_A, axis=0)

        def encode_period(cycle):
            return np.sin(2 * np.pi * t_idx / cycle), np.cos(2 * np.pi * t_idx / cycle)

        # 日/周/年周期编码
        day_sin, day_cos = encode_period(38)
        week_sin, week_cos = encode_period(38 * 5)
        year_sin, year_cos = encode_period(38 * 365)

        # 全局时间（0~T）的进度编码
        total_sin = np.sin(2 * np.pi * t_idx / T)
        total_cos = np.cos(2 * np.pi * t_idx / T)

        # 拼接所有编码
        X_new = np.concatenate([
            X_new,
            day_sin[..., None], day_cos[..., None],
            week_sin[..., None], week_cos[..., None],
            year_sin[..., None], year_cos[..., None],
            total_sin[..., None], total_cos[..., None],
        ], axis=-1)

        print(f"✅ 加入周期+全局编码后: X_new.shape = {X_new.shape}")
    # Ablation: 是否执行 winsorize
    if use_winsorize:
        X_new = winsorize_per_ticker(X_new)

    # -------------------------------
    # Train/Test 时间划分
    # -------------------------------
    times = df_interp["time"].drop_duplicates().sort_values().reset_index(drop=True)
    train_end = pd.Timestamp("2014-06-30 23:59:59", tz="UTC")
    test_start = pd.Timestamp("2014-07-01 00:00:00", tz="UTC")
    train_indices = np.where(times <= train_end)[0]
    test_indices = np.where((times >= test_start) & (times <= pd.Timestamp("2014-12-31", tz="UTC")))[0]
    print(f"✅ Train={len(train_indices)}, Test={len(test_indices)}")

    return X_new, Y_new, train_indices, test_indices
