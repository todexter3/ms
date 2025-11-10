# ==========================================
# dataset_utils.py
# 滑动窗口数据集 & 工具函数
# ==========================================
from config_setup import *

class WindowDataset(Dataset):
    def __init__(self, X, Y, starts, window):
        self.X, self.Y, self.starts, self.window = X, Y, starts, window
    def __len__(self): return len(self.starts)
    def __getitem__(self, idx):
        s, e = self.starts[idx], self.starts[idx]+self.window
        Xw = np.nan_to_num(self.X[:, s:e, :], nan=0.0)
        Yw = np.nan_to_num(self.Y[:, e-1, :], nan=0.0)
        mask = ~np.isnan(self.X[:, s:e, :])
        return {"Xw": torch.from_numpy(Xw).float(),
                "Yw": torch.from_numpy(Yw).float(),
                "mask": torch.from_numpy(mask).float()}

def make_sliding_window_dataset(X_new, Y_new, train_indices, test_indices, window=190, stride=5, feature_dim=9):
    asset_n, T, _ = X_new.shape
    X_seq_all, Y_seq_all, ticker_idx_all, time_idx_all = [], [], [], []
    for a in range(asset_n):
        Xa, Ya = X_new[a], Y_new[a]
        for s in range(0, T-window, stride):
            e = s+window
            if np.isnan(Ya[e-1]).any(): continue
            X_seq_all.append(Xa[s:e,:feature_dim])
            Y_seq_all.append(Ya[e-1])
            ticker_idx_all.append(a)
            time_idx_all.append(e-1)
    X_seq_all, Y_seq_all = np.stack(X_seq_all), np.stack(Y_seq_all)
    ticker_idx_all, time_idx_all = np.array(ticker_idx_all), np.array(time_idx_all)
    train_mask = time_idx_all <= train_indices[-1]
    test_mask  = (time_idx_all >= test_indices[0]) & (time_idx_all <= test_indices[-1])
    return X_seq_all[train_mask], Y_seq_all[train_mask], X_seq_all[test_mask], Y_seq_all[test_mask], ticker_idx_all[test_mask]
