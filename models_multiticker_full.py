# ==========================================
# models_multiticker_full.py
# 完整 MultiTickerModel 实现
# ==========================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from config_setup import *

# =========================
# Per-ticker MLP Imputer (0 & 6)
# =========================
class ImputerMLP(nn.Module):
    """
    Reconstruct feature[0] and feature[6] from the other 15 dims.
    Input: (B, 15)
    Output: (B, 2)
    """
    def __init__(self, in_dim=15, hidden=64, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# =========================
# Shared Transformer Encoder
# =========================
class SharedEncoder(nn.Module):
    """
    A single Transformer encoder block shared across tickers.
    Each ticker window (L, D) → (EMB_DIM) via:
      - token projection (D→EMB_DIM)
      - TransformerEncoder (L × EMB_DIM)
      - mean pooling over L
    """
    def __init__(self, in_dim=17, emb_dim=EMB_DIM, nhead=NHEAD, layers=ENC_LAYERS, ffdim=FF_DIM, dropout=DROPOUT):
        super().__init__()
        self.proj = nn.Linear(in_dim, emb_dim)
        layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead,
                                           dim_feedforward=ffdim, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEndcoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):  # x: (B, L, D)
        h = self.proj(x)
        h = self.encoder(h)
        h = h.mean(dim=1)
        return self.norm(h)


# =========================
# Per-ticker heads & attention scorers
# =========================
class TickerHead(nn.Module):
    """Simple linear head to 2-dim."""
    def __init__(self, emb_dim=EMB_DIM):
        super().__init__()
        self.fc = nn.Linear(emb_dim, 2)

    def forward(self, z):
        return self.fc(z)


class AttentionScorer(nn.Module):
    """
    For ticker a, compute weights for other tickers' embeddings.
    Input: other_embs (B, 8, E)
    Output: scores (B, 8, 2) to weight others' predictions (2)
    """
    def __init__(self, emb_dim=EMB_DIM, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )

    def forward(self, other_embs):  # (B, 8, E)
        return self.net(other_embs)


# =========================
# Full MultiTickerModel
# =========================
class MultiTickerModel(nn.Module):
    def __init__(self, in_dim=17, emb_dim=EMB_DIM):
        super().__init__()
        self.encoder = SharedEncoder(in_dim=in_dim, emb_dim=emb_dim)
        self.heads = nn.ModuleList([TickerHead(emb_dim) for _ in range(9)])
        self.att_scors = nn.ModuleList([AttentionScorer(emb_dim) for _ in range(9)])

        self.tgt_idx = [0, 6]
        self.src_idx = [i for i in range(in_dim) if i not in self.tgt_idx]
        self.imputers = nn.ModuleList([
            ImputerMLP(in_dim=len(self.src_idx)) for _ in range(9)
        ])

    def _impute_0_6(self, x_ticker, m_ticker, ticker_id):
        """
        Impute features [0,6] using ImputerMLP if partially missing.
        """
        B, L, D = x_ticker.shape
        first9_mask = m_ticker[..., :9]
        any_valid = first9_mask.sum(dim=-1) > 0
        all_valid = (first9_mask.sum(dim=-1) == 9)
        need_impute = (any_valid & (~all_valid))

        if need_impute.any():
            xin = x_ticker[..., self.src_idx][need_impute]
            pred = self.imputers[ticker_id](xin)
            x_ticker = x_ticker.clone()
            x_ticker[..., self.tgt_idx][need_impute] = pred
        return x_ticker

    def encode_tickers(self, Xw_9, M_9):
        B, A, L, D = Xw_9.shape
        z_list = []
        for a in range(A):
            x_a = Xw_9[:, a, :, :]
            m_a = M_9[:, a, :, :]
            x_a_imp = self._impute_0_6(x_a, m_a, a)
            z_list.append(self.encoder(x_a_imp))
        return torch.stack(z_list, dim=1)  # (B,9,E)

    def predict_tickers(self, z_9):
        B, A, E = z_9.shape
        preds_9 = []
        all_own = torch.stack([self.heads[a](z_9[:, a, :]) for a in range(A)], dim=1)  # (B,9,2)
        for a in range(A):
            idx = [i for i in range(A) if i != a]
            other_embs = z_9[:, idx, :]
            other_preds = all_own[:, idx, :]
            scores = self.att_scors[a](other_embs)
            agg = (scores * other_preds).sum(dim=1)
            preds_9.append(all_own[:, a, :] + agg)
        return torch.stack(preds_9, dim=1)

    def forward_train9(self, Xw_9, M_9):
        z_9 = self.encode_tickers(Xw_9, M_9)
        return self.predict_tickers(z_9)

    @torch.no_grad()
    def infer_with_unseen(self, Xw_10, M_10):
        """
        Handle unseen ticker #9 (use rank-3..7 mean ensemble).
        """
        B, A, L, D = Xw_10.shape
        assert A == 10
        z_list = []
        for a in range(9):
            x_a = self._impute_0_6(Xw_10[:, a], M_10[:, a], a)
            z_list.append(self.encoder(x_a))
        z9 = self.encoder(Xw_10[:, 9])
        z_10 = torch.stack(z_list + [z9], dim=1)

        preds_0_8 = []
        all_heads_0_8 = torch.stack([self.heads[a](z_10[:, a]) for a in range(9)], dim=1)
        for a in range(9):
            idx_others = [i for i in range(10) if i != a]
            other_embs = z_10[:, idx_others, :]
            other_preds = all_heads_0_8[:, [i for i in range(9) if i != a]]
            other_embs_use, other_preds_use = other_embs[:, :8], other_preds[:, :8]
            scores = self.att_scors[a](other_embs_use)
            agg = (scores * other_preds_use).sum(dim=1)
            preds_0_8.append(all_heads_0_8[:, a] + agg)
        preds_0_8 = torch.stack(preds_0_8, dim=1)
        

        pred9_list = []
        z_self = z_10[:, 9].unsqueeze(1)
        for k in range(9):
            head_k, att_k = self.heads[k], self.att_scors[k]
            own_k = head_k(z_self.squeeze(1))
            other_embs = z_10[:, :9]
            idx_drop = k
            other_embs_use = other_embs[:, [i for i in range(9) if i != idx_drop]]
            other_preds_use = torch.stack([self.heads[i](z_10[:, i]) for i in range(9) if i != idx_drop], dim=1)
            scores = att_k(other_embs_use)
            agg = (scores * other_preds_use).sum(dim=1)
            pred9_list.append(own_k + agg)
        pred9 = torch.stack(pred9_list, dim=1)
        pred9_sorted, _ = torch.sort(pred9, dim=1, descending=True)
        pred9_final = pred9_sorted[:, 2:7].mean(dim=1)
        preds_10 = torch.zeros(B, 10, 2, device=z_10.device)
        preds_10[:, :9] = preds_0_8
        preds_10[:, 9] = pred9_final
        return preds_10
