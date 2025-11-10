# ==========================================
# models.py
# 所有模型 (MLP, GRU, TCN)
# ==========================================
from config_setup import *

class SimpleMLP(nn.Module):
    def __init__(self, in_dim=9, hidden=64, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim))
    def forward(self, x): return self.net(x)

class GRURegressor(nn.Module):
    def __init__(self, in_dim=9, hidden=64, out_dim=2, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, out_dim)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        res = x if self.downsample is None else self.downsample(x)
        return out[:, :, :res.shape[2]] + res

class TCNRegressor(nn.Module):
    def __init__(self, in_dim=9, out_dim=2, hidden=64, levels=3, kernel_size=3):
        super().__init__()
        layers, ch_in = [], in_dim
        for i in range(levels):
            dilation = 2 ** i
            layers.append(TemporalBlock(ch_in, hidden, kernel_size, dilation))
            ch_in = hidden
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden, out_dim)
    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.network(x)
        out = out[:, :, -1]
        return self.fc(out)
    
class SharedTransf(nn.Module):
    """
    Shared Transformer baseline (adapted for generic training pipeline)
    - Compatible with (B, L, D) or (B, A, L, D)
    - Uses shared Transformer encoder + per-ticker linear heads
    - Adds optional global cross-ticker attention
    """
    def __init__(self, in_dim=17, emb_dim=128, n_tickers=10):
        super().__init__()
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.n_tickers = n_tickers

        self.proj = nn.Linear(in_dim, emb_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_dim, nhead=4, batch_first=True),
            num_layers=2
        )

        self.attn = nn.MultiheadAttention(emb_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(emb_dim)
        self.heads = nn.ModuleList([nn.Linear(emb_dim, 2) for _ in range(n_tickers)])

    def forward(self, X):
        """
        X can be:
          - (B, L, D): single asset sequence
          - (B, A, L, D): multi-asset batch
        """
        if X.ndim == 3:
            # === Case 1: (B, L, D) → emulate single-asset mode ===
            h = self.encoder(self.proj(X))      # (B, L, E)
            h = h.mean(dim=1)                   # (B, E)
            out = self.heads[0](h)              # use the first head (or share)
            return out                          # (B, 2)

        elif X.ndim == 4:
            # === Case 2: (B, A, L, D) ===
            B, A, L, D = X.shape
            X_flat = X.view(B*A, L, D)
            h = self.encoder(self.proj(X_flat))  # (B*A, L, E)
            h = h.mean(dim=1).view(B, A, -1)     # (B, A, E)
            attn_out, _ = self.attn(h, h, h)
            h = self.norm(h + attn_out)
            preds = torch.stack([self.heads[i](h[:, i, :]) for i in range(A)], dim=1)
            return preds  # (B, A, 2)

        else:
            raise ValueError(f"Unexpected input shape: {X.shape}")
