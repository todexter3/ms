# ==========================================
# config_setup.py
# 全局配置、路径、超参数、随机种子
# ==========================================
import os, math, contextlib, json, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, Dict, List

# ---------------------------------------------------
# Global paths
# ---------------------------------------------------
DATA_PATH = "/home/fuxi/heiyi/15min_sample.feather"

# ---------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------
SEED = 42
WINDOW = 38 * 5
STRIDE = 5
EMB_DIM = 128
NHEAD = 4
ENC_LAYERS = 2
FF_DIM = 256
DROPOUT = 0.1
BATCH_SIZE = 16
EPOCHS = 20
LR = 3e-4
NUM_WORKERS = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------
# Reproducibility
# ---------------------------------------------------
def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(SEED)
print(f"🚀 Using device: {DEVICE}")
