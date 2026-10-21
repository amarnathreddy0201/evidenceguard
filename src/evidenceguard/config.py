import os, random
from pathlib import Path
import numpy as np

SEED = 42
ROOT = Path(os.environ.get("EG_ROOT", "/content/drive/MyDrive/EvidenceGuard"))
DATA, INDEX, OUT, CKPT = ROOT/"data", ROOT/"index", ROOT/"outputs", ROOT/"ckpt"

def ensure_dirs():
    for p in (DATA, INDEX, OUT, CKPT):
        p.mkdir(parents=True, exist_ok=True)

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
