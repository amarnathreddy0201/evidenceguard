import os, random
from pathlib import Path

SEED = 42


def _default_root() -> Path:
    """EG_ROOT if set; Drive folder on Colab; otherwise the folder that holds the repo (local runs)."""
    if os.environ.get("EG_ROOT"):
        return Path(os.environ["EG_ROOT"])
    if Path("/content/drive").exists():
        return Path("/content/drive/MyDrive/EvidenceGuard")
    main_path= Path(__file__).resolve().parents[3]   # .../Automated_Fact (contains data/averitec)
    print(f"main path : {main_path}")
    return main_path


ROOT = _default_root()
DATA, INDEX, OUT, CKPT = ROOT/"data", ROOT/"index", ROOT/"outputs", ROOT/"ckpt"

# Unzipped AVeriTeC dev knowledge store: one JSON-Lines file per claim, output_dev/<n>.json
STORE_DEV = Path(os.environ.get(
    "EG_STORE_DEV",
    DATA / "averitec_store" / "data_store" / "knowledge_store" / "dev_knowledge_store" / "output_dev"))


def ensure_dirs():
    for p in (DATA, INDEX, OUT, CKPT):
        p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
