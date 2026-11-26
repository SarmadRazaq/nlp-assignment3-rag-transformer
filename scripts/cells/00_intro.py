# %% [markdown]
# # NLP Assignment 3: Encoder → Retrieval → Decoder (RAG) on Amazon Reviews
#
# **Roll no.:** i24-7709
#
# A three-stage pipeline built entirely from scratch in PyTorch (no pretrained models, no
# `nn.Transformer` / `nn.MultiheadAttention` / `nn.TransformerEncoder`):
#
# 1. **Part A: Encoder.** A multi-task encoder-only Transformer predicts 3-class sentiment and a
#    derived *helpfulness* label, and produces a fixed-size review embedding.
# 2. **Part B: Retrieval.** Exact cosine top-k search over the saved training-set embeddings.
# 3. **Part C: Decoder.** A decoder-only (causal) Transformer generates a 1-2 sentence explanation
#    from `[review | predicted sentiment | predicted helpfulness | top-k retrieved reviews]`.
#
# ## How to run
# 1. Put the raw Amazon 5-core files in `Dataset/`: `Beauty_5.json`,
#    `Cell_Phones_and_Accessories_5.json`, `Sports_and_Outdoors_5.json`
#    (from https://nijianmo.github.io/amazon/index.html).
# 2. Install the dependencies: `pip install torch numpy pandas matplotlib seaborn scikit-learn`
#    (a CUDA build of torch is strongly recommended; this notebook ran on an RTX 3050 6 GB).
# 3. Run all cells top to bottom (**Kernel → Restart & Run All**).
#    * `RUN_TUNING = True` re-runs the hyperparameter sweeps (~adds 30-40 min on the GPU). With
#      `False`, the saved sweep logs in `results/` are loaded and shown instead.
#    * `TRAIN = False` skips training and loads the weights from `models/` instead.
#
# Outputs: model weights → `models/`, training-set embeddings, metrics, plots and logs → `results/`.

# %%
import json
import os
import random
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 200)

RUN_TUNING = True   # re-run hyperparameter sweeps (False: load saved logs from results/)
TRAIN = True        # train final models (False: load weights from models/)

SEED = 42
DATA_DIR = Path("Dataset")
MODELS_DIR, RESULTS_DIR = Path("models"), Path("results")
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything()
print("torch", torch.__version__, "| device:", DEVICE,
      torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")
