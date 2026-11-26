# NLP Assignment 3: From-scratch RAG pipeline on Amazon Reviews

An **encoder → retrieval → decoder** system built entirely from scratch in PyTorch. It uses no pretrained models and none of `nn.Transformer`, `nn.MultiheadAttention` or `nn.TransformerEncoder`.

| Stage | What it does |
|---|---|
| **A. Encoder** | Multi-task Transformer encoder: 3-class sentiment + *helpfulness* (derived task) + a 256-d review embedding |
| **B. Retrieval** | Exact cosine top-k search over the saved training-set embeddings |
| **C. Decoder** | Causal (GPT-style) Transformer that generates a 1-2 sentence explanation from `review + predicted labels + top-k retrieved reviews` |

## Repository layout

```
i24-7709-NLP-Assignment3.ipynb   # submission notebook (self-contained, run top to bottom)
src/ragpipe/                     # library code (source of truth; inlined into the notebook)
tests/                           # unit tests: masks, preprocessing, retrieval, template, generation
scripts/build_notebook.py        # regenerates the notebook from src/ + scripts/cells/
scripts/cells/                   # experiment cells (percent format)
models/                          # trained weights: encoder.pt, decoder_rag.pt, decoder_base.pt
results/                         # embeddings, metrics, tuning logs, plots, generations
report/                          # written report
```

## Running

1. Download the 5-core files from <https://nijianmo.github.io/amazon/index.html> and place them in `Dataset/`:
   `Beauty_5.json`, `Cell_Phones_and_Accessories_5.json`, `Sports_and_Outdoors_5.json`.
2. `pip install -r requirements.txt` (install a CUDA build of torch for GPU training).
3. Open `i24-7709-NLP-Assignment3.ipynb` and **Restart & Run All**.
   Set `RUN_TUNING = False` to skip the sweeps (the logs are loaded from `results/`), and `TRAIN = False` to load the weights from `models/`.

Tests: `python -m pytest`. Rebuild the notebook after editing `src/` or `scripts/cells/`: `python scripts/build_notebook.py`.
