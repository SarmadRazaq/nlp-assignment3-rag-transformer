"""Assemble the submission notebook from the tested library modules + experiment cells.

The modules in src/ragpipe are the single source of truth (they are what tests/ cover).
This script inlines them into the notebook so it is self-contained and runs top to
bottom without the package installed.  Usage:  python scripts/build_notebook.py
"""
import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "ragpipe"
OUT = ROOT / "i24-7709-NLP-Assignment3.ipynb"
CELLS_DIR = ROOT / "scripts" / "cells"

MODULES = [
    ("data.py", "Dataset construction"),
    ("preprocess.py", "Preprocessing (cleaning, tokenizer, vocabulary)"),
    ("transformer.py", "Transformer building blocks (from scratch)"),
    ("models.py", "Encoder (Part A) and decoder (Part C) models"),
    ("retrieval.py", "Retrieval index (Part B)"),
    ("explain.py", "Reference explanations and decoder input template (Part C)"),
    ("training.py", "Training loops and metrics"),
]


def module_source(name):
    code = (SRC / name).read_text(encoding="utf8")
    return re.sub(r"^from \.\w* import .*\n", "", code, flags=re.M).strip()


def load_cells(path):
    """Cell files use '# %% [markdown]' / '# %%' separators (jupytext 'percent' style)."""
    cells = []
    for chunk in re.split(r"^# %%(.*)$", path.read_text(encoding="utf8"), flags=re.M)[1:]:
        cells.append(chunk)
    out = []
    for header, body in zip(cells[0::2], cells[1::2]):
        body = body.strip("\n")
        if "[markdown]" in header:
            md = "\n".join(re.sub(r"^# ?", "", l) for l in body.splitlines())
            out.append(nbf.v4.new_markdown_cell(md))
        else:
            out.append(nbf.v4.new_code_cell(body))
    return out


# --smoke: shrink data/epochs so the whole notebook can be dry-run in minutes (CPU ok)
SMOKE_SUBS = [("per_category=12000", "per_category=400"), ("epochs=15", "epochs=1"),
              ("epochs=12", "epochs=1"), ("epochs=4)", "epochs=1)"), ("epochs=3)", "epochs=1)"),
              ("N_GEN = 500", "N_GEN = 6"), ("min_count=20", "min_count=2"),
              ("for name, change in ENC_SWEEP:", "for name, change in ENC_SWEEP[:2]:"),
              ("for name, change in DEC_SWEEP:", "for name, change in DEC_SWEEP[:2]:")]


def main():
    import sys
    smoke = "--smoke" in sys.argv
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    cells = load_cells(CELLS_DIR / "00_intro.py")
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Library code\n\nAll components are implemented from scratch. These cells are "
        "generated from `src/ragpipe/*.py`, which are covered by the unit tests in `tests/` "
        "(causal/padding masks, preprocessing, retrieval, prompt template, generation)."))
    for fname, title in MODULES:
        cells.append(nbf.v4.new_markdown_cell(f"### 1.{MODULES.index((fname, title)) + 1} {title}  (`src/ragpipe/{fname}`)"))
        cells.append(nbf.v4.new_code_cell(module_source(fname)))
    for f in sorted(CELLS_DIR.glob("[0-9][0-9]_*.py"))[1:]:
        cells += load_cells(f)
    out = OUT
    if smoke:
        for c in cells:
            if c.cell_type == "code":
                for a, b in SMOKE_SUBS:
                    c.source = c.source.replace(a, b)
                c.source = c.source.replace('Path("models")', 'Path("smoke/models")').replace(
                    'Path("results")', 'Path("smoke/results")').replace("mkdir(exist_ok=True)", "mkdir(parents=True, exist_ok=True)")
        out = ROOT / "smoke.ipynb"
        (ROOT / "smoke").mkdir(exist_ok=True)
    nb.cells = cells
    nbf.write(nb, out)
    print(f"wrote {out.relative_to(ROOT)} with {len(cells)} cells")


if __name__ == "__main__":
    main()
