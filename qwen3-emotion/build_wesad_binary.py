"""Build aditya369-wesad-binary.ipynb — wrist-only WESAD BINARY stress classifier for Kaggle.

Binary variant of build_wesad.py: stress (WESAD label 2) vs non-stress
(baseline 1 + amusement 3 merged, matching the WESAD paper's binary task).
Transient/meditation windows (0/4/5/6/7) are dropped.
"""
import json

DST = "aditya369-wesad-binary.ipynb"

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}

def code(source):
    return {"cell_type": "code", "metadata": {},
            "source": source.splitlines(keepends=True),
            "outputs": [], "execution_count": None}

cells = []

cells.append(md("""# Aditya369-WESAD-Binary — wrist wearable stress detector (Kaggle free tier)

- **Data:** WESAD — 15 subjects, wrist Empatica E4 (BVP 64Hz, EDA 4Hz, TEMP 4Hz, ACC 32Hz)
- **Task:** BINARY — stress (1) vs non-stress (0 = baseline + amusement merged, the WESAD paper's binary setup)
- **Model:** small 1D-CNN (~225K params) on 60s windows @ 32Hz, wrist only (the consumer-wearable story)
- **Protocol:** leave-one-subject-out, 15 folds — the honest eval; no subject leaks between train and test
- **Budget:** ~1 GPU-hour total on a T4 (each fold trains in 1-3 min)

> **Attach the dataset first:** Add-ons → Add input → search `orvile/wesad-wearable-stress-affect-detection-dataset`
> (fallback: `qiriro/wesad-stress-dataset`). The discovery cell fails loudly if it can't find the subject `.pkl` files.

> **Honest expectations:** the WESAD paper's own binary benchmark hits ~93% accuracy (stress vs non-stress).
> If this notebook reports 99%+, something leaked — check the subject split, not the model.

> **Safety net:** set `PUSH_TO_HUB = True` in the config cell and attach your `HF_TOKEN` secret —
> the best fold's weights, model code, metrics, and a model card are uploaded to
> `agk4444/aditya369-wesad-binary` when training finishes.
"""))

cells.append(code('''# Hardware probe
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
                     capture_output=True, text=True).stdout)
'''))

cells.append(code('''# Secrets — best-effort (dataset is public/ungated, this only enables Hub upload)
import os
token = os.environ.get("HF_TOKEN")
if not token:
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        print("HF_TOKEN not attached — Hub upload will be skipped:", type(e).__name__)
if token:
    os.environ["HF_TOKEN"] = token
    from huggingface_hub import login
    login(token=token)
    print("HF login ok")
'''))

cells.append(code('''# Imports (all preinstalled on the Kaggle image) — versions printed for the record
import torch, scipy, sklearn
import numpy as np
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
print("scipy", scipy.__version__, "| sklearn", sklearn.__version__, "| numpy", np.__version__)
'''))

cells.append(code('''# Config — BINARY: stress vs non-stress (baseline + amusement merged)
import os

CANDIDATE_SLUGS = ["orvile/wesad-wearable-stress-affect-detection-dataset",
                   "qiriro/wesad-stress-dataset"]
FS        = 32          # common rate: BVP 64->32, EDA/TEMP 4->32, ACC native 32
WIN_S     = 60          # window length, seconds
OVERLAP   = 0.5         # 50% overlap
CHANNELS  = ["BVP", "EDA", "TEMP", "ACC"]   # ACC expands to x/y/z -> 6 channels
# WESAD raw labels: 1=baseline, 2=stress, 3=amusement, 0/4/5/6/7=transient+meditation (dropped)
# Binary map: stress -> 1, baseline+amusement -> 0 (the paper's stress-vs-non-stress task)
LABEL_MAP = {0: "non-stress", 1: "stress"}
SEED      = 42
EPOCHS    = 30
BS        = 32
LR        = 1e-3
PATIENCE  = 7           # early stopping on val F1
SMOKE     = False       # True -> 3 subjects x 5 epochs, quick pipeline check
PUSH_TO_HUB = False   # True (+ HF_TOKEN attached) -> upload best fold + card to the Hub
HUB_REPO  = "agk4444/aditya369-wesad-binary"
OUT = "/kaggle/working/aditya369-wesad-binary"
os.makedirs(OUT, exist_ok=True)
np.random.seed(SEED); torch.manual_seed(SEED)
print("OUT", OUT, "| smoke:", SMOKE)
'''))

cells.append(code('''# Data discovery — find the subject .pkl files under /kaggle/input
import glob

def find_wesad():
    hits = []
    for root in glob.glob("/kaggle/input/*"):
        pkls = glob.glob(root + "/**/S*.pkl", recursive=True)
        # keep only real subject files S2..S17 (skip S12, which WESAD excludes)
        pkls = [p for p in pkls if p.split("/")[-1][1:].split(".")[0].isdigit()]
        if pkls:
            hits.append((root, sorted(pkls)))
    return hits

hits = find_wesad()
if not hits:
    raise SystemExit(
        "WESAD not found under /kaggle/input. Add-ons -> Add input -> search for\\n"
        "  orvile/wesad-wearable-stress-affect-detection-dataset\\n"
        "  (fallback: qiriro/wesad-stress-dataset), then re-run."
    )
DATA_ROOT, PKLS = hits[0]
print("using", DATA_ROOT, "| subjects:", len(PKLS))
print("e.g.", PKLS[0])
'''))

cells.append(code('''# Preprocess: resample wrist -> 32Hz, per-subject z-norm, 60s windows, BINARY labels
import pickle
from scipy.signal import resample_poly

WRIST_FS = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32}
CHEST_FS = 700  # label array runs at the chest rate

def load_subject(pkl_path):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f, encoding="latin1")  # py2 pickle
    wrist = d["signal"]["wrist"]
    label = np.asarray(d["label"]).ravel()
    return wrist, label

def resample_to_32(x, fs_in):
    if fs_in == FS:
        return np.asarray(x, dtype=np.float32).ravel()
    # polyphase: exact rational ratios only (64->32, 4->32)
    up, down = FS // np.gcd(FS, fs_in), fs_in // np.gcd(FS, fs_in)
    return resample_poly(np.asarray(x, dtype=np.float32).ravel(), up, down).astype(np.float32)

def subject_windows(pkl_path):
    wrist, label = load_subject(pkl_path)
    # stack channels at 32Hz: BVP, EDA, TEMP, ACCx3
    chans = [resample_to_32(wrist["BVP"], 64),
             resample_to_32(wrist["EDA"], 4),
             resample_to_32(wrist["TEMP"], 4)]
    acc = np.asarray(wrist["ACC"], dtype=np.float32)
    if acc.ndim == 2 and acc.shape[1] == 3:
        acc = acc.T  # -> (3, N)
    for ax in range(3):
        chans.append(resample_to_32(acc[ax], 32))
    n = min(len(c) for c in chans)
    sig = np.stack([c[:n] for c in chans])          # (6, N)
    sig = (sig - sig.mean(axis=1, keepdims=True)) / (sig.std(axis=1, keepdims=True) + 1e-8)  # per-subject z-norm
    win, step = WIN_S * FS, int(WIN_S * FS * (1 - OVERLAP))
    Xs, ys = [], []
    for s in range(0, n - win + 1, step):
        t0 = s / FS
        lab_win = label[int(t0 * CHEST_FS): int((t0 + WIN_S) * CHEST_FS)]
        if len(lab_win) == 0:
            continue
        maj = int(np.bincount(lab_win).argmax())    # majority vote over the window
        if maj == 2:
            y = 1                                   # stress
        elif maj in (1, 3):
            y = 0                                   # baseline + amusement -> non-stress
        else:
            continue                                # drop transient / meditation
        Xs.append(sig[:, s:s + win]); ys.append(y)
    subj = pkl_path.split("/")[-1].split(".")[0]
    return subj, np.stack(Xs).astype(np.float32), np.array(ys, dtype=np.int64)

all_subj, all_X, all_y = [], [], []
for p in PKLS:
    if SMOKE and len(all_subj) >= 3:
        break
    s, X, y = subject_windows(p)
    print(f"{s}: {len(X)} windows", {LABEL_MAP[int(k)]: int((y == k).sum()) for k in sorted(set(y.tolist()))})
    all_subj.append(s); all_X.append(X); all_y.append(y)
X_all = np.concatenate(all_X)          # (n_windows, 6, 1920)
y_all = np.concatenate(all_y)
print("total:", X_all.shape, "classes:", LABEL_MAP)
np.savez_compressed(f"{OUT}/windows.npz", X=X_all, y=y_all,
                    subj=np.array([s for s, x in zip(all_subj, all_X) for _ in range(len(x))]))
'''))

cells.append(code('''# Audit: class balance per subject + one example window per class
import matplotlib.pyplot as plt

subs = np.load(f"{OUT}/windows.npz")["subj"]
Xa = np.load(f"{OUT}/windows.npz")["X"]; ya = np.load(f"{OUT}/windows.npz")["y"]
print(f"{'subject':>8} {'n':>5} " + " ".join(f"{v[:10]:>10}" for v in LABEL_MAP.values()))
for s in sorted(set(subs.tolist())):
    m = subs == s
    print(f"{s:>8} {m.sum():>5} " + " ".join(f"{int((ya[m] == k).sum()):>10}" for k in LABEL_MAP))
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
for ax, k in zip(axes, LABEL_MAP):
    i = np.where(ya == k)[0][0]
    ax.plot(Xa[i, 1], lw=0.8)  # EDA channel: the strongest stress signal
    ax.set_title(f"{LABEL_MAP[k]} — EDA (z-scored, 60s @32Hz)")
    ax.set_ylabel("z")
plt.tight_layout(); plt.savefig(f"{OUT}/examples.png", dpi=100)
print("class totals:", {LABEL_MAP[k]: int((ya == k).sum()) for k in LABEL_MAP})
'''))

# --- Model source: defined once, reused for the notebook cell + model.py on the Hub ---
MODEL_SRC = '''import torch
import torch.nn as nn

class StressCNN(nn.Module):
    def __init__(self, n_ch=6, n_cls=2):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv1d(cin, cout, 7, padding=3), nn.BatchNorm1d(cout), nn.ReLU(),
                nn.Conv1d(cout, cout, 7, padding=3), nn.BatchNorm1d(cout), nn.ReLU(),
                nn.MaxPool1d(4))
        self.feat = nn.Sequential(block(n_ch, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(128, n_cls))
    def forward(self, x):
        return self.head(self.feat(x))'''

cells.append(code('''# Model: small 1D-CNN (~225K params), binary head — minutes per fold on a T4
''' + MODEL_SRC + f'''

model = StressCNN()
n_params = sum(p.numel() for p in model.parameters())
print(f"params: {{n_params:,}} | input: (6, {{WIN_S * FS}})")

# persist the model code now: inspect.getsource() cannot see classes defined in
# notebook cells, so the Hub upload reuses this exact source text instead
with open(f"{{OUT}}/model.py", "w") as _f:
    _f.write({MODEL_SRC!r})
print("wrote", f"{{OUT}}/model.py")
'''))

cells.append(code('''# Train: leave-one-subject-out, 15 folds, early stopping on a val split of the train subjects
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

device = "cuda" if torch.cuda.is_available() else "cpu"
subs = np.load(f"{OUT}/windows.npz")["subj"]
Xa = torch.from_numpy(np.load(f"{OUT}/windows.npz")["X"])
ya = torch.from_numpy(np.load(f"{OUT}/windows.npz")["y"])
uniq = sorted(set(subs.tolist()))
EP = 5 if SMOKE else EPOCHS
rng = np.random.RandomState(SEED)
fold_metrics, best = [], None

for fi, test_s in enumerate(uniq):
    te = subs == test_s
    tr_idx = np.where(~te)[0]
    # stratified 10% val split from the TRAIN subjects only (test subject never touched)
    val_idx = rng.choice(tr_idx, size=max(1, int(0.1 * len(tr_idx))), replace=False)
    trn_idx = np.setdiff1d(tr_idx, val_idx)
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=ya[trn_idx].numpy())
    w = torch.tensor(cw, dtype=torch.float32, device=device)
    tl = DataLoader(TensorDataset(Xa[trn_idx], ya[trn_idx]), batch_size=BS, shuffle=True)
    vl = DataLoader(TensorDataset(Xa[val_idx], ya[val_idx]), batch_size=BS * 4)
    sl = DataLoader(TensorDataset(Xa[te], ya[te]), batch_size=BS * 4)
    m = StressCNN().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss(weight=w)
    best_f1, bad, best_state = -1, 0, None
    for ep in range(EP):
        m.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)   # y already {0,1}
            opt.zero_grad(); crit(m(xb), yb).backward(); opt.step()
        m.eval(); pv, yv = [], []
        with torch.no_grad():
            for xb, yb in vl:
                pv += m(xb.to(device)).argmax(1).cpu().tolist(); yv += yb.tolist()
        f1 = f1_score(yv, pv, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, bad, best_state = f1, 0, {k: v.cpu() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    m.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    m.eval(); pt, yt = [], []
    with torch.no_grad():
        for xb, yb in sl:
            pt += m(xb.to(device)).argmax(1).cpu().tolist(); yt += yb.tolist()
    acc = accuracy_score(yt, pt)
    f1b = f1_score(yt, pt, average="binary", zero_division=0)
    mf1 = f1_score(yt, pt, average="macro", zero_division=0)
    fold_metrics.append({"subject": test_s, "n_test": len(yt), "acc": acc,
                         "f1": f1b, "macro_f1": mf1,
                         "epochs": ep + 1, "y_true": yt, "y_pred": pt})
    if best is None or f1b > best[0]:
        best = (f1b, test_s, best_state)
    print(f"fold {fi + 1}/{len(uniq)} [{test_s}] acc {acc:.3f} F1 {f1b:.3f} (ep {ep + 1})", flush=True)

import json
json.dump([{k: v for k, v in fm.items() if k not in ("y_true", "y_pred")} for fm in fold_metrics],
          open(f"{OUT}/fold_metrics.json", "w"), indent=1)
np.savez_compressed(f"{OUT}/fold_preds.npz",
                    **{f"{fm['subject']}_true": np.array(fm["y_true"], dtype=np.int64)
                       for fm in fold_metrics},
                    **{f"{fm['subject']}_pred": np.array(fm["y_pred"], dtype=np.int64)
                       for fm in fold_metrics})
torch.save(best[2], f"{OUT}/best_model.pt")
print("best fold:", best[1], f"F1 {best[0]:.3f}")
'''))

cells.append(code('''# Report: aggregate LOSO results + pooled confusion matrix
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score

fm = json.load(open(f"{OUT}/fold_metrics.json"))
acc = np.array([f["acc"] for f in fm]); f1 = np.array([f["f1"] for f in fm])
print(f"LOSO folds: {len(fm)}")
print(f"accuracy  {acc.mean():.3f} ± {acc.std():.3f}")
print(f"binary F1 {f1.mean():.3f} ± {f1.std():.3f}")
print(f"{'subject':>8} {'acc':>6} {'F1':>7}")
for f in fm:
    print(f"{f['subject']:>8} {f['acc']:>6.3f} {f['f1']:>7.3f}")

preds = np.load(f"{OUT}/fold_preds.npz")
keys = sorted(set(k[:-5] for k in preds.files if k.endswith("_true")))
yt = np.concatenate([preds[f"{s}_true"] for s in keys])
yp = np.concatenate([preds[f"{s}_pred"] for s in keys])
names = [LABEL_MAP[0], LABEL_MAP[1]]
cm = confusion_matrix(yt, yp, labels=[0, 1])
print("pooled per-class F1:", np.round(f1_score(yt, yp, average=None, zero_division=0), 3))
fig, ax = plt.subplots(figsize=(4.5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(2), names, rotation=20, ha="right"); ax.set_yticks(range(2), names)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.colorbar(im); ax.set_title("Pooled LOSO confusion matrix (true x pred)")
ax.set_xlabel("predicted"); ax.set_ylabel("true")
plt.tight_layout(); plt.savefig(f"{OUT}/cm.png", dpi=120)
'''))

cells.append(code('''# Save: label map + run card
import json
json.dump({str(k): v for k, v in LABEL_MAP.items()}, open(f"{OUT}/label_map.json", "w"), indent=1)
card = {
    "model": "StressCNN 1D-CNN (~225K params), wrist E4 only, binary head",
    "input": f"6 channels x {WIN_S * FS} @ {FS}Hz, 60s windows, 50% overlap, per-subject z-norm",
    "protocol": "leave-one-subject-out",
    "classes": LABEL_MAP,
    "artifacts": ["fold_metrics.json", "fold_preds.npz", "best_model.pt (best fold state_dict)",
                "label_map.json", "cm.png", "examples.png", "windows.npz"],
}
json.dump(card, open(f"{OUT}/run_card.json", "w"), indent=1)
print("saved:", sorted(__import__("os").listdir(OUT)))
'''))

cells.append(code('''# Hub upload: best fold weights + model code + metrics + card
_HUB = bool(PUSH_TO_HUB and os.environ.get("HF_TOKEN"))
if _HUB:
    from huggingface_hub import HfApi
    # OUT/model.py was written by the model cell from the exact class source

    # model card from the actual run metrics
    import json as _json
    fm = _json.load(open(f"{OUT}/fold_metrics.json"))
    acc = np.mean([x["acc"] for x in fm]); f1 = np.mean([x["f1"] for x in fm])
    rows = "\\n".join(
        f"| {x['subject']} | {x['acc']:.3f} | {x['f1']:.3f} | {x['n_test']} |" for x in fm)
    card_md = f"""---
library_name: pytorch
tags:
- stress-detection
- wesad
- wearable
- 1d-cnn
license: apache-2.0
---

# Aditya369-WESAD-Binary — wrist stress detector

by **AGK FIRE INC**

Small 1D-CNN (~225K params) on wrist Empatica E4 signals — BINARY
stress vs non-stress (baseline + amusement merged, the WESAD paper's
binary setup), leave-one-subject-out evaluated.

## Results (LOSO, {len(fm)} folds)

| metric    | mean |
|-----------|------|
| accuracy  | {acc:.3f} |
| binary F1 | {f1:.3f} |

| subject | acc | F1 | n_test |
|---------|-----|----|--------|
{rows}

## Usage

```python
import torch, json
from model import StressCNN

labels = json.load(open("label_map.json"))          # {{"0": "non-stress", "1": "stress"}}
model = StressCNN(n_ch=6, n_cls=2)
model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))
model.eval()
# x: (1, 6, 1920) float32 — 60s window @32Hz, per-subject z-scored,
#    channels [BVP, EDA, TEMP, ACCx, ACCy, ACCz]
with torch.no_grad():
    pred = model(x).argmax(1).item()                # 0=non-stress, 1=stress
```

## Training

- Data: WESAD, wrist only (BVP 64Hz, EDA/TEMP 4Hz, ACC 32Hz), resampled to 32Hz
- 60s windows, 50% overlap, per-subject z-norm, majority-vote labels
- Binary labels: stress (WESAD 2) -> 1; baseline (1) + amusement (3) -> 0
- Leave-one-subject-out, early stopping on a val split of train subjects
- `best_model.pt` = state_dict of the best fold

---

© 2026 AGK FIRE INC. Released under Apache 2.0.
"""
    with open(f"{OUT}/README.md", "w") as f:
        f.write(card_md)

    api = HfApi()
    api.create_repo(repo_id=HUB_REPO, exist_ok=True)
    for fn in ["best_model.pt", "model.py", "label_map.json", "run_card.json",
               "fold_metrics.json", "fold_preds.npz", "cm.png", "examples.png", "README.md"]:
        p = f"{OUT}/{fn}"
        if os.path.exists(p):
            api.upload_file(path_or_fileobj=p, path_in_repo=fn, repo_id=HUB_REPO)
            print("uploaded", fn)
    print("pushed to", HUB_REPO)
else:
    print("hub upload skipped (set PUSH_TO_HUB=True with HF_TOKEN attached to enable)")
'''))

cells.append(md("""## What next

- **Honest baseline set.** Compare any future change (features, transformer, chest fusion) against the LOSO
  numbers above — same protocol, same splits, or the comparison means nothing.
- **3-class variant** lives in `aditya369-wesad.ipynb` (baseline / stress / amusement) — harder, ~0.70 macro-F1
  is the honest target there.
- **GoEmotions / text track** stays in `aditya369.ipynb` — separate notebook, separate story.
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(DST, "w"), indent=1)
print("wrote", DST, "cells:", len(cells))
