"""Build aditya369-goemotions.ipynb from aditya369.ipynb (the hardened Qwen3-4B LoRA notebook).

Changes vs the base:
- Dataset: google/goemotions (simplified config) — 27 emotions + neutral, MULTI-LABEL.
- Labels: multi-hot float vectors; loss swapped to BCEWithLogitsLoss via a
  MultiLabelTrainer subclass; metrics -> per-label/micro/macro F1 + exact match.
- Resume: on start, list HUB_REPO for checkpoint-* dirs, download the newest,
  and pass it as resume_from_checkpoint. FORCE_FRESH=True (or no token / no
  checkpoints) starts from scratch.
- Everything else kept: 4-bit QLoRA r=16, HF_TOKEN dual-read, smoke flag,
  500-step Hub checkpoint pushes, version-drift guards, memory guards.
"""
import json

SRC = "aditya369.ipynb"
DST = "aditya369-goemotions.ipynb"

nb = json.load(open(SRC))
cells = nb["cells"]
assert len(cells) == 17, len(cells)

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}

def code(source):
    return {"cell_type": "code", "metadata": {},
            "source": source.splitlines(keepends=True),
            "outputs": [], "execution_count": None}

# --- Cell 0: title ------------------------------------------------------------
cells[0] = md("""# Aditya369-GoEmotions — Qwen3-4B multi-label emotion classifier (Kaggle free tier)

- **Base model:** `Qwen/Qwen3-4B` (Apache-2.0) — one `MODEL_ID` swap tries 1.7B (lighter) or 8B (heavier)
- **Data:** `google/goemotions`, config `simplified` — ~54k Reddit comments, 27 emotions + neutral, **multi-label**
- **GPU:** T4 auto-assigned, 15GB VRAM, 12h session cap, 30 GPU-h/week
- **Method:** `AutoModelForSequenceClassification` + LoRA r=16 + **BCEWithLogitsLoss** (sigmoid @ 0.5)
- **Budget:** ~8,100 steps x ~3.5 s/step ≈ **7-8 GPU-hours** (~25% of the weekly 30) — near the 12h session cap,
  so **resume-from-Hub-checkpoint is built in**: a killed run continues where it stopped

**VRAM on a T4 (4-bit NF4):** 2.0 GB weights + ~3 GB activations + ~1.5 GB overhead ≈ **7 GB / 15 GB** — comfortable headroom.

> **Smoke test first:** Cell 3 has `SMOKE_STEPS = 0`. Set it to `10`, run all, paste the `[smoke]` lines for diagnosis. Then set it back to `0` and run all for the full train.

> **Safety net:** set `PUSH_TO_HUB = True` in the config cell and attach your `HF_TOKEN` secret — every
> 500-step checkpoint is pushed to your Hub repo *during* training, so even a crashed run leaves usable
> checkpoints behind, and the next session auto-resumes from the newest one. Without the token the notebook
> runs exactly the same, just without Hub pushes (and without resume).

> **Label order:** the 28 GoEmotions labels are defined in paper order in Cell 4 and asserted at runtime
> (ids must be < 28, every split non-empty). Do not reorder without retraining.
""")

# --- Cell 4: config -----------------------------------------------------------
cells[4] = code('''# Config — Qwen3-4B multi-label GoEmotions (swap MODEL_ID for 1.7B or 8B)
import os, torch, random, numpy as np

MODEL_ID = "Qwen/Qwen3-4B"       # "Qwen/Qwen3-1.7B" lighter (~4 GPU-h) | "Qwen/Qwen3-8B" heavier (~12 GPU-h)
DATASET  = "google/goemotions"
DATA_CONFIG = "simplified"          # 27 emotions + neutral, multi-label
MAX_LEN  = 160                      # Cell 6 re-derives from the data (p99, cap 192)
EPOCHS   = 3
LR       = 2e-4
SEED     = 42
SMOKE_STEPS = 0                     # set to 10 for a timed 10-step smoke test (no checkpoints saved)
FORCE_FRESH = False                 # True -> ignore Hub checkpoints, train from scratch
if "8B" in MODEL_ID:                # 8B needs extra headroom on a 15GB T4
    BS, ACCUM = 4, 4                # effective batch 16, same as below
else:
    BS, ACCUM = 8, 2                # effective batch 16
USE_BF16 = torch.cuda.is_bf16_supported()
OUT = "/kaggle/working/aditya369-goemotions"
PUSH_TO_HUB = False   # True (+ HF_TOKEN attached) -> every 500-step checkpoint is pushed
                      #   to your Hub repo DURING training, + merged model at the end.
                      #   A later session auto-resumes from the newest checkpoint.
HUB_REPO = "agk4444/aditya369-goemotions"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
os.makedirs(OUT, exist_ok=True)
print("model", MODEL_ID, "| per-device BS", BS, "x accum", ACCUM, "| bf16:", USE_BF16, "| smoke:", SMOKE_STEPS)
''')

# --- Cell 5: data -------------------------------------------------------------
cells[5] = code('''# Cell 4 — GoEmotions data (multi-label) + label map
from datasets import load_dataset
raw = load_dataset(DATASET, DATA_CONFIG)
print(raw)

# 27 emotions + neutral, GoEmotions paper order (alphabetical, neutral last).
# Asserted at runtime below: ids must be < 28 and every split must be non-empty.
EMOTIONS = ["admiration", "amusement", "anger", "annoyance", "approval", "caring",
            "confusion", "curiosity", "desire", "disappointment", "disapproval",
            "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
            "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
            "remorse", "sadness", "surprise", "neutral"]
assert len(EMOTIONS) == 28
id2label = {i: n for i, n in enumerate(EMOTIONS)}
label2id = {n: i for i, n in id2label.items()}
N_LABELS = len(EMOTIONS)

for split in ("train", "validation", "test"):
    labs = raw[split]["labels"]
    assert len(labs) > 0, f"{split} is empty"
    mx = max((max(l) for l in labs if l), default=-1)
    assert mx < N_LABELS, f"{split}: label id {mx} out of range"
    multi = sum(1 for l in labs if len(l) > 1)
    print(f"{split}: {len(labs)} rows | max id {mx} | multi-label rows {multi}")
print(id2label)
''')

# --- Cell 7: tokenize (multi-hot) ---------------------------------------------
cells[7] = code('''# Cell 6 — tokenize + multi-hot labels (float vectors, one slot per emotion)
import numpy as np

def _multi_hot(id_list):
    v = [0.0] * N_LABELS
    for i in id_list:
        v[int(i)] = 1.0
    return v

def prep(batch):
    enc = tok(batch["text"], truncation=True, max_length=MAX_LEN)
    enc["labels"] = [_multi_hot(l) for l in batch["labels"]]
    return enc

cols = raw["train"].column_names
tokd = raw.map(prep, batched=True, remove_columns=cols)
# audit: labels survived as 28-dim float vectors
ex = tokd["train"][0]
print(tokd)
print("labels len:", len(ex["labels"]), "| sum:", sum(ex["labels"]),
      "| nonzero:", [EMOTIONS[i] for i, v in enumerate(ex["labels"]) if v > 0])
''')

# --- Cell 8: model — patch in problem_type ------------------------------------
model_src = "".join(cells[8]["source"])
_anchor = "model.config.label2id = label2id"
assert _anchor in model_src, "model config anchor changed upstream"
model_src = model_src.replace(
    _anchor,
    _anchor + '\nmodel.config.problem_type = "multi_label_classification"  # BCEWithLogitsLoss path; Trainer subclass makes it explicit')
cells[8] = code(model_src)

# --- Cell 10: metrics (multi-label) -------------------------------------------
cells[10] = code('''# Cell 9 — multi-label metrics: sigmoid @ 0.5 -> per-label / micro / macro F1
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

THRESH = 0.5

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= THRESH).astype(np.int32)
    labels = np.asarray(labels).astype(np.int32)
    return {
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "exact_match": accuracy_score(labels, preds),  # subset accuracy: all 28 slots right
    }
''')

# --- Cell 11: train — BCE loss + Hub resume + all hardening -------------------
cells[11] = code('''# Train — multi-label: BCEWithLogitsLoss, Hub resume, stepwise checkpoints, smoke flag
import inspect, time, subprocess, re
import torch
from transformers import Trainer, TrainingArguments

# --- explicit multi-label loss (not relying on problem_type inference) ---
class MultiLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        loss = torch.nn.BCEWithLogitsLoss()(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

# --- resume: newest checkpoint-* on the Hub repo, if any ---
# Checkpoints are pushed DURING training (hub_strategy="checkpoint"), so a
# session killed at the 12h cap leaves everything needed to continue.
# FORCE_FRESH=True (config cell) skips this and trains from scratch.
RESUME_CKPT = None
if os.environ.get("HF_TOKEN") and not FORCE_FRESH:
    try:
        from huggingface_hub import HfApi, snapshot_download
        _api = HfApi()
        _files = _api.list_repo_files(repo_id=HUB_REPO)
        _ckpts = sorted({f.split("/")[0] for f in _files if re.match(r"checkpoint-\\d+/", f)},
                        key=lambda c: int(c.split("-")[1]))
        if _ckpts:
            _latest = _ckpts[-1]
            _dest = f"{OUT}/resume"
            snapshot_download(repo_id=HUB_REPO, allow_patterns=[f"{_latest}/*"], local_dir=_dest)
            RESUME_CKPT = f"{_dest}/{_latest}"
            print(f"[resume] Hub has {_latest} -> resuming from {RESUME_CKPT}")
        else:
            print("[resume] no checkpoints on the Hub — fresh run")
    except Exception as e:
        print("[resume] Hub check failed — fresh run:", type(e).__name__, str(e)[:150])
else:
    print("[resume] skipped (no HF_TOKEN or FORCE_FRESH=True) — fresh run")

# transformers v5 renamed evaluation_strategy -> eval_strategy; use whichever exists
_EVAL_KW = ("eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
            else "evaluation_strategy")

args = dict(
    output_dir=f"{OUT}/ckpt",
    per_device_train_batch_size=BS,
    per_device_eval_batch_size=BS * 2,
    gradient_accumulation_steps=ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    logging_steps=25,
    save_strategy="steps",          # never go a full run without a checkpoint on disk
    save_steps=500,
    save_total_limit=2,
    bf16=USE_BF16,
    fp16=not USE_BF16,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    group_by_length=True,
    seed=SEED,
    report_to=[],
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
)
args[_EVAL_KW] = "steps"
if SMOKE_STEPS:
    args["max_steps"] = int(SMOKE_STEPS)
    print(f"[smoke] SMOKE_STEPS={SMOKE_STEPS}: {SMOKE_STEPS} timed steps, no checkpoints saved")

# Eval on the same cadence as checkpoints: transformers v5 raises ValueError when
# load_best_model_at_end=True and eval_strategy != save_strategy. Full run:
# eval+save every 500 steps. Smoke run (max 10 steps): never reached, no eval.
args["eval_steps"] = args["save_steps"]

# Safety net: push every checkpoint to the Hub AS training runs, so a dead
# run still leaves usable checkpoints behind (not just the final upload cell).
_HUB = bool(PUSH_TO_HUB and os.environ.get("HF_TOKEN"))
if _HUB:
    args.update(push_to_hub=True, hub_model_id=HUB_REPO, hub_strategy="checkpoint")
    print(f"[hub] every 500-step checkpoint -> {HUB_REPO}")
elif PUSH_TO_HUB:
    print("[hub] PUSH_TO_HUB=True but no HF_TOKEN attached — running without Hub pushes")

# --- version-drift guard (the Kaggle image's transformers rejected warmup_ratio) ---
import transformers as _tf
_SIG = inspect.signature(TrainingArguments.__init__).parameters
if "warmup_ratio" in args and "warmup_ratio" not in _SIG:
    if "warmup_steps" in _SIG:
        _total = int(args.get("max_steps") or (len(tokd["train"]) // (BS * ACCUM) * EPOCHS))
        args["warmup_steps"] = max(1, int(0.05 * _total))
        print(f"[compat] transformers {_tf.__version__} has no warmup_ratio -> warmup_steps={args['warmup_steps']}")
    else:
        print(f"[compat] WARNING: transformers {_tf.__version__} knows neither warmup_ratio nor warmup_steps — training without warmup")
    del args["warmup_ratio"]
for _k in [k for k in list(args) if k not in _SIG]:
    print(f"[compat] dropping unsupported TrainingArguments kwarg: {_k} (transformers {_tf.__version__})")
    del args[_k]
print(f"[compat] transformers {_tf.__version__}: {len(args)} kwargs accepted")
# Failsafe: this Trainer's getattr(model, "quantization_method") check does not
# see through the PEFT wrapper, so it calls model.to(device) — and .to() on
# 4-bit params dequantizes them to fp16 (≈2.5 GB -> 8 GB), OOMing in __init__.
try:
    _true_orig = getattr(Trainer._move_model_to_device, "_agk_true_orig", None)
    if _true_orig is None:
        _true_orig = Trainer._move_model_to_device
    def _is_quantized_deep(m):
        _seen = set()
        _stack = [m]
        while _stack:
            _b = _stack.pop()
            if id(_b) in _seen:
                continue
            _seen.add(id(_b))
            if getattr(_b, "is_quantized", False):
                return f"is_quantized via {type(_b).__name__}"
            _qm = getattr(_b, "quantization_method", None)
            if _qm is not None:
                return f"quantization_method={_qm} via {type(_b).__name__}"
            for _a in ("base_model", "model"):
                _nxt = getattr(_b, _a, None)
                if _nxt is not None and _nxt is not _b:
                    _stack.append(_nxt)
        try:
            import bitsandbytes as _bnb
            for _p in m.parameters():
                if isinstance(_p, (_bnb.nn.Params4bit, _bnb.nn.Int8Params)):
                    return f"bitsandbytes param type {type(_p).__name__}"
        except Exception:
            pass
        return None
    def _bnb_safe_move(self, model, device, _orig=_true_orig):
        _why = _is_quantized_deep(model)
        print(f"[mem] quantized check: {_why if _why else 'no quantization signal'}")
        if _why:
            print("[mem] quantized model — skipping Trainer's .to(device)")
            return model
        return _orig(self, model, device)
    _bnb_safe_move._agk_true_orig = _true_orig
    Trainer._move_model_to_device = _bnb_safe_move
    print("[mem] Trainer device-move guard installed")
except AttributeError:
    print("[mem] Trainer has no _move_model_to_device — nothing to guard")
trainer = MultiLabelTrainer(model=model, args=TrainingArguments(**args),
                            train_dataset=tokd["train"], eval_dataset=tokd["validation"],
                            data_collator=collator, compute_metrics=compute_metrics)
if torch.cuda.is_available():
    print(f"[mem] CUDA allocated before train: {torch.cuda.memory_allocated()/1e9:.2f} GB")
# Belt-and-suspenders: force gradient checkpointing on every nested model object
# and VERIFY the base model reports it active.
def _force_gc(m):
    _seen, _stack = set(), [m]
    while _stack:
        _b = _stack.pop()
        if id(_b) in _seen:
            continue
        _seen.add(id(_b))
        try:
            _b.gradient_checkpointing_enable()
        except Exception:
            pass
        for _a in ("base_model", "model"):
            _n = getattr(_b, _a, None)
            if _n is not None and _n is not _b:
                _stack.append(_n)
_force_gc(model)
_gc_on = getattr(getattr(getattr(model, "base_model", model), "model", model),
                 "gradient_checkpointing", None)
if _gc_on is None:
    _gc_on = getattr(model, "gradient_checkpointing", "unknown")
print(f"[mem] gradient checkpointing verified: {_gc_on}")
model.config.use_cache = False
t0 = time.time()
trainer.train(resume_from_checkpoint=RESUME_CKPT)
dt = time.time() - t0
n_steps = int(SMOKE_STEPS) if SMOKE_STEPS else len(tokd["train"]) // (BS * ACCUM) * EPOCHS
print(f"[smoke] {n_steps} steps in {dt:.0f}s = {dt/max(n_steps,1):.1f} s/step")
print(subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                      "--format=csv"], capture_output=True, text=True).stdout)
if SMOKE_STEPS:
    print("[smoke] done — paste the [smoke] lines for diagnosis, then set SMOKE_STEPS=0 and run all for the full train.")
    raise SystemExit
print(trainer.evaluate(tokd["test"]))
''')

# --- Cell 12: report (multi-label) --------------------------------------------
cells[12] = code('''# Cell 11 — multi-label report: per-emotion F1 + micro/macro
import matplotlib.pyplot as plt

pred = trainer.predict(tokd["test"])
logits, y = pred.predictions, np.asarray(pred.label_ids).astype(int)
probs = 1.0 / (1.0 + np.exp(-logits))
yh = (probs >= THRESH).astype(int)

per_f1 = f1_score(y, yh, average=None, zero_division=0)
micro = f1_score(y, yh, average="micro", zero_division=0)
macro = f1_score(y, yh, average="macro", zero_division=0)
print(f"micro-F1 {micro:.4f} | macro-F1 {macro:.4f} | exact-match "
      f"{(yh == y).all(axis=1).mean():.4f}")
sup = y.sum(axis=0)
print(f"{'emotion':>13} {'F1':>6} {'support':>8}")
for i, name in enumerate(EMOTIONS):
    print(f"{name:>13} {per_f1[i]:>6.3f} {int(sup[i]):>8}")

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(range(len(EMOTIONS)), per_f1)
ax.set_xticks(range(len(EMOTIONS)), EMOTIONS, rotation=60, ha="right")
ax.set_ylabel("F1"); ax.set_title(f"Per-emotion F1 (GoEmotions test, thresh {THRESH})")
plt.tight_layout(); plt.savefig(f"{OUT}/per_label_f1.png", dpi=120)
''')

# --- Cell 13: inference (sigmoid top-k) ---------------------------------------
cells[13] = code('''# Cell 12 — inference: sigmoid + top-k emotions per text
import torch
clf = trainer.model.eval()

def predict(texts, k=3, bs=32):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                truncation=True, max_length=MAX_LEN).to(clf.device)
        with torch.no_grad():
            probs = torch.sigmoid(clf(**b).logits).cpu().tolist()
        for p in probs:
            top = sorted(range(len(p)), key=lambda j: p[j], reverse=True)[:k]
            out.append([(EMOTIONS[j], round(p[j], 3)) for j in top])
    return out

tests = ["i can go from feeling so hopeless to so damned hopeful just from being around someone who cares",
         "this absolutely ruined my entire week",
         "i am so nervous about tomorrow i cannot sleep",
         "she planned a whole surprise party for me i am shocked"]
for t, p_ in zip(tests, predict(tests)):
    print(t[:70])
    print("   ", p_)
''')

# --- Cell 16: closing md -------------------------------------------------------
cells[16] = md("""## What next

- **GoEmotions is the multi-label track.** The per-emotion F1 chart above shows where the model is weak —
  small classes (grief, pride, relief) usually lag; that's the data, not the code.
- **Threshold tuning:** 0.5 is a starting point, not gospel. Sweep `THRESH` on the validation split and
  watch macro-F1 — rare emotions often want lower thresholds.
- **Single-label variant:** take `argmax` of the first emotion per row and train the `aditya369.ipynb`
  cross-entropy path for a direct comparison.
- **Resume chain:** if the 12h cap kills a run, start a fresh session with the same `HUB_REPO` and
  `PUSH_TO_HUB=True` — it auto-resumes from the newest Hub checkpoint. `FORCE_FRESH=True` restarts clean.
""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
json.dump(nb, open(DST, "w"), indent=1)
print("wrote", DST, "cells:", len(cells))
