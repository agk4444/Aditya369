"""Build the hardened Qwen3-4B Kaggle notebook from the original 0.6B notebook."""
import json, copy

SRC = "qwen3-emotion-orig.ipynb"
DST = "aditya369.ipynb"

nb = json.load(open(SRC))
cells = nb["cells"]
assert len(cells) == 15, len(cells)

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}

def code(source):
    return {"cell_type": "code", "metadata": {},
            "source": source.splitlines(keepends=True),
            "outputs": [], "execution_count": None}

# --- Cell 0: title/plan -------------------------------------------------------
cells[0] = md("""# Aditya369 — Qwen3-4B Emotion Classifier (Kaggle free tier)

- **Base model:** `Qwen/Qwen3-4B` (Apache-2.0) — one `MODEL_ID` swap tries 1.7B (lighter) or 8B (heavier)
- **Data:** `dair-ai/emotion` — 16k/2k/2k, 6 classes, free
- **GPU:** T4 auto-assigned, 15GB VRAM, 12h session cap, 30 GPU-h/week
- **Method:** `AutoModelForSequenceClassification` + LoRA r=16
- **Budget:** ~3,000 steps x ~3.5 s/step ≈ **3-3.5 GPU-hours** (~11% of the weekly 30)

**VRAM on a T4 (4-bit NF4):** 2.0 GB weights + ~3 GB activations + ~1.5 GB overhead ≈ **7 GB / 15 GB** — comfortable headroom.

> **Smoke test first:** Cell 3 has `SMOKE_STEPS = 0`. Set it to `10`, run all, paste the `[smoke]` lines for diagnosis. Then set it back to `0` and run all for the full train.

> **Safety net:** set `PUSH_TO_HUB = True` in the config cell and attach your `HF_TOKEN` secret — every
> 500-step checkpoint is pushed to your Hub repo *during* training, so even a crashed run leaves usable
> checkpoints behind. Without the token the notebook runs exactly the same, just without Hub pushes.

> **Label order:** read from `features['label'].names` at runtime in Cell 5. The dataset card legend and the actual feature order disagree — hardcoding indices trains noise.
""")

# --- New cell after installs: secrets ----------------------------------------
secrets_cell = code('''# Secrets — best-effort (model + dataset are public/ungated, this only enables Hub upload)
import os
token = os.environ.get("HF_TOKEN")
if not token:
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        print("HF_TOKEN not attached — fine, everything here is public:", type(e).__name__)
if token:
    os.environ["HF_TOKEN"] = token
    from huggingface_hub import login
    login(token=token)
    print("HF login ok")
''')

# --- Cell 3: config -----------------------------------------------------------
cells[3] = code('''# Config — Qwen3-4B (swap MODEL_ID for 1.7B or 8B)
import os, torch, random, numpy as np

MODEL_ID = "Qwen/Qwen3-4B"       # "Qwen/Qwen3-1.7B" lighter (~2.5 GPU-h) | "Qwen/Qwen3-8B" heavier (~5.5 GPU-h)
DATASET  = "dair-ai/emotion"
MAX_LEN  = 160                      # audited in the length cell, not guessed
EPOCHS   = 3
LR       = 2e-4
SEED     = 42
SMOKE_STEPS = 0                     # set to 10 for a timed 10-step smoke test (no checkpoints saved)
if "8B" in MODEL_ID:                # 8B needs extra headroom on a 15GB T4
    BS, ACCUM = 4, 4                # effective batch 16, same as below
else:
    BS, ACCUM = 8, 2                # effective batch 16
USE_BF16 = torch.cuda.is_bf16_supported()
OUT = "/kaggle/working/aditya369"
PUSH_TO_HUB = False   # True (+ HF_TOKEN attached) -> every 500-step checkpoint is pushed
                      #   to your Hub repo DURING training, + merged model at the end
HUB_REPO = "agk4444/aditya369"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
os.makedirs(OUT, exist_ok=True)
print("model", MODEL_ID, "| per-device BS", BS, "x accum", ACCUM, "| bf16:", USE_BF16, "| smoke:", SMOKE_STEPS)
''')

# --- Cell 5 (index 6 in new layout): fix the redacted pad_token line ----------
# (handled by index after insertion; see below)

# --- Train cell: stepwise checkpoints + smoke + eval-kwarg pin ----------------
train_cell = code('''# Train — hardened: stepwise checkpoints, smoke flag, transformers v4/v5 eval-kwarg pin
import inspect, time, subprocess
from transformers import Trainer, TrainingArguments

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
# Filter every kwarg against the INSTALLED TrainingArguments signature and say loudly
# what was dropped or converted — a renamed/removed kwarg must never kill the run.
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
# Detect quantization by GROUND TRUTH (not string-matching): transformers'
# is_quantized flag, any quantization_method marker, or actual bitsandbytes
# param types in the model. Skip the move when quantized (already on GPU);
# otherwise delegate to the original.
try:
    # Idempotent patching: if a previous guard is already installed (cell
    # re-run, or an older notebook's cell ran in this kernel), unwrap to the
    # TRUE original first. Otherwise the new wrapper captures the old wrapper
    # and they recurse into each other (RecursionError). The original is bound
    # as a default arg, not a global, so re-running can never rebind it.
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
trainer = Trainer(model=model, args=TrainingArguments(**args),
                  train_dataset=tokd["train"], eval_dataset=tokd["validation"],
                  data_collator=collator, compute_metrics=compute_metrics)
if torch.cuda.is_available():
    print(f"[mem] CUDA allocated before train: {torch.cuda.memory_allocated()/1e9:.2f} GB")
# Belt-and-suspenders: the LoRA cell's model.gradient_checkpointing_enable()
# may not have propagated through the PEFT wrapper (14 GB at first backward =
# full activations, i.e. checkpointing silently off). Force it on every nested
# model object and VERIFY the base model reports it active.
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
trainer.train()
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

# --- Save cell: fix size comments for 4B --------------------------------------
save_cell = code('''# Save adapter + merged model
trainer.save_model(f"{OUT}/adapter")     # LoRA only: ~70MB for 4B r=16
tok.save_pretrained(f"{OUT}/adapter")

merged = trainer.model.merge_and_unload()  # fuse LoRA into base weights
merged.save_pretrained(f"{OUT}/merged", safe_serialization=True)  # ~2.5GB merged 4-bit (stays 4-bit; bnb forbids .to() casts)
tok.save_pretrained(f"{OUT}/merged")
print(os.listdir(OUT))
''')

# --- Optional Hub upload -------------------------------------------------------
hub_cell = code('''# Final Hub upload: merged model (checkpoints already pushed during training if enabled)
_HUB2 = bool(PUSH_TO_HUB and os.environ.get("HF_TOKEN"))
if _HUB2:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=HUB_REPO, exist_ok=True)
    api.create_repo(repo_id=HUB_REPO + "-lora", exist_ok=True)
    api.upload_folder(folder_path=f"{OUT}/merged", repo_id=HUB_REPO)
    api.upload_folder(folder_path=f"{OUT}/adapter", repo_id=HUB_REPO + "-lora")
    print("pushed merged model + adapter to", HUB_REPO)
else:
    print("hub upload skipped (set PUSH_TO_HUB=True with HF_TOKEN attached to enable)")
''')

# --- Assemble ------------------------------------------------------------------
# Original indices: 0 md, 1 probe, 2 installs, 3 config, 4 data, 5 tok+audit,
#                   6 tokenize, 7 model, 8 lora, 9 metrics, 10 train, 11 report,
#                   12 inference, 13 save, 14 md(swaps)
new_cells = []
new_cells.append(cells[0])          # 0 title
new_cells.append(cells[1])          # 1 probe
# Kaggle's image ships torchao 0.10.0; PEFT's LoRA dispatcher hard-requires
# torchao>=0.16.0 and raises ImportError on the old copy instead of skipping it.
# We quantize with bitsandbytes, never torchao — uninstall it so PEFT takes the
# normal path. (Uninstall AFTER the -U installs above, in case anything pulls it.)
_inst_src = "".join(cells[2]["source"]).replace(
    "import transformers, peft, datasets", "import transformers, peft")
_inst_src += ("\n!pip uninstall -y -q torchao\n"
              "print('torchao removed:', end=' ')\n"
              "try:\n"
              "    import torchao; print('STILL PRESENT', torchao.__version__)\n"
              "except ImportError:\n"
              "    print('gone')\n")
new_cells.append(code(_inst_src))    # 2 installs (drop unused datasets import, drop torchao)
new_cells.append(secrets_cell)      # 3 secrets (new)
new_cells.append(cells[3])          # 4 config (replaced above)
new_cells.append(cells[4])          # 5 data

tok_cell_src = "".join(cells[5]["source"])
# The EOS-as-pad line must be present and exact (verified at the byte level:
# the file was always correct; "<redacted>" only ever appeared in filtered
# tool output, never on disk).
_eos_line = "    tok.pad_token = tok." + "eos_token"
assert _eos_line in tok_cell_src, "pad_token line missing or changed upstream"
new_cells.append(code(tok_cell_src))  # 6 tokenizer + audit

new_cells.append(cells[6])          # 7 tokenize

# --- Model cell: k-bit prep (patched) ----------------------------------------
model_cell_src = "".join(cells[7]["source"])
# FIX 2026-10-02: 4-bit must be checked FIRST. The original had `if USE_BF16`
# first, and torch.cuda.is_bf16_supported() returns True on the T4, so the
# 4B model always loaded in BF16 (8.18GB) and never hit the 4-bit path —
# causing the backward-pass OOM. 4-bit (~2.5GB) is required for 4B on a T4.
_bf16_block = """if USE_BF16:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(id2label), dtype=torch.bfloat16, **load_kw)
elif torch.cuda.get_device_capability(0)[0] >= 7:
    # T4 (sm_75): 4-bit NF4 works with bitsandbytes>=0.45
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(id2label), quantization_config=bnb, **load_kw)"""
_fourbit_block = """if torch.cuda.get_device_capability(0)[0] >= 7:
    # 4-bit FIRST (T4 sm_75, bitsandbytes>=0.45): the 4B model is 8GB in BF16
    # and OOMs on a T4; 4-bit is ~2.5GB. BF16 is the fallback below.
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(id2label), quantization_config=bnb, **load_kw)
elif USE_BF16:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(id2label), dtype=torch.bfloat16, **load_kw)"""
assert _bf16_block in model_cell_src, "model load block changed upstream"
model_cell_src = model_cell_src.replace(_bf16_block, _fourbit_block)
_kbit_anchor = "        MODEL_ID, num_labels=len(id2label), quantization_config=bnb, **load_kw)"
assert _kbit_anchor in model_cell_src, "4-bit load line missing or changed upstream"
model_cell_src = model_cell_src.replace(
    _kbit_anchor,
    _kbit_anchor + "\n    from peft import prepare_model_for_kbit_training\n"
    "    model = prepare_model_for_kbit_training(model)\n"
    '    print("[mem] k-bit training prep done")',
)
new_cells.append(code(model_cell_src))  # 8 model (patched)

# --- LoRA cell: explicit grad checkpointing + memory audit (patched) ----------
lora_cell_src = "".join(cells[8]["source"])
lora_cell_src += '''
# Memory audit: 4-bit Qwen3-4B + LoRA r=16 should footprint ~2.5-4 GB. If this
# prints ~8 GB, quantization silently failed and the T4 will OOM on backward.
try:
    _fp = model.get_memory_footprint()
    print(f"[mem] model footprint: {_fp/1e9:.2f} GB")
    if _fp > 6e9:
        print("[mem] WARNING: footprint > 6 GB — quantization may not have applied; expect OOM")
except Exception as e:
    print("[mem] footprint check skipped:", type(e).__name__)
# Explicit: do not rely on the Trainer arg alone (version behavior varies) —
# without checkpointing, full activations (~10 GB) OOM the T4 on first backward.
try:
    model.gradient_checkpointing_enable()
    print("[mem] gradient checkpointing: ON")
except Exception as e:
    print("[mem] WARNING: gradient checkpointing not enabled:", type(e).__name__)
model.config.use_cache = False
'''
new_cells.append(code(lora_cell_src))  # 9 lora (patched)
new_cells.append(cells[9])          # 10 metrics
new_cells.append(train_cell)        # 11 train (replaced)
new_cells.append(cells[11])         # 12 report
new_cells.append(cells[12])         # 13 inference
new_cells.append(save_cell)         # 14 save (sizes fixed)
new_cells.append(hub_cell)          # 15 hub upload (new)
new_cells.append(cells[14])         # 16 dataset swaps

nb["cells"] = new_cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
json.dump(nb, open(DST, "w"), indent=1)
print("wrote", DST, "cells:", len(new_cells))
