# Aditya369

by **AGK FIRE INC**

Qwen3-4B fine-tuned for emotion classification on the `dair-ai/emotion` dataset
(20k English short texts, 6 emotions). Trained with LoRA (r=16) in 4-bit on a
single Kaggle T4.

**Models on the Hub:**

- [agk4444/aditya369](https://huggingface.co/agk4444/aditya369) — merged 4-bit model
- [agk4444/aditya369-lora](https://huggingface.co/agk4444/aditya369-lora) — LoRA adapter (~70MB)

## Labels

| id | emotion  |
|----|----------|
| 0  | sadness  |
| 1  | joy      |
| 2  | love     |
| 3  | anger    |
| 4  | fear     |
| 5  | surprise |

## Results

Test-set evaluation (2k held-out examples):

| metric      | score  |
|-------------|--------|
| accuracy    | 0.935  |
| macro F1    | 0.891  |
| weighted F1 | 0.934  |

Per class:

| emotion  | precision | recall | f1    | n   |
|----------|-----------|--------|-------|-----|
| sadness  | 0.968     | 0.979  | 0.973 | 581 |
| joy      | 0.941     | 0.964  | 0.952 | 695 |
| love     | 0.856     | 0.786  | 0.820 | 159 |
| anger    | 0.942     | 0.942  | 0.942 | 275 |
| fear     | 0.905     | 0.893  | 0.899 | 224 |
| surprise | 0.810     | 0.712  | 0.758 | 66  |

Errors concentrate in the two smallest classes (`love`, `surprise`) and land on
semantically close emotions (love→joy, surprise→fear) — the same pairs humans
find ambiguous.

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("agk4444/aditya369")
tok = AutoTokenizer.from_pretrained("agk4444/aditya369")

labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
inputs = tok("I am so happy today!", return_tensors="pt")
pred = model(**inputs).logits.argmax(-1).item()
print(labels[pred])  # joy
```

Requires `transformers` and `bitsandbytes`.

## Training

- Base: `Qwen/Qwen3-4B` in 4-bit (bitsandbytes); LoRA r=16 on attention layers
- Data: `dair-ai/emotion` — 16k train / 2k val / 2k test
- 3 epochs (~1,500 steps), gradient checkpointing, single T4 on Kaggle
- Checkpoints pushed to the Hub during training; final merge + upload from the notebook

## Repo layout

```
qwen3-emotion/
├── build.py                  # regenerates aditya369.ipynb (source of truth)
├── aditya369.ipynb           # hardened Kaggle training notebook — press Run yourself
├── aditya369-wesad.ipynb     # separate track: WESAD wearable-stress CNN (different task)
├── build_wesad.py            # regenerates the WESAD notebook
├── aditya369-wesad-binary.ipynb  # WESAD binary stress-vs-rest (15-fold LOSO)
├── build_wesad_binary.py     # regenerates the binary notebook
├── aditya369-goemotions.ipynb    # GoEmotions multi-label Qwen3-4B (BCE, resume-from-Hub)
└── build_goemotions.py       # regenerates the GoEmotions notebook
```

`build.py` is the source of truth for the notebook — edit it, run
`python3 build.py`, and upload the regenerated `.ipynb` to Kaggle.

## WESAD track (separate task)

Wrist-wearable stress/affect classifier — different data, different model, different
story from the text classifier above.

- **Model:** small 1D-CNN (~225K params) on Empatica E4 wrist signals
  (BVP 64Hz, EDA/TEMP 4Hz, ACC 32Hz → 6 channels @32Hz, 60s windows)
- **Task:** 3-class — baseline / stress / amusement
- **Protocol:** leave-one-subject-out over 15 subjects (the honest eval)
- **Hub:** [agk4444/aditya369-wesad](https://huggingface.co/agk4444/aditya369-wesad)
  (weights, model code, metrics, plots)

Results (15-fold LOSO, ~10 min on a Kaggle T4):

| metric   | mean          |
|----------|---------------|
| accuracy | 0.669 ± 0.155 |
| macro F1 | 0.572 ± 0.171 |

Wide per-subject variance (macro-F1 0.22–0.89) — the known hard part of WESAD.

### Binary variant: stress vs rest (2026-10-03)

Same model and protocol, labels remapped to binary (stress vs non-stress —
baseline + amusement merged, matching the WESAD paper's binary task).

- **Hub:** [agk4444/aditya369-wesad-binary](https://huggingface.co/agk4444/aditya369-wesad-binary)
  (weights, model code, metrics, plots)

Results (15-fold LOSO, ~10 min on a Kaggle T4):

| metric   | mean  |
|----------|-------|
| accuracy | 0.871 |
| F1       | 0.781 |
| macro F1 | 0.845 |

Four subjects perfect (S4, S6, S9, S10). Below the paper's ~93% binary
benchmark — and the same subjects that tanked the 3-class run (S14, S17)
are the worst here too. The per-subject variance persists in binary:
some people's stress physiology just doesn't look like others'.

## Note

This repo previously held an early wearable-concept sketch (src/, docs/) built
on synthetic data. It was removed in favor of the trained model above; the old
code remains in git history.

---

© 2026 AGK FIRE INC. Released under Apache 2.0.
