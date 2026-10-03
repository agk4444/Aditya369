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
├── build.py               # regenerates aditya369.ipynb (source of truth)
├── aditya369.ipynb        # hardened Kaggle training notebook — press Run yourself
├── aditya369-wesad.ipynb  # separate track: WESAD wearable-stress CNN (different task)
└── build_wesad.py         # regenerates the WESAD notebook
```

`build.py` is the source of truth for the notebook — edit it, run
`python3 build.py`, and upload the regenerated `.ipynb` to Kaggle.

## Note

This repo previously held an early wearable-concept sketch (src/, docs/) built
on synthetic data. It was removed in favor of the trained model above; the old
code remains in git history.

---

© 2026 AGK FIRE INC. Released under Apache 2.0.
