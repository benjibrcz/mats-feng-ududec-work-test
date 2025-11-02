## TruthfulQA ICM (Algorithm 1) — Minimal Reimplementation

This repo contains a lean Python implementation of Algorithm 1 (ICM) from "Unsupervised Elicitation of Language Models" applied to TruthfulQA. The logical consistency fix step is intentionally ignored per the task spec.

### Setup

1) Python 3.10+

2) Install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3) Hyperbolic API (OpenAI-compatible): set env vars

```bash
export HYPERBOLIC_API_KEY=...   # your key
export HYPERBOLIC_BASE_URL=...  # e.g., https://api.hyperbolic.xyz/v1
```

Models used (provider-specific IDs; adjust if needed):
- Base: `meta-llama/Meta-Llama-3.1-405B`
- Chat: `meta-llama/Meta-Llama-3.1-405B-Instruct`

Adjust names if your provider uses different model IDs.

### Quickstart (main result)

Run end-to-end (pilot recommended first by lowering steps):

Main figure/files for submission:

- `results_best_200_steps.json`
- `results_best_200_steps.png`

Command used to generate the main result (provider IDs may differ):

```bash
python -m src.run \
  --icm_steps 200 \
  --icm_target_labels 128 \
  --alpha 40 \
  --context_cap 96 \
  --eval_k 64 \
  --eval_mode strict_text \
  --results results_best_200_steps.json \
  --figure results_best_200_steps.png
```

Outputs (default names unless overridden):
- `results.json` — accuracies for four bars (+ optional myth coherence if present)
- `results.png` — bar chart matching Figure 1 layout

Useful flags:
- `--icm_steps` total proposals; use ≈ batch size (e.g., 200)
- `--icm_target_labels` stop after labeling K items (e.g., 128)
- `--alpha` acceptance scale (e.g., 40–50)
- `--context_cap` ICM scoring context cap (<=0 means no cap)
- `--eval_k` eval shots cap for ICM/Golden (<=0 means no cap)
- `--eval_mode {auto|strict_text}` strict_text enforces a single True/False token
- `--icm_seed_myth` seed ICM with myth labels if the dataset provides them
- `--base_model` / `--chat_model`

### What’s implemented

- Data loader for provided 1/10 TruthfulQA train/test JSONs
- Zero-shot classification with robust True/False decoding
- Many-shot prompt builder (ICM and Golden labels)
- ICM search (mutual predictability only; no consistency fix)
- Four-bar evaluation and plotting (+ optional myth coherence metrics)

### Optional utilities

- Multiple seeds with error bars:
```bash
python -m src.run_seeds --seeds 0 1 2 \
  --icm_steps 200 --icm_target_labels 128 --alpha 40 \
  --context_cap 96 --eval_k 64 --eval_mode strict_text \
  --out_prefix results_seeds
```
- K-sweep (performance vs shots):
```bash
python -m src.run_sweep_k --k_list 8 16 32 64 128 \
  --icm_steps 300 --icm_target_labels 128 --alpha 40 \
  --context_cap 96 --seed 0 --eval_mode strict_text \
  --out_prefix results_k_sweep
```

### Notes

- We ignore the paper’s logical consistency fix (Algorithm 2) per instructions; `I(D)=0`.
- Prompts are cached by exact text; progress bars/timers included.
- Strict decoding (`--eval_mode strict_text`) helps avoid logprob/tokenization edge cases.

---

## Coherent Myths (optional dataset to show coherence ≠ truth)

We include a small synthetic dataset demonstrating that internal coherence can diverge from truth.

Generate:
```bash
python -m src.gen_coherent_myths --pairs_per_domain 24  # ~96 items total
```

Run (baseline):
```bash
python -m src.run \
  --train data/coherent_myths_train.json \
  --test  data/coherent_myths_test.json \
  --icm_steps 200 --icm_target_labels 128 --alpha 40 \
  --context_cap 96 --eval_k 64 --eval_mode strict_text \
  --results results_myths.json --figure results_myths.png
```

Run (myth‑seeded ICM, adversarial):
```bash
python -m src.run \
  --train data/coherent_myths_train.json \
  --test  data/coherent_myths_test.json \
  --icm_steps 200 --icm_target_labels 128 --alpha 40 \
  --context_cap 96 --eval_k 64 --eval_mode strict_text --icm_seed_myth \
  --results results_myths_mythseed.json --figure results_myths_mythseed.png
```

In results JSON, fields like "ICM (Base) vs Myth" report coherence with the myth rule labels; high coherence with lower truth accuracy illustrates coherence ≠ truth.

### Quick compare (Neutral vs Myth seed)

```bash
python -m src.run_myths_compare \
  --train data/coherent_myths_train.json \
  --test  data/coherent_myths_test.json \
  --icm_steps 200 --icm_target_labels 128 --alpha 40 \
  --context_cap 96 --eval_k 64 --eval_mode strict_text \
  --out_prefix results_myths_compare
```

Outputs a small 2×2-style artifact (JSON) and a scatter plot (`results_myths_compare.png`) showing truth accuracy vs myth coherence for Neutral vs Myth seeds.


