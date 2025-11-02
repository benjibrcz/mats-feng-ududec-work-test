import json
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from tqdm import tqdm

from .data_loader import Example
from .hyperbolic_client import HyperbolicClient
from .prompts import build_many_shot_prompt, build_zero_shot_prompt


def accuracy(preds: List[int], labels: List[int]) -> float:
    correct = sum(1 for p, y in zip(preds, labels) if p == y)
    return correct / max(1, len(labels))


def coherence(preds: List[int], examples: Sequence[Example]) -> float:
    # Fraction of preds matching myth_label where provided
    pairs = [(p, ex.myth_label) for p, ex in zip(preds, examples) if ex.myth_label is not None]
    if not pairs:
        return 0.0
    return sum(1 for p, m in pairs if p == m) / len(pairs)


def predict_zero_shot(
    client: HyperbolicClient,
    model: str,
    test: Sequence[Example],
    desc: str | None = None,
    mode: str = "auto",  # "auto" uses logprobs-first; "strict_text" forces text-only
) -> List[int]:
    preds: List[int] = []
    iterator = tqdm(test, desc=desc) if desc else test
    for ex in iterator:
        prompt = build_zero_shot_prompt(ex)
        if mode == "strict_text":
            preds.append(client.decide_true_false_strict(model, prompt))
        else:
            logp_true, logp_false = client.score_true_false(model, prompt)
            preds.append(1 if logp_true >= logp_false else 0)
    return preds


def predict_many_shot(
    client: HyperbolicClient,
    model: str,
    shots: Sequence[Example],
    test: Sequence[Example],
    context_cap: int | None = 160,
    seed: int = 0,
    mode: str = "auto",
) -> List[int]:
    rng = random.Random(seed)
    preds: List[int] = []
    shots_list = list(shots)
    iterator = tqdm(test, desc=f"Eval {model}")
    for ex in iterator:
        if context_cap is not None and context_cap > 0 and len(shots_list) > context_cap:
            ctx = rng.sample(shots_list, context_cap)
        else:
            ctx = shots_list
        prompt = build_many_shot_prompt(ctx, ex)
        if mode == "strict_text":
            preds.append(client.decide_true_false_strict(model, prompt))
        else:
            logp_true, logp_false = client.score_true_false(model, prompt)
            preds.append(1 if logp_true >= logp_false else 0)
    return preds


def build_icm_shots(train: Sequence[Example], icm_labels: Dict[int, int]) -> List[Example]:
    shots: List[Example] = []
    for ex in train:
        if ex.example_id in icm_labels:
            shots.append(replace(ex, label=icm_labels[ex.example_id]))
    return shots


def run_full_evaluation(
    client: HyperbolicClient,
    base_model: str,
    chat_model: str,
    train: Sequence[Example],
    test: Sequence[Example],
    icm_labels: Dict[int, int],
    seed: int = 0,
    eval_k: int | None = None,
    eval_mode: str = "auto",
) -> Tuple[Dict[str, float], Dict[str, List[int]]]:
    Path("debug").mkdir(exist_ok=True)
    # Zero-shot Base (no in-context examples)
    t0 = time.time()
    # dump one exemplar prompt
    if len(test) > 0:
        zb_prompt = build_zero_shot_prompt(test[0])
        Path("debug/zero_shot_base.txt").write_text(zb_prompt)
    zs_base_preds = predict_zero_shot(client, base_model, test, desc="Zero-shot (Base)", mode=eval_mode)
    print(f"Zero-shot (Base) completed in {time.time()-t0:.1f}s")
    zs_base_acc = accuracy(zs_base_preds, [ex.label for ex in test])

    # Zero-shot Chat (same required model), with simple retries for transient errors
    last_err: Exception | None = None
    zs_chat_preds: List[int] = []
    t1 = time.time()
    for _ in range(5):
        try:
            if len(test) > 0:
                zc_prompt = build_zero_shot_prompt(test[0])
                Path("debug/zero_shot_chat.txt").write_text(zc_prompt)
            zs_chat_preds = predict_zero_shot(client, chat_model, test, desc="Zero-shot (Chat)", mode=eval_mode)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    if not zs_chat_preds and last_err is not None:
        raise last_err
    print(f"Zero-shot (Chat) completed in {time.time()-t1:.1f}s")
    zs_chat_acc = accuracy(zs_chat_preds, [ex.label for ex in test])

    # Many-shot with ICM labels (Base)
    icm_shots = build_icm_shots(train, icm_labels)
    # Fairness: cap both ICM and Golden to same K if provided (or min with ICM size)
    k_cap = None
    # eval_k <= 0 means unlimited; None keeps prior default behavior
    if eval_k is not None and eval_k <= 0:
        k_cap = None
    elif eval_k is not None:
        k_cap = min(eval_k, len(icm_shots))
    elif len(icm_shots) > 0:
        k_cap = min(len(icm_shots), 160)
    icm_ctx = icm_shots if k_cap is None else icm_shots[:k_cap]
    if len(test) > 0:
        icm_example_prompt = build_many_shot_prompt(icm_ctx, test[0])
        Path("debug/icm_base.txt").write_text(icm_example_prompt)
    t2 = time.time()
    icm_preds = predict_many_shot(
        client,
        base_model,
        icm_ctx,
        test,
        seed=seed,
        context_cap=k_cap,
        mode=eval_mode,
    )
    print(f"ICM (Base) completed in {time.time()-t2:.1f}s")
    icm_acc = accuracy(icm_preds, [ex.label for ex in test])

    # Many-shot with Golden labels (Base)
    golden_shots = list(train)
    if k_cap is not None and len(golden_shots) > k_cap:
        rng = random.Random(seed)
        golden_shots = rng.sample(golden_shots, k_cap)
    golden_ctx = golden_shots
    if len(test) > 0:
        golden_example_prompt = build_many_shot_prompt(golden_ctx, test[0])
        Path("debug/golden_base.txt").write_text(golden_example_prompt)
    t3 = time.time()
    golden_preds = predict_many_shot(
        client,
        base_model,
        golden_ctx,
        test,
        seed=seed,
        context_cap=k_cap,
        mode=eval_mode,
    )
    print(f"Golden (Base) completed in {time.time()-t3:.1f}s")
    golden_acc = accuracy(golden_preds, [ex.label for ex in test])

    # Sanity: print effective K sizes
    icm_k = len(icm_ctx)
    golden_k = len(golden_ctx)
    print(f"Eval K — ICM: {icm_k}, Golden: {golden_k}, Zero-shot: 0")

    results = {
        "Zero-shot (Base)": zs_base_acc,
        "Zero-shot (Chat)": zs_chat_acc,
        "ICM (Base)": icm_acc,
        "Golden (Base)": golden_acc,
    }
    # Optional coherence metrics if myth labels exist
    try:
        results.update({
            "Zero-shot (Base) vs Myth": coherence(zs_base_preds, test),
            "Zero-shot (Chat) vs Myth": coherence(zs_chat_preds, test),
            "ICM (Base) vs Myth": coherence(icm_preds, test),
            "Golden (Base) vs Myth": coherence(golden_preds, test),
        })
    except Exception:
        pass
    predictions = {
        "Zero-shot (Base)": zs_base_preds,
        "Zero-shot (Chat)": zs_chat_preds,
        "ICM (Base)": icm_preds,
        "Golden (Base)": golden_preds,
    }
    return results, predictions


def save_results(path: str, metrics: Dict[str, float]) -> None:
    Path(path).write_text(json.dumps(metrics, indent=2))


