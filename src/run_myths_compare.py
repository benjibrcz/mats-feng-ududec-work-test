import argparse
import json
from pathlib import Path

from .data_loader import load_truthfulqa
from .hyperbolic_client import HyperbolicClient
from .icm import ICMSearch
from .eval import run_full_evaluation
from .plot_results import plot_coherence_tradeoff


def run_once(train_path: str, test_path: str, base_model: str, chat_model: str, seed: int, icm_steps: int, icm_target_labels: int, alpha: float, context_cap: int, eval_k: int, eval_mode: str, icm_seed_myth: bool) -> dict:
    train = load_truthfulqa(train_path)
    test = load_truthfulqa(test_path)
    client = HyperbolicClient()
    icm = ICMSearch(base_model=base_model, client=client, alpha=alpha, context_cap=context_cap, seed=seed)
    if icm_seed_myth and any(ex.myth_label is not None for ex in train):
        import random as _r
        _r.seed(seed)
        pool = [ex for ex in train if ex.myth_label is not None]
        init = _r.sample(pool, k=min(8, len(pool)))
        for ex in init:
            icm.labels[ex.example_id] = int(ex.myth_label)
    else:
        icm.initialize_random(train, k_init=8)
    icm_labels = icm.run(train, steps=icm_steps, target_labels=icm_target_labels)
    metrics, _ = run_full_evaluation(
        client=client,
        base_model=base_model,
        chat_model=chat_model,
        train=train,
        test=test,
        icm_labels=icm_labels,
        seed=seed,
        eval_k=eval_k,
        eval_mode=eval_mode,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Neutral vs Myth-seeded ICM on Coherent Myths")
    parser.add_argument("--train", type=str, default="data/coherent_myths_train.json")
    parser.add_argument("--test", type=str, default="data/coherent_myths_test.json")
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3.1-405B")
    parser.add_argument("--chat_model", type=str, default="meta-llama/Meta-Llama-3.1-405B-Instruct")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--icm_steps", type=int, default=200)
    parser.add_argument("--icm_target_labels", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=40.0)
    parser.add_argument("--context_cap", type=int, default=96)
    parser.add_argument("--eval_k", type=int, default=64)
    parser.add_argument("--eval_mode", type=str, default="strict_text", choices=["auto", "strict_text"])
    parser.add_argument("--out_prefix", type=str, default="results_myths_compare")
    args = parser.parse_args()

    neutral = run_once(
        args.train, args.test, args.base_model, args.chat_model, args.seed,
        args.icm_steps, args.icm_target_labels, args.alpha, args.context_cap, args.eval_k, args.eval_mode, icm_seed_myth=False,
    )
    myth = run_once(
        args.train, args.test, args.base_model, args.chat_model, args.seed,
        args.icm_steps, args.icm_target_labels, args.alpha, args.context_cap, args.eval_k, args.eval_mode, icm_seed_myth=True,
    )

    Path(f"{args.out_prefix}_neutral.json").write_text(json.dumps(neutral, indent=2))
    Path(f"{args.out_prefix}_myth.json").write_text(json.dumps(myth, indent=2))

    points = {
        "Neutral seed": {
            "accuracy": float(neutral.get("ICM (Base)", 0.0)),
            "coherence": float(neutral.get("ICM (Base) vs Myth", 0.0)),
        },
        "Myth seed": {
            "accuracy": float(myth.get("ICM (Base)", 0.0)),
            "coherence": float(myth.get("ICM (Base) vs Myth", 0.0)),
        },
    }
    Path(f"{args.out_prefix}.json").write_text(json.dumps(points, indent=2))
    plot_coherence_tradeoff(points, outfile=f"{args.out_prefix}.png")
    print(f"Saved comparison to {args.out_prefix}.json and {args.out_prefix}.png")


if __name__ == "__main__":
    main()


