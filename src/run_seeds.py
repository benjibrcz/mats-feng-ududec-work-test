import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List

from .data_loader import load_truthfulqa
from .hyperbolic_client import HyperbolicClient
from .icm import ICMSearch
from .eval import run_full_evaluation
from .plot_results import plot_bars_with_errorbars


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple seeds and aggregate error bars")
    parser.add_argument("--train", type=str, default="data/truthfulqa_train.json")
    parser.add_argument("--test", type=str, default="data/truthfulqa_test.json")
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3.1-405B")
    parser.add_argument("--chat_model", type=str, default="meta-llama/Meta-Llama-3.1-405B-Instruct")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--icm_steps", type=int, default=200)
    parser.add_argument("--icm_target_labels", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=40.0)
    parser.add_argument("--context_cap", type=int, default=96)
    parser.add_argument("--eval_k", type=int, default=128)
    parser.add_argument("--out_prefix", type=str, default="results_seeds")
    parser.add_argument("--eval_mode", type=str, default="auto", choices=["auto", "strict_text"])
    args = parser.parse_args()

    train = load_truthfulqa(args.train)
    test = load_truthfulqa(args.test)
    client = HyperbolicClient()

    per_seed_metrics: List[Dict[str, float]] = []
    for seed in args.seeds:
        icm = ICMSearch(
            base_model=args.base_model,
            client=client,
            alpha=args.alpha,
            context_cap=args.context_cap,
            seed=seed,
        )
        icm.initialize_random(train, k_init=8)
        icm_labels = icm.run(
            train,
            steps=args.icm_steps,
            target_labels=args.icm_target_labels,
        )
        metrics, _ = run_full_evaluation(
            client=client,
            base_model=args.base_model,
            chat_model=args.chat_model,
            train=train,
            test=test,
            icm_labels=icm_labels,
            seed=seed,
            eval_k=args.eval_k,
            eval_mode=args.eval_mode,
        )
        per_seed_metrics.append(metrics)
        Path(f"{args.out_prefix}_{seed}.json").write_text(json.dumps(metrics, indent=2))

    # Aggregate means and stds
    keys = ["Zero-shot (Base)", "Zero-shot (Chat)", "ICM (Base)", "Golden (Base)"]
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for k in keys:
        vals = [m[k] for m in per_seed_metrics]
        means[k] = float(statistics.fmean(vals))
        stds[k] = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0

    Path(f"{args.out_prefix}_mean.json").write_text(json.dumps({"mean": means, "std": stds}, indent=2))
    plot_bars_with_errorbars(means, stds, outfile=f"{args.out_prefix}.png")
    print(f"Saved aggregated results to {args.out_prefix}_mean.json and {args.out_prefix}.png")


if __name__ == "__main__":
    main()


