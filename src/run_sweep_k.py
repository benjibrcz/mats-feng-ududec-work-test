import argparse
import json
from pathlib import Path
from typing import Dict, List

from .data_loader import load_truthfulqa
from .hyperbolic_client import HyperbolicClient
from .icm import ICMSearch
from .eval import run_full_evaluation
from .plot_results import plot_k_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep eval_k values and plot accuracies")
    parser.add_argument("--train", type=str, default="data/truthfulqa_train.json")
    parser.add_argument("--test", type=str, default="data/truthfulqa_test.json")
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3.1-405B")
    parser.add_argument("--chat_model", type=str, default="meta-llama/Meta-Llama-3.1-405B-Instruct")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--icm_steps", type=int, default=300)
    parser.add_argument("--icm_target_labels", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=40.0)
    parser.add_argument("--context_cap", type=int, default=96)
    parser.add_argument("--k_list", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--out_prefix", type=str, default="results_k_sweep")
    parser.add_argument("--eval_mode", type=str, default="auto", choices=["auto", "strict_text"])
    args = parser.parse_args()

    train = load_truthfulqa(args.train)
    test = load_truthfulqa(args.test)
    client = HyperbolicClient()

    # Produce a single ICM label set once (to control variance w.r.t. K)
    icm = ICMSearch(
        base_model=args.base_model,
        client=client,
        alpha=args.alpha,
        context_cap=args.context_cap,
        seed=args.seed,
    )
    icm.initialize_random(train, k_init=8)
    icm_labels = icm.run(
        train,
        steps=args.icm_steps,
        target_labels=args.icm_target_labels,
    )

    results_by_k: Dict[int, Dict[str, float]] = {}
    for k in args.k_list:
        metrics, _ = run_full_evaluation(
            client=client,
            base_model=args.base_model,
            chat_model=args.chat_model,
            train=train,
            test=test,
            icm_labels=icm_labels,
            seed=args.seed,
            eval_k=k,
            eval_mode=args.eval_mode,
        )
        results_by_k[k] = metrics
        Path(f"{args.out_prefix}_{k}.json").write_text(json.dumps(metrics, indent=2))

    Path(f"{args.out_prefix}.json").write_text(json.dumps(results_by_k, indent=2))
    plot_k_sweep(args.k_list, results_by_k, outfile=f"{args.out_prefix}.png")
    print(f"Saved sweep to {args.out_prefix}.json and {args.out_prefix}.png")


if __name__ == "__main__":
    main()


