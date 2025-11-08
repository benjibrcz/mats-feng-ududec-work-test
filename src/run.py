import argparse
import time
from pathlib import Path

from .data_loader import load_truthfulqa
from .hyperbolic_client import HyperbolicClient
from .icm import ICMSearch
from .eval import run_full_evaluation, save_results
from .plot_results import plot_bars


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ICM TruthfulQA pipeline")
    parser.add_argument("--train", type=str, default="data/truthfulqa_train.json")
    parser.add_argument("--test", type=str, default="data/truthfulqa_test.json")
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3.1-405B")
    parser.add_argument("--chat_model", type=str, default="meta-llama/Meta-Llama-3.1-405B-Instruct")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--icm_steps", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=50.0)
    parser.add_argument("--context_cap", type=int, default=None) # was 96
    parser.add_argument("--icm_target_labels", type=int, default=None)
    parser.add_argument("--eval_k", type=int, default=None)
    parser.add_argument("--audit_loader", action="store_true")
    parser.add_argument("--eval_mode", type=str, default="auto", choices=["auto", "strict_text"])
    parser.add_argument("--icm_delta", type=str, default="local", choices=["local", "global"], help="Use local or global ΔU for ICM acceptance")
    parser.add_argument("--icm_norm_ctx", action="store_true", help="Normalize local ΔU by context size")
    parser.add_argument("--results", type=str, default="results.json")
    parser.add_argument("--figure", type=str, default="results.png")
    parser.add_argument("--icm_seed_myth", action="store_true", help="Seed ICM initial labels from myth labels if present")
    args = parser.parse_args()

    train = load_truthfulqa(args.train)
    test = load_truthfulqa(args.test)

    client = HyperbolicClient()

    if args.audit_loader:
        print("Loader audit (first 10 examples):")
        for ex in train[:10]:
            print({
                "id": ex.example_id,
                "question": ex.question[:60] + ("..." if len(ex.question) > 60 else ""),
                "claim": ex.claim[:60] + ("..." if len(ex.claim) > 60 else ""),
                "gold": bool(ex.label),
                "group": ex.group_id,
            })

    icm = ICMSearch(
        base_model=args.base_model,
        client=client,
        alpha=args.alpha,
        context_cap=args.context_cap,
        seed=args.seed,
    )
    icm.delta_mode = args.icm_delta
    icm.normalize_by_ctx = bool(args.icm_norm_ctx)
    if args.icm_seed_myth and any(ex.myth_label is not None for ex in train):
        import random as _r
        _r.seed(args.seed)
        pool = [ex for ex in train if ex.myth_label is not None]
        init = _r.sample(pool, k=min(8, len(pool)))
        for ex in init:
            icm.labels[ex.example_id] = int(ex.myth_label)
    else:
        icm.initialize_random(train, k_init=8)
    t_icm = time.time()
    icm_labels = icm.run(
        train,
        steps=args.icm_steps,
        target_labels=args.icm_target_labels,
    )
    print(f"ICM labeling took {time.time()-t_icm:.1f}s; labeled {len(icm_labels)} examples.")

    t_eval = time.time()
    metrics, _preds = run_full_evaluation(
        client=client,
        base_model=args.base_model,
        chat_model=args.chat_model,
        train=train,
        test=test,
        icm_labels=icm_labels,
        seed=args.seed,
        eval_k=args.eval_k,
        eval_mode=args.eval_mode,
    )
    print(f"Evaluation took {time.time()-t_eval:.1f}s")

    save_results(args.results, metrics)
    plot_bars(metrics, outfile=args.figure)
    print(f"Saved {args.results} and {args.figure}")


if __name__ == "__main__":
    main()


