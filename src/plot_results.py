from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import seaborn as sns


def plot_bars(metrics: Dict[str, float], outfile: str = "results.png") -> None:
    keys: List[str] = [
        "Zero-shot (Base)",
        "Zero-shot (Chat)",
        "ICM (Base)",
        "Golden (Base)",
    ]
    values = [metrics[k] for k in keys]
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=keys, y=values, palette="muted")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Condition")
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_bars_with_errorbars(
    means: Dict[str, float],
    stds: Dict[str, float],
    outfile: str = "results_errorbars.png",
) -> None:
    keys: List[str] = [
        "Zero-shot (Base)",
        "Zero-shot (Chat)",
        "ICM (Base)",
        "Golden (Base)",
    ]
    vals = [means.get(k, 0.0) for k in keys]
    errs = [stds.get(k, 0.0) for k in keys]
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=keys, y=vals, palette="muted")
    ax.errorbar(range(len(keys)), vals, yerr=errs, fmt="none", ecolor="black", capsize=5)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Condition")
    for i, v in enumerate(vals):
        ax.text(i, min(0.98, v + 0.02), f"{v:.2f}", ha="center")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_k_sweep(
    k_values: Sequence[int],
    results_by_k: Dict[int, Dict[str, float]],
    outfile: str = "k_sweep.png",
) -> None:
    """results_by_k: {k: {condition: acc}}"""
    keys = ["Zero-shot (Base)", "Zero-shot (Chat)", "ICM (Base)", "Golden (Base)"]
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))
    for key in keys:
        ys = [results_by_k.get(k, {}).get(key, None) for k in k_values]
        plt.plot(k_values, ys, marker="o", label=key)
    plt.ylim(0.0, 1.0)
    plt.xlabel("eval_k (shots)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_coherence_tradeoff(
    points: Dict[str, Dict[str, float]],  # {label: {"accuracy": x, "coherence": y}}
    outfile: str = "coherence_tradeoff.png",
) -> None:
    sns.set(style="whitegrid")
    plt.figure(figsize=(5, 5))
    for label, vals in points.items():
        plt.scatter(vals.get("accuracy", 0.0), vals.get("coherence", 0.0), s=80, label=label)
        plt.text(vals.get("accuracy", 0.0) + 0.01, vals.get("coherence", 0.0) + 0.01, label)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Truth Accuracy (ICM Base)")
    plt.ylabel("Myth Coherence (ICM Base vs Myth)")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


