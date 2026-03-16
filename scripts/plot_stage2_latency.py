"""Generate nice distribution plots for Stage 2 per-QA generation and search time."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _benchmark_logs_root() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    new_root = project_root / "artifacts" / "benchmark_logs"
    if new_root.exists():
        return new_root
    return project_root / "benchmark_logs"


# Allow running from project root; df must exist (run from notebook or load csv)
if __name__ == "__main__":
    csv = _benchmark_logs_root() / "text_gpt-4.1.-mini_graph/stage2_unified_from_logs.csv"
    if not csv.exists():
        print("Run stage2_analysis.ipynb first to create stage2_unified_from_logs.csv")
        sys.exit(1)
    df = pd.read_csv(csv)

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def plot_latency(series, ax, title, color, xlabel="Time (s)"):
    p99 = series.quantile(0.99)
    x = series.clip(upper=p99)
    ax.hist(x, bins=60, color=color, alpha=0.85, edgecolor="white", linewidth=0.4)
    mean_val, median_val = series.mean(), series.median()
    ax.axvline(mean_val, color="darkred", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2f}s")
    ax.axvline(median_val, color="navy", linestyle=":", linewidth=1.5, label=f"Median: {median_val:.2f}s")
    ax.set_title(title, fontsize=13, fontweight="600")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, None)
    n_out = (series > p99).sum()
    if n_out > 0:
        ax.text(0.98, 0.98, f"({n_out} points > {p99:.0f}s not shown)", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="gray")


fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
plot_latency(df["llm_sec"], axes[0], "Per QA: generation time (LLM answer)", color="#2e86ab")
plot_latency(df["search_sec"], axes[1], "Per QA: search time", color="#a23b72")
plt.tight_layout()
out = _benchmark_logs_root() / "text_gpt-4.1.-mini_graph/stage2_latency_plots.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
plt.show()
