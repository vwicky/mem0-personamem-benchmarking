"""Compare question overlap between PersonaMem benchmark exports."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _default_input(filename: str) -> Path:
    new_path = PROJECT_ROOT / "data" / "inputs" / filename
    if new_path.exists():
        return new_path
    return PROJECT_ROOT / "exports" / filename


def _default_output(filename: str) -> Path:
    return PROJECT_ROOT / "artifacts" / "analysis_exports" / filename


def normalize_question(question: str) -> str:
    """Normalize text so semantically identical strings match."""
    return re.sub(r"\s+", " ", question).strip().lower()


def load_questions(jsonl_path: Path) -> set[str]:
    """Load all `user_query` strings from a benchmark JSONL file."""
    questions: set[str] = set()

    with jsonl_path.open("r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue

            bundle = json.loads(line)
            rows_by_split = bundle.get("rows_by_split", {})
            for split_rows in rows_by_split.values():
                for row in split_rows:
                    query = row.get("user_query")
                    if isinstance(query, str) and query.strip():
                        questions.add(normalize_question(query))

    return questions


def save_shared_questions(path: Path, shared_questions: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as outfile:
        for question in sorted(shared_questions):
            outfile.write(f"{question}\n")


def draw_venn(
    text_count: int,
    multimodal_count: int,
    shared_count: int,
    output_path: Path,
) -> None:
    """Draw a 2-set Venn-style diagram without external dependencies."""
    text_only = text_count - shared_count
    multimodal_only = multimodal_count - shared_count

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.3, 1.8)
    ax.set_aspect("equal")
    ax.axis("off")

    left_circle = Circle((0.9, 0.9), 0.8, color="#5DA5DA", alpha=0.45, lw=2)
    right_circle = Circle((1.5, 0.9), 0.8, color="#F17CB0", alpha=0.45, lw=2)
    ax.add_patch(left_circle)
    ax.add_patch(right_circle)

    ax.text(0.6, 1.65, "Text Benchmark", ha="center", fontsize=12, fontweight="bold")
    ax.text(1.8, 1.65, "Multimodal Benchmark", ha="center", fontsize=12, fontweight="bold")

    ax.text(0.55, 0.9, f"{text_only:,}", ha="center", va="center", fontsize=14)
    ax.text(1.2, 0.9, f"{shared_count:,}", ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(1.85, 0.9, f"{multimodal_only:,}", ha="center", va="center", fontsize=14)

    ax.text(0.55, 0.45, "Text only", ha="center", fontsize=10, color="#2F5D88")
    ax.text(1.2, 0.45, "Shared", ha="center", fontsize=10)
    ax.text(1.85, 0.45, "Multimodal only", ha="center", fontsize=10, color="#9D3E73")

    ax.set_title("PersonaMem Question Overlap", fontsize=15, pad=20)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare shared questions across PersonaMem benchmark exports."
    )
    parser.add_argument(
        "--text-jsonl",
        type=Path,
        default=_default_input("personamem_benchmark_text_32k_user_bundles.jsonl"),
        help="Path to text benchmark JSONL export.",
    )
    parser.add_argument(
        "--multimodal-jsonl",
        type=Path,
        default=_default_input("personamem_benchmark_multimodal_32k_user_bundles.jsonl"),
        help="Path to multimodal benchmark JSONL export.",
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=_default_output("personamem_question_overlap_venn.png"),
        help="Where to save the venn diagram image.",
    )
    parser.add_argument(
        "--shared-questions-output",
        type=Path,
        default=_default_output("personamem_shared_questions.txt"),
        help="Where to save sorted shared questions.",
    )
    args = parser.parse_args()

    text_questions = load_questions(args.text_jsonl)
    multimodal_questions = load_questions(args.multimodal_jsonl)
    shared_questions = text_questions & multimodal_questions

    print(f"Text unique questions: {len(text_questions):,}")
    print(f"Multimodal unique questions: {len(multimodal_questions):,}")
    print(f"Shared identical questions: {len(shared_questions):,}")

    save_shared_questions(args.shared_questions_output, shared_questions)
    draw_venn(
        text_count=len(text_questions),
        multimodal_count=len(multimodal_questions),
        shared_count=len(shared_questions),
        output_path=args.output_image,
    )

    print(f"Venn diagram saved to: {args.output_image}")
    print(f"Shared questions list saved to: {args.shared_questions_output}")


if __name__ == "__main__":
    main()
