"""
Run PersonaMem multimodal experiment from CLI.

Notebook reference:
- notebooks/experiment_personamem_multi.ipynb
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_EXPERIMENT_ID = "multi_gpt-4.1.-mini_graph-img-preserved-2"
DEFAULT_JSONL = Path("data/inputs/personamem_benchmark_multimodal_32k_user_bundles.jsonl")


def _print_stage1_summary(log1) -> None:
    print(
        f"Stage 1 complete: {log1.num_users} users, "
        f"{log1.total_add_calls} successful adds, {len(log1.errors)} errors"
    )
    print(f"Total wall time: {log1.total_wall_seconds:.1f}s")
    if log1.total_input_tokens:
        print(f"Total input tokens (approx): {log1.total_input_tokens}")
    used_graph_s1 = log1.per_user[0].get("used_graph") if log1.per_user else None
    print(
        f"Graph memory: {'enabled' if used_graph_s1 else 'disabled (vector-only fallback)'}"
    )


def _print_stage2_summary(log2) -> None:
    print(
        f"Stage 2 complete: {log2.num_qa_pairs} QA pairs, "
        f"{log2.total_search_calls} searches"
    )
    print(f"Total wall time: {log2.total_wall_seconds:.1f}s")
    print(f"Answer LLM calls: {log2.total_answer_llm_calls}")
    if log2.total_answer_input_tokens:
        print(f"Answer input tokens: {log2.total_answer_input_tokens}")
    if log2.errors:
        print(f"Errors: {len(log2.errors)}")
    used_graph_s2 = log2.per_qa[0].get("used_graph") if log2.per_qa else None
    print(
        f"Graph memory: {'enabled' if used_graph_s2 else 'disabled (vector-only fallback)'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PersonaMem multimodal benchmark experiment (stage1/stage2)."
    )
    parser.add_argument(
        "--stage",
        choices=["1", "2", "both"],
        default="both",
        help="Run stage 1, stage 2, or both.",
    )
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--max-qa-per-user", type=int, default=None)
    parser.add_argument("--graph-retries", type=int, default=3)
    parser.add_argument(
        "--enable-graph",
        action="store_true",
        help="Enable Neo4j graph mode. Default is vector-only fallback mode.",
    )
    parser.add_argument(
        "--use-async-stage1",
        action="store_true",
        help="Use async stage1 ingestion. Notebook default is sync.",
    )
    parser.add_argument("--max-concurrent", type=int, default=5)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--stream-path", type=Path, default=None)
    parser.add_argument("--resume-from-stream", type=Path, default=None)
    parser.add_argument("--stage1-log-path", type=Path, default=None)
    parser.add_argument("--no-skip-failed-stage1-users", action="store_true")
    parser.add_argument(
        "--close-cached-clients-first",
        action="store_true",
        help="Call close_cached_mem_clients() before starting (useful between repeated runs).",
    )
    args = parser.parse_args()

    from experiment_runner import (
        run_stage1_fill_experiment,
        run_stage2_qa_experiment,
        get_experiment_log_dir,
        close_cached_mem_clients,
    )
    from personamem.dataset_exports import ensure_bundle_jsonl

    args.jsonl = ensure_bundle_jsonl(args.jsonl, split="benchmark_multimodal")

    if args.close_cached_clients_first:
        close_cached_mem_clients()

    log_dir = get_experiment_log_dir(args.experiment_id)
    print(f"Experiment ID: {args.experiment_id}")
    print(f"Bundles: {args.jsonl}")
    print(f"Log directory: {log_dir}")

    try:
        if args.stage in {"1", "both"}:
            log1 = run_stage1_fill_experiment(
                experiment_id=args.experiment_id,
                jsonl_path=args.jsonl,
                split="benchmark_multimodal",
                max_users=args.max_users,
                use_async=args.use_async_stage1,
                max_concurrent=args.max_concurrent,
                graph_retries=args.graph_retries,
                enable_graph=args.enable_graph,
            )
            _print_stage1_summary(log1)
            if args.stage == "both":
                # Release local Qdrant lock before Stage 2 creates its client.
                close_cached_mem_clients()

        if args.stage in {"2", "both"}:
            log2 = run_stage2_qa_experiment(
                experiment_id=args.experiment_id,
                jsonl_path=args.jsonl,
                split="benchmark_multimodal",
                max_users=args.max_users,
                max_qa_per_user=args.max_qa_per_user,
                rerank=not args.no_rerank,
                graph_retries=args.graph_retries,
                enable_graph=args.enable_graph,
                stream_path=args.stream_path,
                resume_from_stream=args.resume_from_stream,
                stage1_log_path=args.stage1_log_path,
                skip_failed_stage1_users=not args.no_skip_failed_stage1_users,
            )
            _print_stage2_summary(log2)
    finally:
        # Avoid noisy Qdrant destructor warnings during interpreter shutdown.
        close_cached_mem_clients()


if __name__ == "__main__":
    main()
