from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class UserMem0Bundle:
    """
    Lightweight, mem0-friendly view of one PersonaMem user.

    - `user_id`: the PersonaMem `persona_id`
    - `chat_history`: list of {role, content} messages (what you ingest into mem0)
    - `metadata`: metadata from the chat history JSON (e.g. total_messages, final_token_count)
    - `benchmark_rows`: all benchmark rows for this user in the chosen split
    - `raw_bundle`: optional full original JSON line (if you need extra fields later)
    """

    user_id: int
    chat_history: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    benchmark_rows: List[Dict[str, Any]]
    raw_bundle: Optional[Dict[str, Any]] = None


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield parsed JSON objects from a .jsonl file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_personamem_jsonl_for_mem0(
    jsonl_path: str | Path,
    split: str = "benchmark_text",
    *,
    include_raw_bundle: bool = False,
) -> List[UserMem0Bundle]:
    """
    Read the exported PersonaMem JSONL and return data prepared for feeding into mem0.

    Parameters
    ----------
    jsonl_path:
        Path to `personamem_*_user_bundles.jsonl` created in `checking_data.ipynb`.
    split:
        Which split's rows to treat as the benchmark for this user
        (e.g. 'benchmark_text', 'benchmark_multimodal').
    include_raw_bundle:
        If True, also keep the original JSON object per user in `raw_bundle`.

    Returns
    -------
    List[UserMem0Bundle]

    Each element contains:
      - user_id: `persona_id`
      - chat_history: list of messages (role/content) for ingestion into mem0
      - metadata: dict from the chat history JSON's `metadata`
      - benchmark_rows: list of row dicts for this user in the chosen split
    """

    path = Path(jsonl_path)
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    bundles: List[UserMem0Bundle] = []

    for obj in _iter_jsonl(path):
        persona_id = int(obj["persona_id"])

        # Chat history JSON comes directly from PersonaMem chat_history_32k JSON.
        chat_json = obj.get("chat_history_json") or {}
        metadata = chat_json.get("metadata", {}) if isinstance(chat_json, dict) else {}

        # The main thing mem0 needs: ordered conversation messages.
        chat_history = chat_json.get("chat_history", [])
        if not isinstance(chat_history, list):
            chat_history = []

        rows_by_split = obj.get("rows_by_split", {})
        benchmark_rows = rows_by_split.get(split, []) or []

        bundle = UserMem0Bundle(
            user_id=persona_id,
            chat_history=chat_history,
            metadata=metadata,
            benchmark_rows=benchmark_rows,
            raw_bundle=obj if include_raw_bundle else None,
        )
        bundles.append(bundle)

    return bundles


def short_view_user_bundle(bundle: UserMem0Bundle) -> str:
    """
    Return a short human-readable summary of a UserMem0Bundle.

    Useful in notebooks / logs when you just want to sanity-check
    what will be sent to mem0 for one user.
    """

    n_messages = len(bundle.chat_history)
    n_rows = len(bundle.benchmark_rows)
    meta = bundle.metadata or {}

    # Try to surface a few meaningful metadata fields if present.
    total_messages = meta.get("total_messages")
    final_token_count = meta.get("final_token_count")

    lines = [
        f"user_id: {bundle.user_id}",
        f"chat_history: {n_messages} messages",
        f"benchmark_rows: {n_rows}",
    ]

    meta_lines = []
    if total_messages is not None:
        meta_lines.append(f"total_messages: {total_messages}")
    if final_token_count is not None:
        meta_lines.append(f"final_token_count: {final_token_count}")
    if meta_lines:
        lines.append("metadata:")
        lines.extend([f"  - {line}" for line in meta_lines])

    # Show first and last message on separate blocks for readability.
    if bundle.chat_history:
        first = bundle.chat_history[0]
        last = bundle.chat_history[-1]
        lines.append("first_message:")
        lines.append(f"  role: {first.get('role')}")
        lines.append(f"  content: {(first.get('content') or '')[:200]!r}")

        if n_messages > 1:
            lines.append("last_message:")
            lines.append(f"  role: {last.get('role')}")
            lines.append(f"  content: {(last.get('content') or '')[:200]!r}")

    return "\n".join(lines)


__all__ = [
    "UserMem0Bundle",
    "load_personamem_jsonl_for_mem0",
    "short_view_user_bundle",
]


