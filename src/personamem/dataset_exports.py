from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import pandas as pd
from datasets import DatasetDict, load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

DATASET_REPO = "bowen-upenn/PersonaMem-v2"
DEFAULT_HISTORY_COL = "chat_history_32k_link"


def _open_repo_json(filename: str) -> tuple[str, Dict[str, Any]]:
    """Download (if needed) and open a JSON file from the dataset repo."""
    local_path = hf_hub_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        filename=filename,
    )
    with open(local_path, "r", encoding="utf-8") as f:
        return local_path, json.load(f)


def _build_split_cache(ds: DatasetDict, splits: Sequence[str]) -> dict[str, pd.DataFrame]:
    cache: dict[str, pd.DataFrame] = {}
    for split in splits:
        if split not in ds:
            raise ValueError(f"Unknown split: {split}. Available: {list(ds.keys())}")
        cache[split] = ds[split].to_pandas()
    return cache


def _get_user_bundle_multi(
    user_id: int,
    split_df_cache: dict[str, pd.DataFrame],
    splits: Sequence[str],
    history_col: str = DEFAULT_HISTORY_COL,
) -> Dict[str, Any]:
    rows_by_split: dict[str, pd.DataFrame] = {}
    history_repo_path: Optional[str] = None

    for split in splits:
        df = split_df_cache[split]
        user_rows = df[df["persona_id"] == int(user_id)].copy()
        rows_by_split[split] = user_rows

        if history_repo_path is None and not user_rows.empty:
            if history_col not in user_rows.columns:
                raise ValueError(
                    f"Column '{history_col}' not found in split='{split}'. "
                    f"Available columns: {list(user_rows.columns)}"
                )
            history_repo_path = str(user_rows[history_col].iloc[0])

    if history_repo_path is None:
        raise ValueError(f"persona_id={user_id} not found in splits={splits}")

    chat_local_path, chat_json = _open_repo_json(history_repo_path)

    return {
        "persona_id": int(user_id),
        "splits": list(splits),
        "chat_history_path": history_repo_path,
        "chat_history_local_path": chat_local_path,
        "chat_history_json": chat_json,
        "rows_by_split": {
            split: df.reset_index(drop=True).to_dict(orient="records")
            for split, df in rows_by_split.items()
        },
    }


def export_user_bundles_jsonl(
    out_path: str | Path,
    *,
    splits: Iterable[str] = ("benchmark_text",),
    history_col: str = DEFAULT_HISTORY_COL,
) -> Path:
    """
    Write one JSON object per user (persona_id) into a JSONL file.
    """
    out_path = Path(out_path)
    splits = tuple(splits)
    if not splits:
        raise ValueError("splits must be non-empty")

    ds = load_dataset(DATASET_REPO)
    split_df_cache = _build_split_cache(ds, splits)

    persona_ids: set[int] = set()
    for split in splits:
        persona_ids |= set(split_df_cache[split]["persona_id"].astype(int).unique().tolist())
    persona_ids_sorted = sorted(persona_ids)

    print(f"Exporting {len(persona_ids_sorted)} users to {out_path}...")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for pid in tqdm(persona_ids_sorted, desc="Exporting user bundles", unit="user"):
            bundle = _get_user_bundle_multi(
                pid,
                split_df_cache=split_df_cache,
                splits=splits,
                history_col=history_col,
            )
            f.write(json.dumps(bundle, ensure_ascii=False))
            f.write("\n")

    print("Done.")
    return out_path


def ensure_bundle_jsonl(
    out_path: str | Path,
    *,
    split: str,
    history_col: str = DEFAULT_HISTORY_COL,
) -> Path:
    """
    Ensure the benchmark user-bundle JSONL exists.
    If missing, generate from PersonaMem-v2 on Hugging Face.
    """
    out_path = Path(out_path)
    if out_path.is_file():
        return out_path

    print(f"Input JSONL not found: {out_path}")
    print(f"Generating from dataset repo '{DATASET_REPO}' for split '{split}'...")
    return export_user_bundles_jsonl(
        out_path=out_path,
        splits=(split,),
        history_col=history_col,
    )
