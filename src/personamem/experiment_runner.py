"""
Experiment-scoped Mem0 runs: persistent on-disk memory per experiment_id,
graph used with 3 retries then fallback, and structured logs under benchmark_logs/<experiment_id>/.

Use this module from the experiment notebook for Chapter 1 (fill memory) and Chapter 2 (QA).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .paths import ENV_FILE, resolve_benchmark_logs_dir

try:
    from dotenv import load_dotenv
    load_dotenv(ENV_FILE)
except ImportError:
    pass

# Apply safe JSON parsing for Mem0 LLM responses (truncated/malformed JSON repair)
from . import mem0_safe_json  # noqa: F401
# Apply robust graph entity parsing for malformed extract_entities tool payloads
from . import mem0_safe_graph  # noqa: F401

from .personamem_mem0_prep import load_personamem_jsonl_for_mem0, UserMem0Bundle
from .mem0_full_stack import get_mem, get_async_mem, LLM_MODEL
from mem0 import __version__ as MEM0_VERSION

# Import helpers and dataclasses from benchmark (no circular: benchmark doesn't import us)
from .personamem_benchmark import (
    _normalize_user_query,
    _answer_with_llm,
    count_images_in_chat_history,
    Stage1Log,
    Stage2Log,
    Stage1UserRecord,
    Stage2QARecord,
)
from tqdm import tqdm

try:
    from openai import __version__ as OPENAI_VERSION
except ImportError:
    OPENAI_VERSION = None

# --- Experiment and retry constants ---
DEFAULT_EXPERIMENT_ID = "text_gpt-4.1.-mini_graph"
GRAPH_RETRIES = 3
QDRANT_LOCK_RETRIES = 8
QDRANT_LOCK_RETRY_BACKOFF_SEC = 2.0

# One local Qdrant path cannot be opened by multiple clients concurrently.
# Reuse clients in-process to avoid opening the same path repeatedly.
_MEM_CLIENT_CACHE: Dict[tuple[str, bool, bool], Any] = {}


def _redact_secrets(value: Any) -> Any:
    """
    Recursively redact sensitive values before writing logs.
    """
    secret_keys = {"password", "api_key", "apikey", "token", "secret", "access_key"}
    if isinstance(value, dict):
        redacted: Dict[str, Any] = {}
        for k, v in value.items():
            key_l = str(k).lower()
            if any(s in key_l for s in secret_keys):
                redacted[k] = "***REDACTED***"
            else:
                redacted[k] = _redact_secrets(v)
        return redacted
    if isinstance(value, list):
        return [_redact_secrets(v) for v in value]
    return value


def _is_qdrant_lock_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "already accessed by another instance of qdrant client" in msg or "alreadylocked" in msg


def _is_qdrant_closed_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "qdrantlocal instance is closed" in msg


def get_experiment_log_dir(experiment_id: str) -> Path:
    """Log directory for this experiment: benchmark_logs/<experiment_id>/."""
    log_dir = resolve_benchmark_logs_dir() / experiment_id
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _timestamp_suffix() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _next_stage_run_index(log_dir: Path, stage_prefix: str) -> int:
    """
    Return next run index for files named like:
    <stage_prefix>_RUN-<n>_...
    """
    pattern = re.compile(rf"^{re.escape(stage_prefix)}_RUN-(\d+)_")
    max_idx = 0
    for p in log_dir.iterdir():
        if not p.is_file():
            continue
        m = pattern.match(p.name)
        if not m:
            continue
        try:
            max_idx = max(max_idx, int(m.group(1)))
        except ValueError:
            continue
    return max_idx + 1


def _flow_log_line(stream, message: str) -> None:
    """Write a timestamped line to the flow log and flush so tail -f shows it."""
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    stream.write(f"[{ts}] {message}\n")
    stream.flush()


def _short_error_text(err: Optional[Exception], max_len: int = 220) -> str:
    """Return single-line, bounded error text for compact flow logs."""
    if err is None:
        return ""
    text = str(err).replace("\n", " ").replace("\r", " ").strip()
    while "  " in text:
        text = text.replace("  ", " ")
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _query_to_search_string(raw: Any) -> str:
    """
    Normalize user_query from a benchmark row to a string for search/LLM.
    Handles dict (e.g. {"role": "user", "content": "..."}) or string (including stringified dict).
    """
    if raw is None:
        return ""
    if isinstance(raw, dict):
        return (raw.get("content") or "").strip() or str(raw)[:2000]
    s = (raw if isinstance(raw, str) else str(raw)).strip()
    return _normalize_user_query(s) if s else ""


def _content_to_text_for_mem0(content: Any) -> str:
    """
    Convert potentially multimodal message content into plain text for Mem0 ingestion.

    - str content is passed through
    - list[blocks] keeps text blocks and replaces image blocks with a marker
    - any other content is stringified
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type in {"text", "input_text"}:
                    txt = block.get("text")
                    if isinstance(txt, str) and txt.strip():
                        parts.append(txt.strip())
                elif block_type in {"image_url", "input_image", "image"}:
                    parts.append("[Image content omitted]")
                else:
                    # Keep unknown block types visible in compact form.
                    parts.append(f"[{block_type or 'unknown_block'}]")
            else:
                raw = str(block).strip()
                if raw:
                    parts.append(raw)
        return "\n".join(parts).strip()
    if isinstance(content, dict):
        # Some tool/user payloads may come as dict content.
        txt = content.get("text") if isinstance(content.get("text"), str) else ""
        return txt.strip() if txt else str(content)
    return str(content)


def _normalize_chat_history_for_mem0(chat_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize chat messages so Mem0 always receives plain-text `content`.
    """
    normalized: List[Dict[str, Any]] = []
    for msg in chat_history:
        if not isinstance(msg, dict):
            continue
        msg_copy = dict(msg)
        msg_copy["content"] = _content_to_text_for_mem0(msg.get("content"))
        normalized.append(msg_copy)
    return normalized


def _should_retry_with_text_only(error: Exception) -> bool:
    """
    Heuristic: when raw multimodal payload fails because of content parsing,
    retry once with text-only normalized content.
    """
    msg = str(error).lower()
    indicators = (
        "list",
        "image",
        "vision",
        "content",
        "lower",
        "json",
        "parse",
        "invalid type",
    )
    return any(tok in msg for tok in indicators)


def _load_failed_stage1_user_ids(stage1_log_path: Path) -> set[int]:
    """
    Read a Stage 1 log and return user_ids that failed ingestion.
    """
    try:
        data = json.loads(stage1_log_path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    failed: set[int] = set()
    for rec in data.get("per_user", []):
        if not rec.get("success"):
            try:
                failed.add(int(rec["user_id"]))
            except Exception:
                continue
    return failed


def _latest_stage1_log_path(log_dir: Path) -> Optional[Path]:
    """
    Return the latest stage1 fill log in this experiment directory.
    """
    candidates = [
        p
        for p in log_dir.glob("stage1_*_fill_*.json")
        if p.is_file() and "_errors_" not in p.name
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def close_cached_mem_clients() -> None:
    """
    Best-effort cleanup of cached Mem0 clients to release local Qdrant file locks.
    Safe to call between notebook runs.
    """
    for mem in list(_MEM_CLIENT_CACHE.values()):
        try:
            client = getattr(getattr(mem, "vector_store", None), "client", None)
            if client and hasattr(client, "close"):
                client.close()
        except Exception:
            pass
    _MEM_CLIENT_CACHE.clear()


def _disable_graph_in_place(mem: Any) -> bool:
    """
    Best-effort graph disable on an existing Mem0 client without rebuilding vector store.
    This avoids reopening local Qdrant (and lock contention) during mid-run fallback.
    """
    changed = False
    try:
        if hasattr(mem, "enable_graph"):
            setattr(mem, "enable_graph", False)
            changed = True
        cfg = getattr(mem, "config", None)
        if cfg is not None and hasattr(cfg, "graph_store"):
            setattr(cfg, "graph_store", None)
            changed = True
    except Exception:
        return False
    return changed


def _set_graph_enabled(mem: Any, enabled: bool) -> bool:
    """Best-effort toggle for Mem0 graph execution on an existing client."""
    try:
        if hasattr(mem, "enable_graph"):
            setattr(mem, "enable_graph", bool(enabled))
            return True
    except Exception:
        return False
    return False


def _get_graph_malformed_skip_count() -> int:
    """
    Read global malformed-entity skip counter exposed by mem0_safe_graph patch.
    Returns 0 when unavailable.
    """
    try:
        getter = getattr(mem0_safe_graph, "get_malformed_entity_skip_count", None)
        if callable(getter):
            return int(getter())
    except Exception:
        return 0
    return 0


def _create_mem_with_graph(
    experiment_id: str,
    max_retries: int = GRAPH_RETRIES,
    enable_reranker: bool = False,
) -> Tuple[Any, Optional[str]]:
    """
    Create Memory client with graph enabled for this experiment.
    Returns (Memory or None, error_message).
    Tries up to max_retries times; on total failure returns (None, last_error).
    """
    cache_key = (experiment_id, True, bool(enable_reranker))
    if cache_key in _MEM_CLIENT_CACHE:
        return _MEM_CLIENT_CACHE[cache_key], None

    last_error: Optional[str] = None
    total_retries = max(max_retries, QDRANT_LOCK_RETRIES)
    for attempt in range(total_retries):
        try:
            mem = get_mem(enable_graph=True, enable_reranker=enable_reranker, experiment_id=experiment_id)
            _MEM_CLIENT_CACHE[cache_key] = mem
            return mem, None
        except Exception as e:
            last_error = str(e)
            if attempt < total_retries - 1:
                if _is_qdrant_lock_error(e):
                    time.sleep(QDRANT_LOCK_RETRY_BACKOFF_SEC * (attempt + 1))
                else:
                    time.sleep(1.0 * (attempt + 1))
    return None, last_error


def _create_mem_without_graph(experiment_id: str, enable_reranker: bool = False) -> Any:
    """Create Memory client without graph; same vector store (persistent per experiment)."""
    cache_key = (experiment_id, False, bool(enable_reranker))
    if cache_key in _MEM_CLIENT_CACHE:
        return _MEM_CLIENT_CACHE[cache_key]

    last_exc: Optional[Exception] = None
    for attempt in range(QDRANT_LOCK_RETRIES):
        try:
            mem = get_mem(enable_graph=False, enable_reranker=enable_reranker, experiment_id=experiment_id)
            _MEM_CLIENT_CACHE[cache_key] = mem
            return mem
        except Exception as e:
            last_exc = e
            if _is_qdrant_lock_error(e) and attempt < QDRANT_LOCK_RETRIES - 1:
                time.sleep(QDRANT_LOCK_RETRY_BACKOFF_SEC * (attempt + 1))
                continue
            raise

    raise RuntimeError(
        "Could not open experiment Qdrant store after retries; it is likely locked by another running kernel/process. "
        "Stop the other run or use a different experiment_id."
    ) from last_exc


def _scoped_user_id(experiment_id: str, user_id: int | str) -> str:
    """
    Scope user_id with experiment_id so records are separated per experiment in storage.
    Use this for all Mem0 add/search calls; keep original user_id in logs for readability.
    """
    return f"{experiment_id}:{user_id}"


def _get_single_mem_for_experiment(
    experiment_id: str,
    graph_retries: int = GRAPH_RETRIES,
    enable_graph: bool = True,
    enable_reranker: bool = False,
) -> Tuple[Any, bool]:
    """
    Return a single Memory client for this experiment (only one client can use local Qdrant path).
    If enable_graph is False, returns a client without graph (no Neo4j).
    If enable_graph is True, tries with graph up to graph_retries times; on failure returns a client without graph.
    Returns (memory_client, used_graph: bool).
    """
    cache_key = (experiment_id, bool(enable_graph), bool(enable_reranker))
    if cache_key in _MEM_CLIENT_CACHE:
        return _MEM_CLIENT_CACHE[cache_key], bool(enable_graph)

    if not enable_graph:
        return _create_mem_without_graph(experiment_id, enable_reranker=enable_reranker), False
    # For graph mode, preserve existing retry behavior.
    last_error: Optional[Exception] = None
    total_retries = max(graph_retries, QDRANT_LOCK_RETRIES)
    for attempt in range(total_retries):
        try:
            mem = get_mem(enable_graph=True, enable_reranker=enable_reranker, experiment_id=experiment_id)
            _MEM_CLIENT_CACHE[cache_key] = mem
            return mem, True
        except Exception as e:
            last_error = e
            if attempt < total_retries - 1:
                if _is_qdrant_lock_error(e):
                    time.sleep(QDRANT_LOCK_RETRY_BACKOFF_SEC * (attempt + 1))
                else:
                    time.sleep(1.0 * (attempt + 1))
    mem, _ = _create_mem_with_graph(
        experiment_id,
        max_retries=graph_retries,
        enable_reranker=enable_reranker,
    )
    if mem is not None:
        return mem, True
    if last_error and _is_qdrant_lock_error(last_error):
        raise RuntimeError(
            f"Qdrant storage for experiment '{experiment_id}' is locked by another process/kernel."
        ) from last_error
    return _create_mem_without_graph(experiment_id, enable_reranker=enable_reranker), False


# ---------- Stage 1: Fill memory ----------


def run_stage1_fill_experiment(
    experiment_id: str,
    jsonl_path: Path,
    *,
    split: str = "benchmark_text",
    max_users: Optional[int] = None,
    use_async: bool = False,
    max_concurrent: int = 5,
    graph_retries: int = GRAPH_RETRIES,
    enable_graph: bool = True,
) -> Stage1Log:
    """
    Fill Mem0 memory for each user from PersonaMem bundles.
    Memory is persistent and scoped to experiment_id.
    Stored user_id is prefixed with experiment_id (e.g. "exp1:123") so records are separated per experiment.
    Creates a single Memory client: tries to create a graph-enabled client up to graph_retries times;
    on failure uses a vector-only client. Each user's add() is then retried up to graph_retries times with that client.

    enable_graph: If False, graph (Neo4j) is disabled for this experiment; only vector store is used.

    use_async: Default False. Mem0's Memory client is not thread-safe; use_async=True
    runs add() from multiple threads and can cause concurrency errors. Use sync (False) for reliability.

    Writes logs to benchmark_logs/<experiment_id>/:
      - stage1_fill_<timestamp>.json (full log: per_user, errors, config, tokens, latency, etc.)
      - stage1_errors_<timestamp>.jsonl (one JSON object per error for easy parsing)
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    bundles = load_personamem_jsonl_for_mem0(jsonl_path, split=split)
    if max_users is not None:
        bundles = bundles[:max_users]

    log_dir = get_experiment_log_dir(experiment_id)
    ts = _timestamp_suffix()
    run_idx = _next_stage_run_index(log_dir, "stage1")
    run_tag = f"RUN-{run_idx}"
    main_log_path = log_dir / f"stage1_{run_tag}_fill_{ts}.json"
    errors_log_path = log_dir / f"stage1_{run_tag}_errors_{ts}.jsonl"
    flow_log_path = log_dir / f"flow_stage1_{run_tag}_{ts}.log"

    # Build config with experiment_id for logging (same as what we use for Memory)
    from .mem0_full_stack import build_full_mem0_config
    cfg = build_full_mem0_config(enable_graph=enable_graph, experiment_id=experiment_id)

    log = Stage1Log(
        stage="fill_memory",
        jsonl_path=str(jsonl_path),
        num_users=len(bundles),
        config=cfg,
        mem0_version=MEM0_VERSION,
        openai_version=OPENAI_VERSION,
    )

    with open(flow_log_path, "w", encoding="utf-8") as flow_stream:
        _flow_log_line(flow_stream, f"Stage 1 fill_memory | experiment_id={experiment_id} | num_users={len(bundles)}")
        if use_async:
            log = asyncio.run(_run_stage1_async(
                experiment_id=experiment_id,
                bundles=bundles,
                log=log,
                graph_retries=graph_retries,
                max_concurrent=max_concurrent,
                enable_graph=enable_graph,
                flow_stream=flow_stream,
            ))
        else:
            log = _run_stage1_sync(
                experiment_id=experiment_id,
                bundles=bundles,
                log=log,
                graph_retries=graph_retries,
                enable_graph=enable_graph,
                flow_stream=flow_stream,
            )
        _flow_log_line(flow_stream, f"Stage 1 complete | total_wall={log.total_wall_seconds:.1f}s | errors={len(log.errors)}")

    print(f"Flow log: {flow_log_path}")

    # Write main log
    payload = asdict(log)
    payload["config"] = _redact_secrets(payload.get("config", {}))
    payload["summary"] = log.summary_dict()
    with open(main_log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Stage 1 log written to {main_log_path}")

    # Write errors file (one JSON object per line)
    with open(errors_log_path, "w", encoding="utf-8") as f:
        for err_entry in log.errors:
            line = json.dumps({"error": err_entry}, ensure_ascii=False) + "\n"
            f.write(line)
    if log.errors:
        print(f"Stage 1 errors ({len(log.errors)}) written to {errors_log_path}")

    return log


def _run_stage1_sync(
    experiment_id: str,
    bundles: List[UserMem0Bundle],
    log: Stage1Log,
    graph_retries: int,
    enable_graph: bool = True,
    flow_stream: Optional[Any] = None,
) -> Stage1Log:
    # Single client only: local Qdrant path allows only one open client at a time.
    if flow_stream:
        _flow_log_line(flow_stream, "Creating Memory client...")
    mem, used_graph = _get_single_mem_for_experiment(
        experiment_id,
        graph_retries,
        enable_graph,
        False,
    )
    graph_disabled_for_run = False
    if flow_stream:
        _flow_log_line(flow_stream, f"Memory client ready (graph={used_graph}). Processing users.")

    total_wall = 0.0
    total_add_calls = 0
    total_images_processed = 0
    total_input_tokens = 0
    n = len(bundles)

    for idx, bundle in enumerate(tqdm(bundles, desc="Stage 1 (fill memory)", unit="user"), start=1):
        meta = bundle.metadata or {}
        approx_tokens = meta.get("final_token_count")
        image_count = count_images_in_chat_history(bundle.chat_history)
        raw_chat_history = bundle.chat_history
        normalized_chat_history = _normalize_chat_history_for_mem0(bundle.chat_history)
        if flow_stream:
            _flow_log_line(flow_stream, f"User {idx}/{n} (user_id={bundle.user_id}) | mem.add() starting")

        t0 = time.perf_counter()
        error: Optional[Exception] = None
        graph_error: Optional[str] = None
        user_used_graph = used_graph
        use_text_only = False
        had_text_only_fallback = False
        temp_graph_disabled_for_user = False
        malformed_before = _get_graph_malformed_skip_count()

        for attempt in range(graph_retries):
            try:
                payload = normalized_chat_history if use_text_only else raw_chat_history
                mem.add(
                    payload,
                    user_id=_scoped_user_id(experiment_id, bundle.user_id),
                    metadata={**(bundle.metadata or {}), "benchmark": "personamem"},
                )
                error = None
                break
            except Exception as e:
                error = e
                err_msg = str(e)
                graph_error = err_msg
                if _is_qdrant_closed_error(e):
                    # Recover from a poisoned/closed local client state.
                    close_cached_mem_clients()
                    mem, used_graph = _get_single_mem_for_experiment(
                        experiment_id,
                        graph_retries,
                        False,
                        False,
                    )
                    user_used_graph = False
                    graph_disabled_for_run = True
                    continue
                # Graph extraction can intermittently fail on malformed tool JSON.
                # Prefer temporary per-user graph-off retry (no client recreation).
                if used_graph and (not graph_disabled_for_run):
                    graph_like_failure = any(
                        token in err_msg.lower()
                        for token in [
                            "unterminated string",
                            "generate_response",
                            "graph",
                            "neo4j",
                            "entity",
                            "relationship",
                        ]
                    )
                    if graph_like_failure:
                        toggled = _set_graph_enabled(mem, False)
                        if toggled:
                            temp_graph_disabled_for_user = True
                            user_used_graph = False
                        else:
                            # If toggle is not possible, permanently disable graph in-place.
                            _disable_graph_in_place(mem)
                            graph_disabled_for_run = True
                            used_graph = False
                            user_used_graph = False
                        # Retry current user immediately with graph disabled.
                        continue
                if (not use_text_only) and _should_retry_with_text_only(e):
                    use_text_only = True
                    had_text_only_fallback = True
                    continue
                if attempt < graph_retries - 1:
                    time.sleep(1.0 * (attempt + 1))

        # Restore graph for next users when temporary per-user fallback was used.
        if temp_graph_disabled_for_user and (not graph_disabled_for_run) and used_graph:
            _set_graph_enabled(mem, True)
        malformed_after = _get_graph_malformed_skip_count()
        malformed_delta = max(0, malformed_after - malformed_before)
        if malformed_delta > 0 and graph_error is None:
            graph_error = f"graph_extract_malformed_entities_skipped={malformed_delta}"
        image_ingest_mode = "none" if image_count == 0 else ("text_only_fallback" if had_text_only_fallback else "vision")
        image_vision_succeeded = None if image_count == 0 else (error is None and (not had_text_only_fallback))

        wall = time.perf_counter() - t0
        total_wall += wall
        if flow_stream:
            status = "success" if error is None else "error"
            err_suffix = f", error: {_short_error_text(error)}" if error is not None else ""
            _flow_log_line(
                flow_stream,
                f"User {idx}/{n} (user_id={bundle.user_id}) | mem.add() done ({status}, {wall:.1f}s{err_suffix})",
            )

        if error is None:
            rec = Stage1UserRecord(
                user_id=bundle.user_id,
                num_messages=len(bundle.chat_history),
                images_processed=image_count,
                wall_seconds=wall,
                success=True,
                add_calls=1,
                input_tokens=approx_tokens,
                used_graph=user_used_graph,
                graph_error=graph_error,
                had_text_only_fallback=had_text_only_fallback,
                image_ingest_mode=image_ingest_mode,
                image_vision_succeeded=image_vision_succeeded,
            )
            total_add_calls += 1
        else:
            rec = Stage1UserRecord(
                user_id=bundle.user_id,
                num_messages=len(bundle.chat_history),
                images_processed=image_count,
                wall_seconds=wall,
                success=False,
                error=str(error),
                input_tokens=approx_tokens,
                used_graph=user_used_graph,
                graph_error=graph_error,
                had_text_only_fallback=had_text_only_fallback,
                image_ingest_mode=image_ingest_mode,
                image_vision_succeeded=image_vision_succeeded,
            )
            log.errors.append(f"user_id={bundle.user_id}: {error}")

        if rec.input_tokens is not None:
            total_input_tokens += rec.input_tokens
        total_images_processed += rec.images_processed
        log.per_user.append(asdict(rec))

    log.total_wall_seconds = total_wall
    log.total_add_calls = total_add_calls
    log.total_images_processed = total_images_processed
    if total_input_tokens:
        log.total_input_tokens = total_input_tokens
    return log


async def _run_stage1_async(
    experiment_id: str,
    bundles: List[UserMem0Bundle],
    log: Stage1Log,
    graph_retries: int,
    max_concurrent: int,
    enable_graph: bool = True,
    flow_stream: Optional[Any] = None,
) -> Stage1Log:
    # Single client only: local Qdrant path allows only one open client at a time.
    if flow_stream:
        _flow_log_line(flow_stream, "Creating Memory client...")
    mem, used_graph = _get_single_mem_for_experiment(
        experiment_id,
        graph_retries,
        enable_graph,
        False,
    )
    graph_disabled_for_run = False
    if flow_stream:
        _flow_log_line(flow_stream, f"Memory client ready (graph={used_graph}). Processing users (async, max_concurrent={max_concurrent}).")
    sem = asyncio.Semaphore(max_concurrent)

    def add_one_sync(bundle: UserMem0Bundle) -> Stage1UserRecord:
        nonlocal mem, used_graph, graph_disabled_for_run
        meta = bundle.metadata or {}
        approx_tokens = meta.get("final_token_count")
        image_count = count_images_in_chat_history(bundle.chat_history)
        raw_chat_history = bundle.chat_history
        normalized_chat_history = _normalize_chat_history_for_mem0(bundle.chat_history)
        t0 = time.perf_counter()
        error: Optional[Exception] = None
        graph_error: Optional[str] = None
        user_used_graph = used_graph
        use_text_only = False
        had_text_only_fallback = False
        temp_graph_disabled_for_user = False
        malformed_before = _get_graph_malformed_skip_count()

        for attempt in range(graph_retries):
            try:
                payload = normalized_chat_history if use_text_only else raw_chat_history
                mem.add(
                    payload,
                    user_id=_scoped_user_id(experiment_id, bundle.user_id),
                    metadata={**(bundle.metadata or {}), "benchmark": "personamem"},
                )
                error = None
                break
            except Exception as e:
                error = e
                err_msg = str(e)
                graph_error = err_msg
                if _is_qdrant_closed_error(e):
                    close_cached_mem_clients()
                    mem, used_graph = _get_single_mem_for_experiment(
                        experiment_id,
                        graph_retries,
                        False,
                        False,
                    )
                    user_used_graph = False
                    graph_disabled_for_run = True
                    continue
                if used_graph and (not graph_disabled_for_run):
                    graph_like_failure = any(
                        token in err_msg.lower()
                        for token in [
                            "unterminated string",
                            "generate_response",
                            "graph",
                            "neo4j",
                            "entity",
                            "relationship",
                        ]
                    )
                    if graph_like_failure:
                        toggled = _set_graph_enabled(mem, False)
                        if toggled:
                            temp_graph_disabled_for_user = True
                            user_used_graph = False
                        else:
                            _disable_graph_in_place(mem)
                            graph_disabled_for_run = True
                            used_graph = False
                            user_used_graph = False
                        continue
                if (not use_text_only) and _should_retry_with_text_only(e):
                    use_text_only = True
                    had_text_only_fallback = True
                    continue
                if attempt < graph_retries - 1:
                    time.sleep(1.0 * (attempt + 1))

        if temp_graph_disabled_for_user and (not graph_disabled_for_run) and used_graph:
            _set_graph_enabled(mem, True)
        malformed_after = _get_graph_malformed_skip_count()
        malformed_delta = max(0, malformed_after - malformed_before)
        if malformed_delta > 0 and graph_error is None:
            graph_error = f"graph_extract_malformed_entities_skipped={malformed_delta}"
        image_ingest_mode = "none" if image_count == 0 else ("text_only_fallback" if had_text_only_fallback else "vision")
        image_vision_succeeded = None if image_count == 0 else (error is None and (not had_text_only_fallback))

        wall = time.perf_counter() - t0
        if error is None:
            return Stage1UserRecord(
                user_id=bundle.user_id,
                num_messages=len(bundle.chat_history),
                images_processed=image_count,
                wall_seconds=wall,
                success=True,
                add_calls=1,
                input_tokens=approx_tokens,
                used_graph=user_used_graph,
                graph_error=graph_error,
                had_text_only_fallback=had_text_only_fallback,
                image_ingest_mode=image_ingest_mode,
                image_vision_succeeded=image_vision_succeeded,
            )
        else:
            return Stage1UserRecord(
                user_id=bundle.user_id,
                num_messages=len(bundle.chat_history),
                images_processed=image_count,
                wall_seconds=wall,
                success=False,
                error=str(error),
                input_tokens=approx_tokens,
                used_graph=user_used_graph,
                graph_error=graph_error,
                had_text_only_fallback=had_text_only_fallback,
                image_ingest_mode=image_ingest_mode,
                image_vision_succeeded=image_vision_succeeded,
            )

    async def add_one(bundle: UserMem0Bundle) -> Stage1UserRecord:
        async with sem:
            return await asyncio.to_thread(add_one_sync, bundle)

    tasks = [add_one(b) for b in bundles]
    results: List[Stage1UserRecord] = []
    done = 0
    n = len(bundles)
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Stage 1 (fill memory)", unit="user"):
        rec = await coro
        results.append(rec)
        done += 1
        if flow_stream:
            status = "success" if rec.error is None else "error"
            err_suffix = f", error: {_short_error_text(Exception(rec.error))}" if rec.error else ""
            _flow_log_line(
                flow_stream,
                f"User {done}/{n} (user_id={rec.user_id}) | mem.add() done ({status}, {rec.wall_seconds:.1f}s{err_suffix})",
            )
        if rec.error:
            log.errors.append(f"user_id={rec.user_id}: {rec.error}")

    results.sort(key=lambda r: r.user_id)
    log.per_user = [asdict(r) for r in results]
    log.total_wall_seconds = sum(r.wall_seconds for r in results)
    log.total_add_calls = sum(r.add_calls for r in results if r.success)
    log.total_images_processed = sum(r.images_processed for r in results)
    token_sum = sum((r.input_tokens or 0) for r in results)
    if token_sum:
        log.total_input_tokens = token_sum
    return log


# ---------- Stage 2: QA ----------


def run_stage2_qa_experiment(
    experiment_id: str,
    jsonl_path: Path,
    *,
    split: str = "benchmark_text",
    max_users: Optional[int] = None,
    max_qa_per_user: Optional[int] = None,
    rerank: bool = True,
    graph_retries: int = GRAPH_RETRIES,
    enable_graph: bool = True,
    stream_path: Optional[Path] = None,
    resume_from_stream: Optional[Path] = None,
    stage1_log_path: Optional[Path] = None,
    skip_failed_stage1_users: bool = True,
) -> Stage2Log:
    """
    Run QA: for each benchmark question, search memory (with graph when possible, 3 retries then fallback),
    then answer with LLM. Search uses experiment-scoped user_id (experiment_id:user_id) so only this experiment's data is queried.
    Logs to benchmark_logs/<experiment_id>/:
      - stage2_qa_<timestamp>.json (full log)
      - stage2_qa_<timestamp>_stream.jsonl (one QA record per line; when resuming, appends to resume_from_stream so stream matches main log)
      - stage2_qa_<timestamp>_answers.json (per-user answers)
    Each QA record includes: retrieved_memories, mem0_answer, correct_answer, user_query, and all row metadata.

    resume_from_stream: If set and the file exists, already-processed question_ids are skipped and new results
    are appended to this same file (when stream_path is not explicitly provided), keeping stream and main JSON consistent.
    enable_graph: If False, graph (Neo4j) is disabled for this experiment; only vector store is used.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    bundles = load_personamem_jsonl_for_mem0(jsonl_path, split=split)
    if max_users is not None:
        bundles = bundles[:max_users]

    log_dir = get_experiment_log_dir(experiment_id)
    ts = _timestamp_suffix()
    run_idx = _next_stage_run_index(log_dir, "stage2")
    run_tag = f"RUN-{run_idx}"
    out_path = log_dir / f"stage2_{run_tag}_qa_{ts}.json"
    answers_path = log_dir / f"stage2_{run_tag}_qa_{ts}_answers.json"
    flow_log_path = log_dir / f"flow_stage2_{run_tag}_{ts}.log"
    if stream_path is None:
        # When resuming, append to the same stream file so stream and main JSON stay consistent.
        stream_path = Path(resume_from_stream) if resume_from_stream and Path(resume_from_stream).is_file() else None
        if stream_path is None:
            stream_path = log_dir / f"stage2_{run_tag}_qa_{ts}_stream.jsonl"

    from .mem0_full_stack import build_full_mem0_config
    cfg = build_full_mem0_config(enable_graph=enable_graph, experiment_id=experiment_id)

    log = Stage2Log(
        stage="question_answering",
        jsonl_path=str(jsonl_path),
        config=cfg,
        mem0_version=MEM0_VERSION,
        openai_version=OPENAI_VERSION,
    )

    processed_qids: set = set()
    if resume_from_stream and Path(resume_from_stream).is_file():
        with open(resume_from_stream, "r", encoding="utf-8") as sf:
            for line in sf:
                line = line.strip()
                if not line:
                    continue
                try:
                    qa = json.loads(line)
                except Exception:
                    continue
                qid = qa.get("question_id")
                if qid:
                    processed_qids.add(qid)
                    log.per_qa.append(qa)

    failed_stage1_users: set[int] = set()
    if skip_failed_stage1_users:
        resolved_stage1 = Path(stage1_log_path) if stage1_log_path else _latest_stage1_log_path(log_dir)
        if resolved_stage1 is not None and resolved_stage1.is_file():
            failed_stage1_users = _load_failed_stage1_user_ids(resolved_stage1)
            if failed_stage1_users:
                cfg["filtered_failed_stage1_users"] = {
                    "stage1_log_path": str(resolved_stage1),
                    "num_filtered_users": len(failed_stage1_users),
                }

    qa_pairs: List[Tuple[Any, int, Dict]] = []
    for b in bundles:
        if failed_stage1_users and int(b.user_id) in failed_stage1_users:
            continue
        rows = b.benchmark_rows if max_qa_per_user is None else b.benchmark_rows[:max_qa_per_user]
        for i, row in enumerate(rows):
            qa_pairs.append((b, i, row))

    total_wall = 0.0
    total_search = 0
    total_input_tokens = 0
    total_answer_llm_calls = 0
    total_answer_input_tokens = 0
    total_answer_output_tokens = 0
    total_answer_wall_seconds = 0.0
    images_processed_per_user = {
        str(b.user_id): count_images_in_chat_history(b.chat_history) for b in bundles
    }
    log.images_processed_per_user = images_processed_per_user
    log.total_images_processed = sum(images_processed_per_user.values())

    # Seed aggregate counters from resumed rows so summary reflects full output.
    if log.per_qa:
        for qa in log.per_qa:
            total_wall += float(qa.get("wall_seconds", 0.0))
            total_search += int(qa.get("search_calls", 0))
            itoks = qa.get("input_tokens")
            if itoks is not None:
                total_input_tokens += int(itoks)
            total_answer_llm_calls += int(qa.get("answer_llm_calls", 0))
            total_answer_wall_seconds += float(qa.get("answer_wall_seconds", 0.0) or 0.0)
            a_in = qa.get("answer_input_tokens")
            if a_in is not None:
                total_answer_input_tokens += int(a_in)
            a_out = qa.get("answer_output_tokens")
            if a_out is not None:
                total_answer_output_tokens += int(a_out)

    total_qa = len(qa_pairs)
    qa_processed = 0
    stream_f = open(stream_path, "a", encoding="utf-8")
    num_resumed = len(processed_qids)
    with open(flow_log_path, "w", encoding="utf-8") as flow_stream:
        _flow_log_line(flow_stream, f"Stage 2 QA | experiment_id={experiment_id} | num_qa_pairs={total_qa}")
        if num_resumed:
            _flow_log_line(flow_stream, f"Resuming: {num_resumed} already done (from stream), processing remaining.")
        _flow_log_line(flow_stream, "Creating Memory client...")
        mem, _used_graph = _get_single_mem_for_experiment(
            experiment_id,
            graph_retries,
            enable_graph,
            rerank,
        )
        _flow_log_line(flow_stream, "Memory client ready. Processing QA pairs.")
        try:
            for bundle, row_index, row in tqdm(qa_pairs, desc="Stage 2 (QA)", unit="qa"):
                query = row.get("user_query") or ""
                qid = f"{bundle.user_id}:{row_index}"
                if qid in processed_qids:
                    continue

                qa_processed += 1
                approx_tokens = (
                    row.get("total_tokens_in_chat_history_32k")
                    or (row.get("num_persona_relevant_tokens_32k") or 0) + (row.get("num_persona_irrelevant_tokens_32k") or 0)
                )

                query_str = _query_to_search_string(query)
                if not query_str:
                    rec = Stage2QARecord(
                        user_id=bundle.user_id,
                        row_index=row_index,
                        question_id=qid,
                        user_query="",
                        user_images_processed=images_processed_per_user.get(str(bundle.user_id), 0),
                        wall_seconds=0.0,
                        success=False,
                        error="missing user_query",
                        input_tokens=approx_tokens,
                        used_graph=_used_graph,
                        correct_answer=row.get("correct_answer"),
                        incorrect_answers=row.get("incorrect_answers"),
                        short_persona=row.get("short_persona"),
                        expanded_persona=row.get("expanded_persona"),
                        preference=row.get("preference"),
                        topic_preference=row.get("topic_preference"),
                        pref_type=row.get("pref_type"),
                        sensitive_info=row.get("sensitive_info"),
                        conversation_scenario=row.get("conversation_scenario"),
                        topic_query=row.get("topic_query"),
                    )
                    qa_dict = asdict(rec)
                    log.per_qa.append(qa_dict)
                    stream_f.write(json.dumps(qa_dict, ensure_ascii=False) + "\n")
                    continue

                _flow_log_line(flow_stream, f"QA {qa_processed}/{total_qa} (user_id={bundle.user_id}, row={row_index}) | search starting")
                search_query = query_str
                t0 = time.perf_counter()
                search_error: Optional[Exception] = None
                results_dict: Optional[Dict] = None

                for attempt in range(graph_retries):
                    try:
                        res = mem.search(
                            search_query,
                            user_id=_scoped_user_id(experiment_id, bundle.user_id),
                            limit=10,
                            rerank=rerank,
                        )
                        results_dict = res
                        search_error = None
                        break
                    except Exception as e:
                        search_error = e
                        if attempt < graph_retries - 1:
                            time.sleep(1.0 * (attempt + 1))

                search_wall_pre = time.perf_counter() - t0
                if search_error is not None or results_dict is None:
                    _flow_log_line(flow_stream, f"QA {qa_processed}/{total_qa} | search failed ({search_wall_pre:.1f}s)")
                else:
                    _flow_log_line(flow_stream, f"QA {qa_processed}/{total_qa} | search done ({len(results_dict.get('results') or [])} results, {search_wall_pre:.1f}s)")

                if search_error is not None or results_dict is None:
                    rec = Stage2QARecord(
                        user_id=bundle.user_id,
                        row_index=row_index,
                        question_id=qid,
                        user_query=query_str[:500],
                        user_images_processed=images_processed_per_user.get(str(bundle.user_id), 0),
                        wall_seconds=time.perf_counter() - t0,
                        success=False,
                        error=str(search_error) if search_error else "search failed",
                        input_tokens=approx_tokens,
                        used_graph=_used_graph,
                        correct_answer=row.get("correct_answer"),
                        incorrect_answers=row.get("incorrect_answers"),
                        short_persona=row.get("short_persona"),
                        expanded_persona=row.get("expanded_persona"),
                        preference=row.get("preference"),
                        topic_preference=row.get("topic_preference"),
                        pref_type=row.get("pref_type"),
                        sensitive_info=row.get("sensitive_info"),
                        conversation_scenario=row.get("conversation_scenario"),
                        topic_query=row.get("topic_query"),
                    )
                else:
                    hits = results_dict.get("results") or []
                    memories = [h.get("memory") for h in hits if h.get("memory")]
                    search_wall = time.perf_counter() - t0
                    total_search += 1

                    _flow_log_line(flow_stream, f"QA {qa_processed}/{total_qa} | answer LLM starting")
                    mem0_answer, ans_in_tok, ans_out_tok, answer_err, answer_wall = _answer_with_llm(
                        search_query, memories, model=LLM_MODEL
                    )
                    _flow_log_line(flow_stream, f"QA {qa_processed}/{total_qa} | answer LLM done ({(answer_wall or 0):.1f}s)")
                    rec = Stage2QARecord(
                        user_id=bundle.user_id,
                        row_index=row_index,
                        question_id=qid,
                        user_query=query_str[:500],
                        user_images_processed=images_processed_per_user.get(str(bundle.user_id), 0),
                        wall_seconds=search_wall,
                        success=True,
                        search_calls=1,
                        num_results=len(hits),
                        retrieved_memories=memories,
                        mem0_answer=mem0_answer,
                        answer_llm_calls=1,
                        answer_input_tokens=ans_in_tok if ans_in_tok else None,
                        answer_output_tokens=ans_out_tok if ans_out_tok else None,
                        answer_error=answer_err,
                        answer_wall_seconds=answer_wall,
                        input_tokens=approx_tokens,
                        used_graph=_used_graph,
                        correct_answer=row.get("correct_answer"),
                        incorrect_answers=row.get("incorrect_answers"),
                        short_persona=row.get("short_persona"),
                        expanded_persona=row.get("expanded_persona"),
                        preference=row.get("preference"),
                        topic_preference=row.get("topic_preference"),
                        pref_type=row.get("pref_type"),
                        sensitive_info=row.get("sensitive_info"),
                        conversation_scenario=row.get("conversation_scenario"),
                        topic_query=row.get("topic_query"),
                    )
                    if answer_err:
                        log.errors.append(f"user_id={bundle.user_id} row={row_index} answer_llm: {answer_err}")

                total_wall += rec.wall_seconds
                if rec.input_tokens is not None:
                    total_input_tokens += rec.input_tokens
                total_answer_llm_calls += rec.answer_llm_calls
                if rec.answer_input_tokens is not None:
                    total_answer_input_tokens += rec.answer_input_tokens
                if rec.answer_output_tokens is not None:
                    total_answer_output_tokens += rec.answer_output_tokens
                if getattr(rec, "answer_wall_seconds", None) is not None:
                    total_answer_wall_seconds += rec.answer_wall_seconds

                qa_dict = asdict(rec)
                log.per_qa.append(qa_dict)
                stream_f.write(json.dumps(qa_dict, ensure_ascii=False) + "\n")
        finally:
            stream_f.close()

    print(f"Flow log: {flow_log_path}")
    log.num_qa_pairs = len(log.per_qa)
    log.total_wall_seconds = total_wall
    log.total_search_calls = total_search
    log.total_input_tokens = total_input_tokens or None
    log.total_answer_llm_calls = total_answer_llm_calls
    log.total_answer_input_tokens = total_answer_input_tokens or None
    log.total_answer_output_tokens = total_answer_output_tokens or None
    log.total_answer_wall_seconds = total_answer_wall_seconds or None

    payload = asdict(log)
    payload["config"] = _redact_secrets(payload.get("config", {}))
    payload["summary"] = log.summary_dict()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Stage 2 log written to {out_path}")

    answers_by_user: Dict[str, Any] = {}
    for qa in log.per_qa:
        uid = str(qa["user_id"])
        answers_by_user.setdefault(uid, {"user_id": uid, "qa": []})["qa"].append({
            "question_id": qa.get("question_id"),
            "row_index": qa["row_index"],
            "user_query": qa.get("user_query"),
            "user_images_processed": qa.get("user_images_processed"),
            "success": qa["success"],
            "error": qa.get("error"),
            "num_results": qa.get("num_results", 0),
            "retrieved_memories": qa.get("retrieved_memories") or [],
            "mem0_answer": qa.get("mem0_answer"),
            "correct_answer": qa.get("correct_answer"),
            "incorrect_answers": qa.get("incorrect_answers"),
            "short_persona": qa.get("short_persona"),
            "expanded_persona": qa.get("expanded_persona"),
            "preference": qa.get("preference"),
            "topic_preference": qa.get("topic_preference"),
            "pref_type": qa.get("pref_type"),
            "sensitive_info": qa.get("sensitive_info"),
            "conversation_scenario": qa.get("conversation_scenario"),
            "topic_query": qa.get("topic_query"),
            "answer_llm_calls": qa.get("answer_llm_calls", 0),
            "answer_input_tokens": qa.get("answer_input_tokens"),
            "answer_output_tokens": qa.get("answer_output_tokens"),
            "answer_error": qa.get("answer_error"),
            "answer_wall_seconds": qa.get("answer_wall_seconds"),
            "wall_seconds": qa.get("wall_seconds"),
            "used_graph": qa.get("used_graph"),
        })
    with open(answers_path, "w", encoding="utf-8") as f:
        json.dump({"jsonl_path": log.jsonl_path, "summary": log.summary_dict(), "per_user": list(answers_by_user.values())}, f, indent=2, ensure_ascii=False)
    print(f"Stage 2 answers written to {answers_path}")

    return log
