"""
PersonaMem benchmark: two-stage evaluation with Mem0.

Stage 1: Fill memory for each user (ingest chat_history into Mem0).
Stage 2: Question answering (run user_query against memory, log results).

Uses tqdm for progress. Writes one dedicated .json log file per stage to
benchmark_logs/ with:
  - Wall-clock time (per item and total)
  - LLM/API call counts (add_calls, search_calls)
  - Optional token counts (input_tokens, output_tokens) if you add a custom
    OpenAI client wrapper that records response.usage and inject it into the log
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _normalize_user_query(raw: str) -> str:
    """Extract actual question text when user_query is a stringified dict like {"role":"user","content":"..."}."""
    if not raw or not raw.strip():
        return raw
    s = raw.strip()
    if (s.startswith("{") and "content" in s) or (s.startswith("'") and "content" in s):
        try:
            # Accept both JSON and Python literal (single-quoted) forms
            obj = json.loads(s) if s.startswith("{") else ast.literal_eval(s)
            if isinstance(obj, dict) and "content" in obj:
                return (obj["content"] or raw).strip()
        except (json.JSONDecodeError, ValueError, SyntaxError):
            pass
    return raw


def _count_images_in_message_content(content: Any) -> int:
    """
    Count image blocks in one message content.

    Supports multimodal OpenAI-style list content:
    [{"type": "text", ...}, {"type": "image_url", ...}, ...]
    """
    if not isinstance(content, list):
        return 0
    n = 0
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type in {"image_url", "input_image", "image"}:
            n += 1
    return n


def count_images_in_chat_history(chat_history: List[Dict[str, Any]]) -> int:
    """Count total image blocks across a user's chat history."""
    total = 0
    for msg in chat_history:
        if not isinstance(msg, dict):
            continue
        total += _count_images_in_message_content(msg.get("content"))
    return total

from tqdm import tqdm

from .paths import (
    ENV_FILE,
    resolve_benchmark_logs_dir,
    resolve_input_file,
)

# Load .env so OPENAI_API_KEY etc. are set when run from CLI
try:
    from dotenv import load_dotenv
    load_dotenv(ENV_FILE)
except ImportError:
    pass

from .personamem_mem0_prep import load_personamem_jsonl_for_mem0, UserMem0Bundle
from .mem0_full_stack import get_mem, get_async_mem, build_full_mem0_config, LLM_MODEL
from mem0 import __version__ as MEM0_VERSION
try:
    from openai import OpenAI, __version__ as OPENAI_VERSION
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[misc, assignment]
    OPENAI_VERSION = None

# Retry config for answer LLM
_ANSWER_LLM_MAX_RETRIES = 3
_ANSWER_LLM_RETRY_BACKOFF_SEC = 2.0


def _answer_with_llm(
    question: str,
    retrieved_memories: List[str],
    model: str = LLM_MODEL,
) -> Tuple[Optional[str], int, int, Optional[str], float]:
    """
    Call OpenAI chat to produce an answer from the question and retrieved memories.
    Returns (answer_text, input_tokens, output_tokens, error_message, answer_wall_seconds).
    On failure returns (None, 0, 0, error_message, 0.0). Tracks LLM calls, tokens, and per-answer generation time.
    """
    if OpenAI is None:
        return None, 0, 0, "openai package not available", 0.0
    try:
        from .prompts import MEM0_ANSWER_SYSTEM_PROMPT
        system = MEM0_ANSWER_SYSTEM_PROMPT.strip()
    except ImportError:
        system = (
            "You are a helpful assistant. Answer the user's question concisely using ONLY the provided retrieved memories. "
            "If the memories do not contain enough information, say so briefly. Do not invent details."
        )
    client = OpenAI()
    memories_block = "\n".join(f"- {m}" for m in retrieved_memories) if retrieved_memories else "(No relevant memories retrieved.)"
    user_content = f"Retrieved memories:\n{memories_block}\n\nUser question: {question}"
    last_error: Optional[str] = None
    for attempt in range(_ANSWER_LLM_MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
            )
            answer_wall = time.perf_counter() - t0
            choice = (resp.choices or [None])[0]
            if not choice or not getattr(choice, "message", None):
                last_error = "empty response"
                continue
            text = (choice.message.content or "").strip()
            usage = getattr(resp, "usage", None)
            in_tok = int(usage.prompt_tokens) if usage and getattr(usage, "prompt_tokens", None) is not None else 0
            out_tok = int(usage.completion_tokens) if usage and getattr(usage, "completion_tokens", None) is not None else 0
            return text, in_tok, out_tok, None, answer_wall
        except Exception as e:
            last_error = str(e)
            if attempt < _ANSWER_LLM_MAX_RETRIES - 1:
                time.sleep(_ANSWER_LLM_RETRY_BACKOFF_SEC * (attempt + 1))
    return None, 0, 0, last_error, 0.0


# Default paths
DEFAULT_JSONL = resolve_input_file("personamem_benchmark_text_32k_user_bundles.jsonl")
LOG_DIR = resolve_benchmark_logs_dir()


@dataclass
class Stage1UserRecord:
    user_id: int
    num_messages: int
    wall_seconds: float
    success: bool
    images_processed: int = 0
    error: Optional[str] = None
    add_calls: int = 1
    # Token/usage if we can get it from Mem0 or a wrapper (optional)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    # Graph memory usage
    used_graph: Optional[bool] = None
    graph_error: Optional[str] = None
    # Image ingestion observability
    had_text_only_fallback: bool = False
    image_ingest_mode: Optional[str] = None  # "none" | "vision" | "text_only_fallback"
    image_vision_succeeded: Optional[bool] = None


@dataclass
class Stage1Log:
    stage: str = "fill_memory"
    jsonl_path: str = ""
    num_users: int = 0
    total_wall_seconds: float = 0.0
    total_add_calls: int = 0
    total_images_processed: int = 0
    total_input_tokens: Optional[int] = None
    total_output_tokens: Optional[int] = None
    per_user: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    mem0_version: Optional[str] = None
    openai_version: Optional[str] = None

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "num_users": self.num_users,
            "total_wall_seconds": self.total_wall_seconds,
            "total_add_calls": self.total_add_calls,
            "total_images_processed": self.total_images_processed,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "num_errors": len(self.errors),
        }


@dataclass
class Stage2QARecord:
    user_id: int
    row_index: int
    question_id: str
    user_query: str
    wall_seconds: float
    success: bool
    user_images_processed: Optional[int] = None
    error: Optional[str] = None
    search_calls: int = 1
    num_results: int = 0
    retrieved_memories: Optional[List[str]] = None
    # LLM-generated answer from question + retrieved_memories (same model as Mem0: gpt-4o-mini)
    mem0_answer: Optional[str] = None
    answer_llm_calls: int = 0
    answer_input_tokens: Optional[int] = None
    answer_output_tokens: Optional[int] = None
    answer_error: Optional[str] = None
    answer_wall_seconds: Optional[float] = None  # per-answer LLM generation time
    # Benchmark supervision / metadata
    correct_answer: Optional[str] = None
    incorrect_answers: Optional[List[str]] = None
    short_persona: Optional[str] = None
    expanded_persona: Optional[str] = None
    preference: Optional[str] = None
    topic_preference: Optional[str] = None
    pref_type: Optional[str] = None
    sensitive_info: Optional[bool] = None
    conversation_scenario: Optional[str] = None
    topic_query: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    used_graph: Optional[bool] = None
    graph_error: Optional[str] = None


@dataclass
class Stage2Log:
    stage: str = "question_answering"
    jsonl_path: str = ""
    num_qa_pairs: int = 0
    total_wall_seconds: float = 0.0
    total_search_calls: int = 0
    total_images_processed: int = 0
    images_processed_per_user: Dict[str, int] = field(default_factory=dict)
    total_input_tokens: Optional[int] = None
    total_output_tokens: Optional[int] = None
    total_answer_llm_calls: int = 0
    total_answer_input_tokens: Optional[int] = None
    total_answer_output_tokens: Optional[int] = None
    total_answer_wall_seconds: Optional[float] = None  # sum of per-answer LLM generation time
    per_qa: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    mem0_version: Optional[str] = None
    openai_version: Optional[str] = None

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "num_qa_pairs": self.num_qa_pairs,
            "total_wall_seconds": self.total_wall_seconds,
            "total_search_calls": self.total_search_calls,
            "total_images_processed": self.total_images_processed,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_answer_llm_calls": self.total_answer_llm_calls,
            "total_answer_input_tokens": self.total_answer_input_tokens,
            "total_answer_output_tokens": self.total_answer_output_tokens,
            "total_answer_wall_seconds": self.total_answer_wall_seconds,
            "num_errors": len(self.errors),
        }


def _ensure_log_dir() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR


def _timestamp_suffix() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


# ---------- Stage 1: Fill memory ----------


def run_stage1_fill_memory(
    jsonl_path: Path,
    *,
    split: str = "benchmark_text",
    max_users: Optional[int] = None,
    use_async: bool = True,
    max_concurrent: int = 5,
    log_path: Optional[Path] = None,
    enable_graph: bool = False,
) -> Stage1Log:
    """
    Load PersonaMem bundles and add each user's chat_history to Mem0.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    bundles = load_personamem_jsonl_for_mem0(jsonl_path, split=split)
    if max_users is not None:
        bundles = bundles[:max_users]

    cfg = build_full_mem0_config(enable_graph=enable_graph)
    log = Stage1Log(
        jsonl_path=str(jsonl_path),
        num_users=len(bundles),
        config=cfg,
        mem0_version=MEM0_VERSION,
        openai_version=OPENAI_VERSION,
    )
    _ensure_log_dir()
    out_path = log_path or (LOG_DIR / f"stage1_fill_memory_{_timestamp_suffix()}.json")

    if use_async:
        log = asyncio.run(_run_stage1_async(bundles, log, out_path, max_concurrent, enable_graph))
    else:
        log = _run_stage1_sync(bundles, log, out_path, enable_graph)

    payload = asdict(log)
    payload["summary"] = log.summary_dict()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Stage 1 log written to {out_path}")
    return log


def _run_stage1_sync(
    bundles: List[UserMem0Bundle],
    log: Stage1Log,
    out_path: Path,
    enable_graph: bool = False,
) -> Stage1Log:
    mem = get_mem(enable_graph=enable_graph)
    mem_plain: Optional[Any] = None  # fallback client without graph
    total_add_calls = 0
    total_images_processed = 0
    total_wall = 0.0
    total_input_tokens = 0

    for bundle in tqdm(bundles, desc="Stage 1 (fill memory)", unit="user"):
        meta = bundle.metadata or {}
        approx_tokens = meta.get("final_token_count")
        image_count = count_images_in_chat_history(bundle.chat_history)

        t0 = time.perf_counter()
        error: Optional[Exception] = None
        graph_error: Optional[str] = None
        graph_disabled = False

        for attempt in range(2):  # simple retry once on failure
            try:
                client = mem if not (enable_graph and graph_disabled and mem_plain) else (mem_plain or mem)
                client.add(
                    bundle.chat_history,
                    user_id=str(bundle.user_id),
                    metadata={**(bundle.metadata or {}), "benchmark": "personamem"},
                )
                error = None
                total_add_calls += 1
                break
            except Exception as e:  # keep error and retry once
                error = e
                msg = str(e)
                # If graph is enabled and the failure looks like a graph extraction issue,
                # remember it and retry once with graph disabled by switching to a non-graph client.
                if enable_graph and (("entity_type" in msg) or ("graph" in msg)) and not graph_disabled:
                    graph_error = msg
                    graph_disabled = True
                    if mem_plain is None:
                        mem_plain = get_mem(enable_graph=False)
                    continue
                if attempt == 0:
                    continue

        if error is None:
            rec = Stage1UserRecord(
                user_id=bundle.user_id,
                num_messages=len(bundle.chat_history),
                images_processed=image_count,
                wall_seconds=time.perf_counter() - t0,
                success=True,
                add_calls=1,
                input_tokens=approx_tokens,
                used_graph=enable_graph and not graph_disabled,
                graph_error=graph_error,
            )
        else:
            rec = Stage1UserRecord(
                user_id=bundle.user_id,
                num_messages=len(bundle.chat_history),
                images_processed=image_count,
                wall_seconds=time.perf_counter() - t0,
                success=False,
                error=str(error),
                input_tokens=approx_tokens,
                used_graph=enable_graph and not graph_disabled,
                graph_error=graph_error,
            )
            log.errors.append(f"user_id={bundle.user_id}: {error}")
        total_wall += rec.wall_seconds
        total_images_processed += rec.images_processed
        if rec.input_tokens is not None:
            total_input_tokens += rec.input_tokens
        log.per_user.append(asdict(rec))

    log.total_wall_seconds = total_wall
    log.total_add_calls = total_add_calls
    log.total_images_processed = total_images_processed
    if total_input_tokens:
        log.total_input_tokens = total_input_tokens
    return log


async def _run_stage1_async(
    bundles: List[UserMem0Bundle],
    log: Stage1Log,
    out_path: Path,
    max_concurrent: int,
    enable_graph: bool = False,
) -> Stage1Log:
    amem = await get_async_mem(enable_graph=enable_graph)
    amem_plain: Optional[Any] = None  # fallback client without graph
    sem = asyncio.Semaphore(max_concurrent)

    async def add_one(bundle: UserMem0Bundle) -> Stage1UserRecord:
        meta = bundle.metadata or {}
        approx_tokens = meta.get("final_token_count")
        image_count = count_images_in_chat_history(bundle.chat_history)

        t0 = time.perf_counter()
        async with sem:
            error: Optional[Exception] = None
            graph_error: Optional[str] = None
            graph_disabled = False

            for attempt in range(2):  # simple retry once on failure
                try:
                    client = amem if not (enable_graph and graph_disabled and amem_plain) else (amem_plain or amem)
                    await client.add(
                        bundle.chat_history,
                        user_id=str(bundle.user_id),
                        metadata={**(bundle.metadata or {}), "benchmark": "personamem"},
                    )
                    error = None
                    break
                except Exception as e:
                    error = e
                    msg = str(e)
                    if enable_graph and (("entity_type" in msg) or ("graph" in msg)) and not graph_disabled:
                        graph_error = msg
                        graph_disabled = True
                        if amem_plain is None:
                            amem_plain = await get_async_mem(enable_graph=False)
                        continue
                    if attempt == 0:
                        continue

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
                    used_graph=enable_graph and not graph_disabled,
                    graph_error=graph_error,
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
                    used_graph=enable_graph and not graph_disabled,
                    graph_error=graph_error,
                )

    tasks = [add_one(b) for b in bundles]
    results: List[Stage1UserRecord] = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Stage 1 (fill memory)", unit="user"):
        rec = await coro
        results.append(rec)
        if rec.error:
            log.errors.append(f"user_id={rec.user_id}: {rec.error}")

    # Sort by user_id for stable log
    results.sort(key=lambda r: r.user_id)
    log.per_user = [asdict(r) for r in results]
    log.total_wall_seconds = sum(r.wall_seconds for r in results)
    log.total_add_calls = sum(r.add_calls for r in results if r.success)
    log.total_images_processed = sum(r.images_processed for r in results)
    token_sum = sum((r.input_tokens or 0) for r in results)
    if token_sum:
        log.total_input_tokens = token_sum
    return log


# ---------- Stage 2: Question answering ----------


def run_stage2_qa(
    jsonl_path: Path,
    *,
    split: str = "benchmark_text",
    max_users: Optional[int] = None,
    max_qa_per_user: Optional[int] = None,
    rerank: bool = True,
    log_path: Optional[Path] = None,
    enable_graph: bool = False,
    stream_path: Optional[Path] = None,
    resume_from_stream: Optional[Path] = None,
) -> Stage2Log:
    """
    For each user and each benchmark row, run search(user_query) and log results.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    bundles = load_personamem_jsonl_for_mem0(jsonl_path, split=split)
    if max_users is not None:
        bundles = bundles[:max_users]

    cfg = build_full_mem0_config(enable_graph=enable_graph)
    log = Stage2Log(
        jsonl_path=str(jsonl_path),
        config=cfg,
        mem0_version=MEM0_VERSION,
        openai_version=OPENAI_VERSION,
    )
    _ensure_log_dir()
    out_path = log_path or (LOG_DIR / f"stage2_qa_{_timestamp_suffix()}.json")
    answers_path = out_path.with_name(out_path.stem + "_answers.json")
    if stream_path is None:
        stream_path = out_path.with_name(out_path.stem + "_stream.jsonl")

    mem = get_mem(enable_graph=enable_graph)
    total_search = 0
    total_wall = 0.0
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

    # If resuming, load existing QA records from the stream file and mark processed question_ids
    processed_qids: set[str] = set()
    if resume_from_stream is not None:
        resume_path = Path(resume_from_stream)
        if resume_path.is_file():
            with resume_path.open("r", encoding="utf-8") as sf:
                for line in sf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        qa = json.loads(line)
                    except Exception:
                        continue
                    qid = qa.get("question_id")
                    if not qid:
                        continue
                    processed_qids.add(qid)
                    log.per_qa.append(qa)
                    total_wall += float(qa.get("wall_seconds", 0.0))
                    total_search += int(qa.get("search_calls", 1))
                    itoks = qa.get("input_tokens")
                    if itoks is not None:
                        total_input_tokens += int(itoks)
                    total_answer_llm_calls += int(qa.get("answer_llm_calls", 0))
                    total_answer_wall_seconds += float(qa.get("answer_wall_seconds", 0.0))
                    a_in = qa.get("answer_input_tokens")
                    if a_in is not None:
                        total_answer_input_tokens += int(a_in)
                    a_out = qa.get("answer_output_tokens")
                    if a_out is not None:
                        total_answer_output_tokens += int(a_out)
    qa_pairs: List[tuple] = []
    for b in bundles:
        rows = b.benchmark_rows if max_qa_per_user is None else b.benchmark_rows[:max_qa_per_user]
        for i, row in enumerate(rows):
            q = (b.user_id, i, row.get("user_query") or "")
            qa_pairs.append((b, i, row))

    # Open stream file for appending new QA records
    stream_f = Path(stream_path).open("a", encoding="utf-8")
    try:
        for bundle, row_index, row in tqdm(qa_pairs, desc="Stage 2 (QA)", unit="qa"):
            query = row.get("user_query") or ""
            qid = f"{bundle.user_id}:{row_index}"
            approx_tokens = (
                row.get("total_tokens_in_chat_history_32k")
                or (
                    (row.get("num_persona_relevant_tokens_32k") or 0)
                    + (row.get("num_persona_irrelevant_tokens_32k") or 0)
                )
            )
            # Skip QA pairs that are already recorded in the stream (resume mode)
            if qid in processed_qids:
                continue
            if not query:
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
                    used_graph=enable_graph,
                )
                qa_dict = asdict(rec)
                log.per_qa.append(qa_dict)
                stream_f.write(json.dumps(qa_dict, ensure_ascii=False) + "\n")
                continue
            search_query = _normalize_user_query(query)
            t0 = time.perf_counter()
            try:
                results = mem.search(
                    search_query,
                    user_id=str(bundle.user_id),
                    limit=10,
                    rerank=rerank,
                )
                hits = results.get("results") or []
                memories = [h.get("memory") for h in hits if h.get("memory")]
                search_wall = time.perf_counter() - t0
                total_search += 1
                # Generate answer via LLM (same model as Mem0: gpt-4o-mini) from question + retrieved_memories
                mem0_answer, ans_in_tok, ans_out_tok, answer_err, answer_wall = _answer_with_llm(
                    search_query, memories, model=LLM_MODEL
                )
                rec = Stage2QARecord(
                    user_id=bundle.user_id,
                    row_index=row_index,
                    question_id=qid,
                    user_query=query[:200],
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
                    used_graph=enable_graph,
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
            except Exception as e:
                rec = Stage2QARecord(
                    user_id=bundle.user_id,
                    row_index=row_index,
                    question_id=qid,
                    user_query=query[:200],
                    user_images_processed=images_processed_per_user.get(str(bundle.user_id), 0),
                    wall_seconds=time.perf_counter() - t0,
                    success=False,
                    error=str(e),
                    input_tokens=approx_tokens,
                    used_graph=enable_graph,
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
                log.errors.append(f"user_id={bundle.user_id} row={row_index}: {e}")
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

    log.num_qa_pairs = len(log.per_qa)
    log.total_wall_seconds = total_wall
    log.total_search_calls = total_search
    if total_input_tokens:
        log.total_input_tokens = total_input_tokens
    log.total_answer_llm_calls = total_answer_llm_calls
    if total_answer_input_tokens:
        log.total_answer_input_tokens = total_answer_input_tokens
    if total_answer_output_tokens:
        log.total_answer_output_tokens = total_answer_output_tokens
    if total_answer_wall_seconds:
        log.total_answer_wall_seconds = total_answer_wall_seconds

    # Main QA log
    payload = asdict(log)
    payload["summary"] = log.summary_dict()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Stage 2 log written to {out_path}")

    # Additional per-user answers file: groups questions + Mem0 retrieved memories by user.
    answers_by_user: Dict[str, Any] = {}
    for qa in log.per_qa:
        uid = str(qa["user_id"])
        answers_by_user.setdefault(uid, {"user_id": uid, "qa": []})["qa"].append(
            {
                "question_id": qa.get("question_id"),
                "row_index": qa["row_index"],
                "user_query": qa["user_query"],
                "user_images_processed": qa.get("user_images_processed"),
                "success": qa["success"],
                "error": qa.get("error"),
                "num_results": qa["num_results"],
                "retrieved_memories": qa.get("retrieved_memories") or [],
                "mem0_answer": qa.get("mem0_answer"),
                "answer_llm_calls": qa.get("answer_llm_calls", 0),
                "answer_input_tokens": qa.get("answer_input_tokens"),
                "answer_output_tokens": qa.get("answer_output_tokens"),
                "answer_error": qa.get("answer_error"),
                "answer_wall_seconds": qa.get("answer_wall_seconds"),
                "input_tokens": qa.get("input_tokens"),
                "wall_seconds": qa["wall_seconds"],
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
                "used_graph": qa.get("used_graph"),
                "graph_error": qa.get("graph_error"),
            }
        )

    answers_payload = {
        "jsonl_path": log.jsonl_path,
        "summary": log.summary_dict(),
        "per_user": list(answers_by_user.values()),
    }
    with open(answers_path, "w", encoding="utf-8") as f:
        json.dump(answers_payload, f, indent=2, ensure_ascii=False)
    print(f"Stage 2 answers-by-user log written to {answers_path}")

    return log


# ---------- CLI ----------


def main() -> None:
    global LOG_DIR

    parser = argparse.ArgumentParser(description="PersonaMem benchmark: fill memory (stage 1) and QA (stage 2).")
    parser.add_argument(
        "stage",
        choices=["1", "2", "both"],
        help="Run stage 1 (fill memory), stage 2 (QA), or both in order.",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=DEFAULT_JSONL,
        help=f"Path to PersonaMem user bundles JSONL (default: {DEFAULT_JSONL}).",
    )
    parser.add_argument("--split", default="benchmark_text", help="Split name in rows_by_split (default: benchmark_text).")
    parser.add_argument("--max-users", type=int, default=None, help="Cap number of users (for quick runs).")
    parser.add_argument("--max-qa-per-user", type=int, default=None, help="Cap QA pairs per user in stage 2.")
    parser.add_argument("--sync", action="store_true", help="Stage 1: use sync Memory instead of async.")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Stage 1 async: max concurrent add() calls.")
    parser.add_argument("--no-rerank", action="store_true", help="Stage 2: disable reranker on search.")
    parser.add_argument("--enable-graph", action="store_true", help="Enable Neo4j graph storage (requires NEO4J_URL, NEO4J_PASSWORD).")
    parser.add_argument("--qa-stream", type=Path, default=None, help="Optional JSONL stream log for Stage 2 QA.")
    parser.add_argument("--qa-resume-from", type=Path, default=None, help="Resume Stage 2 QA from an existing JSONL stream.")
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR, help="Directory for JSON log files.")
    args = parser.parse_args()
    LOG_DIR = args.log_dir

    if args.stage in ("1", "both"):
        run_stage1_fill_memory(
            args.jsonl,
            split=args.split,
            max_users=args.max_users,
            use_async=not args.sync,
            max_concurrent=args.max_concurrent,
            enable_graph=args.enable_graph,
        )
    if args.stage in ("2", "both"):
        run_stage2_qa(
            args.jsonl,
            split=args.split,
            max_users=args.max_users,
            max_qa_per_user=args.max_qa_per_user,
            rerank=not args.no_rerank,
            enable_graph=args.enable_graph,
            stream_path=args.qa_stream,
            resume_from_stream=args.qa_resume_from,
        )


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    main()
