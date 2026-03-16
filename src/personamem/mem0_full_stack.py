import os
from typing import Any, Dict, Optional

from mem0 import Memory, AsyncMemory


EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMS = 1536  # OpenAI text-embedding-3-small
LLM_MODEL = "gpt-4.1-mini"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _default_rerank_device() -> str:
    """
    Choose a sensible default device for the reranker.

    - On Apple Silicon, prefer MPS if available.
    - Otherwise fall back to CPU.
    """
    # Allow overriding via env for quick experiments.
    env_device = os.environ.get("MEM0_RERANK_DEVICE")
    if env_device:
        return env_device

    # Rough heuristic: on macOS ARM, assume MPS-capable PyTorch is installed.
    if os.uname().sysname == "Darwin" and os.uname().machine in {"arm64", "aarch64"}:
        return "mps"

    return "cpu"


def _experiment_vector_store_path(experiment_id: str) -> str:
    """Persistent on-disk path for an experiment's Qdrant data (one dir per experiment_id)."""
    base = os.environ.get("MEM0_DIR") or os.path.join(os.path.expanduser("~"), ".mem0")
    return os.path.join(base, "experiments", experiment_id, "qdrant")


def _normalize_neo4j_url(url: Optional[str]) -> Optional[str]:
    """
    Normalize Neo4j URL for common local setup pitfalls.

    Local single-instance Neo4j often works best with `bolt://localhost:7687`.
    Using `neo4j://` may trigger routing-table errors when routing is unavailable.
    """
    if not url:
        return url
    s = url.strip()
    lower = s.lower()

    is_localhost = (
        "localhost" in lower
        or "127.0.0.1" in lower
        or "0.0.0.0" in lower
    )
    if is_localhost and (lower.startswith("neo4j://") or lower.startswith("neo4j+s://")):
        # Keep host:port/path and swap scheme to non-routing bolt.
        return "bolt://" + s.split("://", 1)[1]

    return s


def build_full_mem0_config(
    *,
    enable_graph: Optional[bool] = None,
    enable_vision: Optional[bool] = None,
    enable_reranker: bool = True,
    rerank_device: Optional[str] = None,
    experiment_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a complete Mem0 config:

    - LLM: OpenAI (LLM_MODEL), with optional vision parsing enabled
    - Embedder: OpenAI (EMBEDDING_MODEL, EMBEDDING_DIMS)
    - Vector store: Qdrant (embedding_model_dims=EMBEDDING_DIMS).
      If experiment_id is set: persistent on-disk path per experiment.
    - Graph store: Neo4j (optional, controlled by env + enable_graph)
    - Reranker: HuggingFace cross-encoder (local)

    Graph behaviour:
    - If enable_graph is True: require NEO4J_URL / NEO4J_PASSWORD and attach graph_store.
    - If enable_graph is False: never attach graph_store.
    - If enable_graph is None (default): auto-enable when Neo4j env vars exist.

    Vision behaviour:
    - If enable_vision is True: Mem0 parses multimodal messages with the configured LLM.
    - If enable_vision is False: Mem0 uses text-only parsing path for message content.
    - If enable_vision is None (default): enabled by default, can be overridden via
      MEM0_ENABLE_VISION env var ("0"/"false"/"no" to disable).

    experiment_id: If provided, vector store uses a dedicated on-disk path so memory
    persists across sessions and is isolated per experiment.
    """
    vector_config: Dict[str, Any] = {
        "embedding_model_dims": EMBEDDING_DIMS,
    }
    exp_dir: Optional[str] = None
    if experiment_id:
        base = os.environ.get("MEM0_DIR") or os.path.join(os.path.expanduser("~"), ".mem0")
        exp_dir = os.path.join(base, "experiments", experiment_id)
        vector_config["path"] = os.path.join(exp_dir, "qdrant")
        vector_config["on_disk"] = True

    if enable_vision is None:
        env_vision = os.environ.get("MEM0_ENABLE_VISION")
        if env_vision is None:
            resolved_enable_vision = True
        else:
            resolved_enable_vision = env_vision.strip().lower() not in {"0", "false", "no", "off"}
    else:
        resolved_enable_vision = bool(enable_vision)

    cfg: Dict[str, Any] = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": LLM_MODEL,
                "enable_vision": resolved_enable_vision,
                "vision_details": "auto",
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": EMBEDDING_MODEL,
                "embedding_dims": EMBEDDING_DIMS,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": vector_config,
        },
    }
    if enable_reranker:
        cfg["reranker"] = {
            "provider": "huggingface",
            "config": {
                "model": RERANKER_MODEL,
                "device": rerank_device or _default_rerank_device(),
                "batch_size": 32,
                "max_length": 512,
                "top_k": 10,
                "normalize": False,
            },
        }
    if exp_dir is not None:
        cfg["history_db_path"] = os.path.join(exp_dir, "history.db")

    neo4j_url = _normalize_neo4j_url(os.environ.get("NEO4J_URL"))
    neo4j_user = os.environ.get("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD")

    should_attach_graph: bool
    if enable_graph is True:
        should_attach_graph = True
    elif enable_graph is False:
        should_attach_graph = False
    else:
        # Auto mode: only if creds exist.
        should_attach_graph = bool(neo4j_url and neo4j_password)

    if should_attach_graph:
        if not (neo4j_url and neo4j_password):
            raise RuntimeError(
                "Graph requested (enable_graph=True) but NEO4J_URL / NEO4J_PASSWORD are not set."
            )
        cfg["graph_store"] = {
            "provider": "neo4j",
            "config": {
                "url": neo4j_url,
                "username": neo4j_user,
                "password": neo4j_password,
                "database": "neo4j",
            },
            # Explicit graph-LLM config avoids implicit provider/config fallbacks.
            "llm": {
                "provider": "openai",
                "config": {
                    "model": LLM_MODEL,
                    "api_key": os.environ.get("OPENAI_API_KEY"),
                },
            },
        }

    return cfg


def get_mem(
    enable_graph: Optional[bool] = None,
    enable_vision: Optional[bool] = None,
    enable_reranker: bool = True,
    rerank_device: Optional[str] = None,
    experiment_id: Optional[str] = None,
) -> Memory:
    """
    Create a synchronous Memory client with the full stack config.

    If experiment_id is set, vector store is persistent and scoped to that experiment.
    """
    cfg = build_full_mem0_config(
        enable_graph=enable_graph,
        enable_vision=enable_vision,
        enable_reranker=enable_reranker,
        rerank_device=rerank_device,
        experiment_id=experiment_id,
    )
    return Memory.from_config(cfg)


async def get_async_mem(
    enable_graph: Optional[bool] = None,
    enable_vision: Optional[bool] = None,
    enable_reranker: bool = True,
    rerank_device: Optional[str] = None,
    experiment_id: Optional[str] = None,
) -> AsyncMemory:
    """
    Create an AsyncMemory client with the full stack config.

    If experiment_id is set, vector store is persistent and scoped to that experiment.
    """
    cfg = build_full_mem0_config(
        enable_graph=enable_graph,
        enable_vision=enable_vision,
        enable_reranker=enable_reranker,
        rerank_device=rerank_device,
        experiment_id=experiment_id,
    )
    return await AsyncMemory.from_config(cfg)


__all__ = [
    "build_full_mem0_config",
    "_experiment_vector_store_path",
    "get_mem",
    "get_async_mem",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMS",
    "LLM_MODEL",
    "RERANKER_MODEL",
]

