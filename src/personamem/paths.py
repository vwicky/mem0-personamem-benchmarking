from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DATA_INPUTS_DIR = DATA_DIR / "inputs"
ANALYSIS_EXPORTS_DIR = ARTIFACTS_DIR / "analysis_exports"
BENCHMARK_LOGS_DIR = ARTIFACTS_DIR / "benchmark_logs"

# Legacy locations for backward compatibility during migration.
LEGACY_EXPORTS_DIR = PROJECT_ROOT / "exports"
LEGACY_BENCHMARK_LOGS_DIR = PROJECT_ROOT / "benchmark_logs"


def resolve_input_file(filename: str) -> Path:
    """Resolve benchmark input file from new path with legacy fallback."""
    primary = DATA_INPUTS_DIR / filename
    if primary.exists():
        return primary
    return LEGACY_EXPORTS_DIR / filename


def resolve_benchmark_logs_dir() -> Path:
    """
    Return benchmark logs directory, preferring explicit env override,
    then new layout, then legacy path when it already exists.
    """
    env_path = os.environ.get("PERSONAMEM_BENCHMARK_LOGS_DIR")
    if env_path:
        return Path(env_path).expanduser()
    if BENCHMARK_LOGS_DIR.exists():
        return BENCHMARK_LOGS_DIR
    if LEGACY_BENCHMARK_LOGS_DIR.exists():
        return LEGACY_BENCHMARK_LOGS_DIR
    return BENCHMARK_LOGS_DIR
