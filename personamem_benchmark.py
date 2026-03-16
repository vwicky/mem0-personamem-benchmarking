"""Compatibility CLI wrapper for moved package module."""

from pathlib import Path
import sys

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from personamem.personamem_benchmark import *  # noqa: F401,F403
from personamem.personamem_benchmark import main


if __name__ == "__main__":
    main()
