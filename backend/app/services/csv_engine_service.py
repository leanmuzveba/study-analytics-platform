"""
Bridges the API layer to the two pieces that already work standalone:
  1. The compiled C++ engine (cpp_engine/build/studytok_engine) — parses
     and validates a raw CSV, rejecting bad rows.
  2. ml/feature_engineering.py — aggregates the clean CSV into
     per-student features.

This module just orchestrates: raw upload -> C++ engine -> clean CSV ->
Python feature pipeline -> DataFrame the model service can predict on.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
CPP_ENGINE_BINARY = REPO_ROOT / "cpp_engine" / "build" / "studytok_engine"

# So we can import ml/feature_engineering.py without turning ml/ into an
# installed package — this is a small portfolio project, not a monorepo.
sys.path.insert(0, str(REPO_ROOT / "ml"))
from feature_engineering import build_student_features  # noqa: E402


class CppEngineNotBuiltError(RuntimeError):
    """Raised when the C++ engine hasn't been compiled yet."""


class ParseSummary:
    def __init__(self, rows_read: int, rows_accepted: int, rows_rejected: int):
        self.rows_read = rows_read
        self.rows_accepted = rows_accepted
        self.rows_rejected = rows_rejected


def _run_cpp_engine(raw_csv_path: Path, clean_csv_path: Path) -> ParseSummary:
    if not CPP_ENGINE_BINARY.exists():
        raise CppEngineNotBuiltError(
            f"C++ engine not found at {CPP_ENGINE_BINARY}. "
            "Build it first: g++ -std=c++17 -O2 -Icpp_engine/include "
            "cpp_engine/src/*.cpp -o cpp_engine/build/studytok_engine"
        )

    result = subprocess.run(
        [str(CPP_ENGINE_BINARY), str(raw_csv_path), str(clean_csv_path)],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"C++ engine failed: {result.stderr.strip()}")

    stdout = result.stdout
    rows_read = int(re.search(r"Rows read:\s+(\d+)", stdout).group(1))
    rows_accepted = int(re.search(r"Rows accepted:\s+(\d+)", stdout).group(1))
    rows_rejected = int(re.search(r"Rows rejected:\s+(\d+)", stdout).group(1))
    return ParseSummary(rows_read, rows_accepted, rows_rejected)


def process_raw_csv(raw_csv_bytes: bytes) -> tuple[ParseSummary, pd.DataFrame]:
    """
    Takes raw uploaded CSV bytes, runs them through the C++ engine then
    the feature pipeline, and returns (parse_summary, student_features_df).
    """
    with tempfile.TemporaryDirectory() as tmp:
        raw_path = Path(tmp) / "raw.csv"
        clean_path = Path(tmp) / "clean.csv"
        raw_path.write_bytes(raw_csv_bytes)

        summary = _run_cpp_engine(raw_path, clean_path)
        features_df = build_student_features(str(clean_path))

    return summary, features_df
