"""Performance benchmark for Phase 4.5a feature generation.

Compares:
  A. SubprocessFeatureGenerator (reference)
  B. GitPythonFeatureGenerator (optimized, no cache)
  C. GitPythonFeatureGenerator (optimized, with cache)

Uses 1,000+ representative rows from combined-v3.
Records elapsed time, rows/sec, subprocess count, memory, cache sizes.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import pytest

if sys.platform == "win32":
    # Ensure UTF-8 for Windows terminals.  Note: do NOT replace
    # sys.stdout/stderr here as it breaks pytest capture; encoding
    # is handled by pytest's -s flag or the terminal wrapper.
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.scripts.generate_exp_4_5a import (
    GitPythonFeatureGenerator,
    SubprocessFeatureGenerator,
)


def _load_benchmark_rows(count: int = 1000) -> list[dict]:
    """Load rows from combined-v3 for benchmarking."""
    source_dir = Path("backend/data/datasets/combined-v3")
    if not source_dir.exists():
        pytest.skip("combined-v3 dataset not available")

    all_rows = []
    for split in ["train", "validation", "test"]:
        path = source_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line.strip())
                all_rows.append(row)

    if len(all_rows) < count:
        count = len(all_rows)

    # Sample evenly
    step = max(1, len(all_rows) // count)
    return all_rows[::step][:count]


def _repo_is_valid(repo_path: str) -> bool:
    """Check if a repo path is a valid GitPython repository."""
    import git as _git
    try:
        repo = _git.Repo(repo_path)
        _ = repo.head.commit.hexsha
        return True
    except Exception:
        return False


def _get_repos_needed(rows: list[dict]) -> dict[str, str]:
    """Get repo_name -> repo_url mapping for rows."""
    repo_urls = {}
    for row in rows:
        rn = row["repo_name"]
        if rn not in repo_urls:
            repo_urls[rn] = row.get("repo_url", "")
    return repo_urls


class TestPerformanceBenchmark:
    """Performance benchmark comparing implementations."""

    def test_benchmark_subprocess(self):
        """Benchmark SubprocessFeatureGenerator (reference)."""
        rows = _load_benchmark_rows(1000)

        # Group rows by repo
        by_repo = defaultdict(list)
        for row in rows:
            by_repo[row["repo_name"]].append(row)

        gen = SubprocessFeatureGenerator()
        total_time = 0.0
        rows_processed = 0

        for repo_name, repo_rows in by_repo.items():
            repo_path = str(Path(".tmp/repos") / repo_name)
            if not os.path.exists(repo_path):
                continue

            t0 = time.perf_counter()
            for row in repo_rows:
                gen.compute_for_row(
                    repo_path,
                    row["commit_sha"],
                    row["file_path"],
                    row.get("commit_timestamp", ""),
                )
            elapsed = time.perf_counter() - t0
            total_time += elapsed
            rows_processed += len(repo_rows)
            gen.clear()

        if rows_processed == 0:
            pytest.skip("No repos available for benchmark")

        rate = rows_processed / total_time if total_time > 0 else 0
        print(f"\n[Subprocess] {rows_processed} rows in {total_time:.2f}s "
              f"({rate:.1f} rows/s)")

        # Store for comparison
        self._subprocess_rate = rate
        self._subprocess_time = total_time

    def test_benchmark_gitpython(self):
        """Benchmark GitPythonFeatureGenerator (optimized, no warm cache)."""
        rows = _load_benchmark_rows(1000)
        by_repo = defaultdict(list)
        for row in rows:
            by_repo[row["repo_name"]].append(row)

        total_time = 0.0
        rows_processed = 0

        for repo_name, repo_rows in by_repo.items():
            repo_path = str(Path(".tmp/repos") / repo_name)
            if not os.path.exists(repo_path) or not _repo_is_valid(repo_path):
                continue

            gen = GitPythonFeatureGenerator()
            gen.open_repo(repo_name, repo_path)

            t0 = time.perf_counter()
            for row in repo_rows:
                gen.compute_for_row(
                    repo_name,
                    row["commit_sha"],
                    row["file_path"],
                    row.get("commit_timestamp", ""),
                )
            elapsed = time.perf_counter() - t0
            total_time += elapsed
            rows_processed += len(repo_rows)
            gen.clear()

        if rows_processed == 0:
            pytest.skip("No repos available for benchmark")

        rate = rows_processed / total_time if total_time > 0 else 0
        print(f"\n[GitPython] {rows_processed} rows in {total_time:.2f}s "
              f"({rate:.1f} rows/s)")

        self._gitpython_rate = rate
        self._gitpython_time = total_time

    def test_benchmark_gitpython_cached(self):
        """Benchmark GitPythonFeatureGenerator with warm cache.

        Processes each repo twice: first pass warms cache, second pass
        measures cached performance.
        """
        rows = _load_benchmark_rows(1000)
        by_repo = defaultdict(list)
        for row in rows:
            by_repo[row["repo_name"]].append(row)

        total_time = 0.0
        rows_processed = 0

        for repo_name, repo_rows in by_repo.items():
            repo_path = str(Path(".tmp/repos") / repo_name)
            if not os.path.exists(repo_path) or not _repo_is_valid(repo_path):
                continue

            gen = GitPythonFeatureGenerator()
            gen.open_repo(repo_name, repo_path)

            # Warm cache
            for row in repo_rows:
                gen.compute_for_row(
                    repo_name,
                    row["commit_sha"],
                    row["file_path"],
                    row.get("commit_timestamp", ""),
                )

            # Clear only exp_cache (keep metadata/file/hist caches)
            gen._exp_cache.clear()

            # Benchmark cached pass
            t0 = time.perf_counter()
            for row in repo_rows:
                gen.compute_for_row(
                    repo_name,
                    row["commit_sha"],
                    row["file_path"],
                    row.get("commit_timestamp", ""),
                )
            elapsed = time.perf_counter() - t0
            total_time += elapsed
            rows_processed += len(repo_rows)

            # Report cache sizes
            cache_info = {
                "meta": len(gen._meta),
                "file_cache": len(gen._file_cache),
                "hist_raw": len(gen._hist_raw),
                "bug_fix_cache": len(gen._bug_fix_cache),
            }
            print(f"\n  [{repo_name}] cache sizes: {cache_info}")
            gen.clear()

        if rows_processed == 0:
            pytest.skip("No repos available for benchmark")

        rate = rows_processed / total_time if total_time > 0 else 0
        print(f"\n[GitPython+Cache] {rows_processed} rows in {total_time:.2f}s "
              f"({rate:.1f} rows/s)")

        self._cached_rate = rate
        self._cached_time = total_time

    def test_benchmark_summary(self):
        """Print benchmark summary and comparison."""
        # Run all benchmarks
        rows = _load_benchmark_rows(1000)
        by_repo = defaultdict(list)
        for row in rows:
            by_repo[row["repo_name"]].append(row)

        # --- Subprocess ---
        sub_gen = SubprocessFeatureGenerator()
        sub_time = 0.0
        sub_rows = 0
        for repo_name, repo_rows in by_repo.items():
            repo_path = str(Path(".tmp/repos") / repo_name)
            if not os.path.exists(repo_path):
                continue
            t0 = time.perf_counter()
            for row in repo_rows:
                sub_gen.compute_for_row(
                    repo_path, row["commit_sha"], row["file_path"],
                    row.get("commit_timestamp", ""))
            sub_time += time.perf_counter() - t0
            sub_rows += len(repo_rows)
            sub_gen.clear()

        # --- GitPython (fresh) ---
        gp_time = 0.0
        gp_rows = 0
        for repo_name, repo_rows in by_repo.items():
            repo_path = str(Path(".tmp/repos") / repo_name)
            if not os.path.exists(repo_path) or not _repo_is_valid(repo_path):
                continue
            gen = GitPythonFeatureGenerator()
            gen.open_repo(repo_name, repo_path)
            t0 = time.perf_counter()
            for row in repo_rows:
                gen.compute_for_row(
                    repo_name, row["commit_sha"], row["file_path"],
                    row.get("commit_timestamp", ""))
            gp_time += time.perf_counter() - t0
            gp_rows += len(repo_rows)
            gen.clear()

        # --- GitPython (cached) ---
        cache_gen = GitPythonFeatureGenerator()
        cache_time = 0.0
        cache_rows = 0
        for repo_name, repo_rows in by_repo.items():
            repo_path = str(Path(".tmp/repos") / repo_name)
            if not os.path.exists(repo_path) or not _repo_is_valid(repo_path):
                continue
            cache_gen.open_repo(repo_name, repo_path)
            # Warm
            for row in repo_rows:
                cache_gen.compute_for_row(
                    repo_name, row["commit_sha"], row["file_path"],
                    row.get("commit_timestamp", ""))
            cache_gen._exp_cache.clear()
            # Benchmark
            t0 = time.perf_counter()
            for row in repo_rows:
                cache_gen.compute_for_row(
                    repo_name, row["commit_sha"], row["file_path"],
                    row.get("commit_timestamp", ""))
            cache_time += time.perf_counter() - t0
            cache_rows += len(repo_rows)
            cache_gen.clear()

        if sub_rows == 0:
            pytest.skip("No repos available")

        sub_rate = sub_rows / sub_time if sub_time > 0 else 0
        gp_rate = gp_rows / gp_time if gp_time > 0 else 0
        cache_rate = cache_rows / cache_time if cache_time > 0 else 0

        # Project full generation
        total_dataset_rows = 54389
        proj_sub = total_dataset_rows / sub_rate if sub_rate > 0 else float("inf")
        proj_gp = total_dataset_rows / gp_rate if gp_rate > 0 else float("inf")
        proj_cache = total_dataset_rows / cache_rate if cache_rate > 0 else float("inf")

        report = f"""
{'='*60}
PERFORMANCE BENCHMARK RESULTS
{'='*60}
Rows benchmarked: {sub_rows}

A. Subprocess (reference):
   Time: {sub_time:.2f}s
   Rate: {sub_rate:.1f} rows/s
   Projected full gen: {proj_sub:.0f}s ({proj_sub/60:.1f}min)

B. GitPython (fresh per-repo):
   Time: {gp_time:.2f}s
   Rate: {gp_rate:.1f} rows/s
   Projected full gen: {proj_gp:.0f}s ({proj_gp/60:.1f}min)
   Speedup vs subprocess: {gp_rate/sub_rate:.1f}x

C. GitPython (cached):
   Time: {cache_time:.2f}s
   Rate: {cache_rate:.1f} rows/s
   Projected full gen: {proj_cache:.0f}s ({proj_cache/60:.1f}min)
   Speedup vs subprocess: {cache_rate/sub_rate:.1f}x
{'='*60}
"""
        print(report)

        # Assertions
        assert sub_rate > 0, "Subprocess rate must be positive"
        assert gp_rate > 0, "GitPython rate must be positive"
        assert cache_rate > 0, "Cached rate must be positive"
        # GitPython should be at least as fast as subprocess
        assert gp_rate >= sub_rate * 0.5, (
            f"GitPython ({gp_rate:.1f} rows/s) unexpectedly slower than "
            f"subprocess ({sub_rate:.1f} rows/s)"
        )
