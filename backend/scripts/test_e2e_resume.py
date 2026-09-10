"""End-to-end interruption/resume test for Phase 4.5a generation.

Creates a small temporary repository, runs the generator with a simulated
interruption, resumes, and compares against a clean uninterrupted run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.scripts.generate_exp_4_5a import (
    GitPythonFeatureGenerator,
    _append_jsonl,
    _clone_repo,
    _read_jsonl,
    _read_jsonl_lenient,
)

PYTHON = sys.executable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _run(
    cmd: list[str], cwd: str | None = None, env: dict | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8",
        errors="replace", cwd=cwd, timeout=120, env=env,
    )


def _git(repo: str, args: list[str]) -> str:
    return _run(["git"] + args, cwd=repo).stdout.strip()


def _make_repo(path: str) -> list[str]:
    """Create a temp repo with 8 commits touching 3 files."""
    _git(path, ["init", "-q"])
    _git(path, ["config", "user.email", "test@test.com"])
    _git(path, ["config", "user.name", "Test"])

    ts = 1700000000
    commits = []

    def _commit(filename: str, content: str, msg: str) -> str:
        nonlocal ts
        fp = os.path.join(path, filename)
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, "w") as f:
            f.write(content)
        _git(path, ["add", filename])
        env = os.environ.copy()
        env["GIT_AUTHOR_DATE"] = f"@{ts} +0000"
        env["GIT_COMMITTER_DATE"] = f"@{ts} +0000"
        _run(["git", "-C", path, "commit", "-q", "-m", msg], env=env)
        sha = _git(path, ["rev-parse", "HEAD"])
        ts += 86400
        commits.append(sha)
        return sha

    _commit("app.py", "import os\n\ndef main():\n    pass\n", "initial commit")
    _commit("app.py", "import os\nimport sys\n\ndef main():\n    pass\n", "add sys import")
    _commit("utils.py", "def helper():\n    return 42\n", "add utils")
    _commit("app.py", "import os\nimport sys\n\ndef main():\n    print('hello')\n", "add print")
    _commit("utils.py", "def helper():\n    return 42\n\ndef other():\n    pass\n",
            "add other func")
    _commit("tests.py", "def test_main():\n    assert True\n", "add tests")
    _commit("app.py", "import os\nimport sys\n\ndef main():\n    print('done')\n", "fix print msg")
    _commit("utils.py", "def helper():\n    return 99\n\ndef other():\n    pass\n",
            "fix return value")

    return commits


def _make_dataset(repo_name: str, repo_url: str, commits: list[str],
                  files: list[str], out_dir: Path) -> None:
    """Create mini combined-v3 dataset."""
    rows_by_split: dict[str, list[dict]] = defaultdict(list)
    for i, sha in enumerate(commits):
        split = ["train", "validation", "test"][i % 3]
        for fp in files:
            rows_by_split[split].append({
                "repo_name": repo_name,
                "repo_url": repo_url,
                "commit_sha": sha,
                "file_path": fp,
                "commit_timestamp": f"2024-{1 + i:02d}-01T00:00:00Z",
                "label": 0,
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        p = out_dir / f"{split}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


def _run_generator(output_base: Path, source_dir: Path) -> None:
    """Run generate_exp_4_5a as subprocess, patching source_dir."""
    # Write a small wrapper script that patches the source dir
    script_content = f"""\
import sys, os
sys.path.insert(0, {str(PROJECT_ROOT)!r})
os.chdir({str(PROJECT_ROOT)!r})

from pathlib import Path
from backend.scripts import generate_exp_4_5a as gen_mod

# Patch source_dir in generate function
_original_generate = gen_mod.generate

def _patched_generate(output_base, repo_filter=None):
    import types
    # Monkeypatch the source_dir constant inside the function
    source_dir = Path({str(source_dir)!r})
    output_base = Path({str(output_base)!r})

    from collections import defaultdict

    gen = gen_mod.GitPythonFeatureGenerator()
    checkpoint_dir = output_base / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress_path = checkpoint_dir / "progress.json"

    split_data = {{}}
    for split in ["train", "validation", "test"]:
        split_data[split] = gen_mod._read_jsonl(source_dir / f"{{split}}.jsonl")

    repo_commits = defaultdict(lambda: defaultdict(list))
    for split, rows in split_data.items():
        for row in rows:
            rn = row["repo_name"]
            if repo_filter and rn != repo_filter:
                continue
            repo_commits[rn][row["commit_sha"]].append(
                (row["file_path"], split, row.get("commit_timestamp", "")),
            )

    total_commits = sum(len(v) for v in repo_commits.values())
    print(f"Target: {{total_commits}} commits across {{len(repo_commits)}} repos")

    progress = gen_mod._load_progress(progress_path)
    completed_repos = {{k for k, v in progress.items() if v.get("done")}}
    if completed_repos:
        print(f"Resuming: {{len(completed_repos)}} repos already completed")

    processed = 0
    import time
    t0 = time.time()

    for repo_name in sorted(repo_commits.keys()):
        if repo_name in completed_repos:
            repo_progress = progress.get(repo_name, {{}})
            processed += repo_progress.get("rows", 0)
            print(f"[{{repo_name}}] Skipping (completed)")
            continue

        commits = repo_commits[repo_name]
        repo_url = gen_mod._find_repo_url(split_data, repo_name)
        if not repo_url:
            continue

        print(f"[{{repo_name}}] Cloning ({{len(commits)}} commits)...")
        repo_path = gen_mod._clone_repo(repo_url, repo_name)
        if not repo_path:
            continue

        gen.open_repo(repo_name, repo_path)
        repo_ckpt_path = checkpoint_dir / f"{{repo_name}}.jsonl"

        repo_done_keys = set()
        repo_row_count = 0
        if repo_ckpt_path.exists():
            existing = gen_mod._read_jsonl_lenient(repo_ckpt_path)
            repo_row_count = len(existing)
            for r in existing:
                repo_done_keys.add((r["commit_sha"], r["file_path"]))
            if existing:
                gen_mod._atomic_write_jsonl(repo_ckpt_path, existing)
            print(f"[{{repo_name}}] Resuming from checkpoint ({{repo_row_count}} rows)")

        new_rows = 0
        for commit_sha in sorted(commits.keys()):
            all_files_done = all(
                (commit_sha, fp) in repo_done_keys
                for fp, _, _ in commits[commit_sha]
            )
            if all_files_done:
                processed += 1
                continue

            for file_path, split, commit_timestamp in commits[commit_sha]:
                if (commit_sha, file_path) in repo_done_keys:
                    continue
                features = gen.compute_for_row(
                    repo_name, commit_sha, file_path, commit_timestamp,
                )
                row = {{
                    "commit_sha": commit_sha,
                    "file_path": file_path,
                    "experimental_feature_version": "exp-4.5a",
                    "experimental_features": features,
                }}
                gen_mod._append_jsonl(repo_ckpt_path, row)
                new_rows += 1

            processed += 1
            if processed % 100 == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total_commits - processed) / rate if rate > 0 else 0
                print(f"  {{processed}}/{{total_commits}} commits "
                      f"[{{rate:.1f}} commits/s, ETA {{eta:.0f}}s]")

        progress[repo_name] = {{"done": True, "rows": repo_row_count + new_rows}}
        gen_mod._save_progress(progress_path, progress)
        gen.clear()
        print(f"[{{repo_name}}] Done ({{repo_row_count + new_rows}} rows)")

    elapsed_total = time.time() - t0
    print(f"Generation complete: {{elapsed_total:.1f}}s total")

    gen_mod._merge_checkpoints(output_base, checkpoint_dir, split_data)

    total = sum(
        len(gen_mod._read_jsonl(output_base / "combined-v3" / f"{{split}}.jsonl"))
        for split in ["train", "validation", "test"]
    )
    metadata = {{
        "experimental_feature_version": "exp-4.5a",
        "source_dataset": "combined-v3",
        "source_dataset_version": "v1",
        "feature_count": 6,
        "feature_names": [
            "file_ast_functions_added", "file_ast_functions_deleted",
            "file_ast_try_except_changed", "file_commit_count",
            "file_historical_bug_fixes", "file_days_since_last_change",
        ],
        "total_rows": total,
        "split_row_counts": {{
            split: len(gen_mod._read_jsonl(output_base / "combined-v3" / f"{{split}}.jsonl"))
            for split in ["train", "validation", "test"]
        }},
    }}
    with open(output_base / "combined-v3" / "metadata.json", "w", encoding="utf-8") as f:
        import json
        json.dump(metadata, f, indent=2, ensure_ascii=True)

    print(f"Done. Total: {{total}} rows")

_patched_generate(Path({str(output_base)!r}))
"""
    import tempfile as _tf
    sp = Path(_tf.mkdtemp()) / "run_gen.py"
    sp.write_text(script_content, encoding="utf-8")
    try:
        result = _run([PYTHON, str(sp)])
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                print(f"  {line}")
        if result.returncode != 0:
            print(f"  STDERR: {result.stderr[-2000:] if result.stderr else '(empty)'}")
    finally:
        sp.unlink(missing_ok=True)
        sp.parent.rmdir()


def main() -> None:
    tmp = tempfile.mkdtemp(prefix="e2e_resume_")
    try:
        # ── Step 1: Create temp repo ──────────────────────────────────
        print("=" * 60)
        print("STEP 1: Create temporary test repository")
        print("=" * 60)
        repo_dir = os.path.join(tmp, "testrepo")
        os.makedirs(repo_dir)
        commits = _make_repo(repo_dir)
        files = ["app.py", "utils.py", "tests.py"]
        n_commits = len(commits)
        n_files = len(files)
        print(f"  {n_commits} commits, {n_files} files = {n_commits*n_files} expected rows")

        # Clean up any stale clone from previous runs
        stale_clone = PROJECT_ROOT / ".tmp" / "repos" / "testrepo"
        if stale_clone.exists():
            shutil.rmtree(str(stale_clone), ignore_errors=True)

        # ── Step 2: Create dataset ────────────────────────────────────
        source_dir = Path(tmp) / "source_data"
        _make_dataset("testrepo", repo_dir, commits, files, source_dir)
        print(f"  Dataset written to {source_dir}")

        # ── Step 3: Clean uninterrupted run ───────────────────────────
        print("\n" + "=" * 60)
        print("STEP 2: Clean uninterrupted run (baseline)")
        print("=" * 60)
        clean_out = Path(tmp) / "clean_output"
        _run_generator(clean_out, source_dir)

        # Verify clean run
        clean_ckpt = clean_out / "checkpoints" / "testrepo.jsonl"
        clean_rows = _read_jsonl(clean_ckpt)
        clean_progress = json.loads((clean_out / "checkpoints" / "progress.json").read_text())
        print(f"  Checkpoint rows: {len(clean_rows)}")
        print(f"  Progress: {json.dumps(clean_progress)}")

        clean_merged = {}
        for split in ["train", "validation", "test"]:
            p = clean_out / "combined-v3" / f"{split}.jsonl"
            clean_merged[split] = _read_jsonl(p) if p.exists() else []
        clean_total = sum(len(v) for v in clean_merged.values())
        print(f"  Merged total: {clean_total}")

        # ── Step 4: Interrupted run ───────────────────────────────────
        print("\n" + "=" * 60)
        print("STEP 3: Interrupted run (simulate crash after 3 commits)")
        print("=" * 60)

        # Clean clone so interrupted run uses the same repo state as clean run
        stale_clone = PROJECT_ROOT / ".tmp" / "repos" / "testrepo"
        if stale_clone.exists():
            shutil.rmtree(str(stale_clone), ignore_errors=True)

        interrupt_out = Path(tmp) / "interrupt_output"
        int_ckpt_dir = interrupt_out / "checkpoints"
        int_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset
        split_data = {}
        for split in ["train", "validation", "test"]:
            split_data[split] = _read_jsonl(source_dir / f"{split}.jsonl")

        repo_commits = defaultdict(lambda: defaultdict(list))
        for split, rows in split_data.items():
            for row in rows:
                rn = row["repo_name"]
                repo_commits[rn][row["commit_sha"]].append(
                    (row["file_path"], split, row.get("commit_timestamp", "")),
                )

        # Clone (same path as the generator would use)
        repo_path = _clone_repo(repo_dir, "testrepo")
        gen = GitPythonFeatureGenerator()
        gen.open_repo("testrepo", repo_path)
        repo_ckpt = int_ckpt_dir / "testrepo.jsonl"

        commits_sorted = sorted(repo_commits["testrepo"].keys())
        print(f"  Total commits: {len(commits_sorted)}")

        # Process commits 1-3, then write partial commit 4
        count = 0
        for sha in commits_sorted:
            for fp, split, ts in repo_commits["testrepo"][sha]:
                features = gen.compute_for_row("testrepo", sha, fp, ts)
                row = {
                    "commit_sha": sha, "file_path": fp,
                    "experimental_feature_version": "exp-4.5a",
                    "experimental_features": features,
                }
                _append_jsonl(repo_ckpt, row)
            count += 1
            print(f"  Processed commit {count}: {sha[:8]}")
            if count == 3:
                print("  >>> SIMULATING CRASH after commit 3 <<<")
                sha4 = commits_sorted[3]
                fp4_0 = repo_commits["testrepo"][sha4][0][0]
                fp4_1 = repo_commits["testrepo"][sha4][1][0]
                ts4 = repo_commits["testrepo"][sha4][0][2]
                features4 = gen.compute_for_row("testrepo", sha4, fp4_0, ts4)
                _append_jsonl(repo_ckpt, {
                    "commit_sha": sha4, "file_path": fp4_0,
                    "experimental_feature_version": "exp-4.5a",
                    "experimental_features": features4,
                })
                with open(repo_ckpt, "a", encoding="utf-8") as f:
                    partial = ('{"commit_sha": "' + sha4 + '", "file_path": "'
                               + fp4_1 + '", "experimental_feature_version":'
                               ' "exp-4.5a", "experimental_features": [0.')
                    f.write(partial)
                print(f"  Wrote truncated row for {sha4[:8]}:{fp4_1}")
                break

        gen.clear()

        # Verify checkpoint state
        print("\n  Checkpoint verification (post-crash):")
        ckpt_rows = _read_jsonl_lenient(repo_ckpt)
        print(f"    Recovered rows: {len(ckpt_rows)}")
        done_shas = sorted({r["commit_sha"] for r in ckpt_rows})
        print(f"    Completed commits: {[s[:8] for s in done_shas]}")
        ckpt_keys = sorted([(r["commit_sha"], r["file_path"]) for r in ckpt_rows])
        print(f"    Key tuples: {len(ckpt_keys)}")
        print(f"    Duplicates: {len(ckpt_keys) - len(set(ckpt_keys))}")
        progress_path = int_ckpt_dir / "progress.json"
        repo_done = False
        if progress_path.exists():
            prog = json.loads(progress_path.read_text())
            repo_done = prog.get("testrepo", {}).get("done", False)
        print(f"    Repo marked done: {repo_done}")

        # ── Step 5: Resume ────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("STEP 4: Resume generation")
        print("=" * 60)
        _run_generator(interrupt_out, source_dir)

        # Verify resume
        resume_rows = _read_jsonl(int_ckpt_dir / "testrepo.jsonl")
        resume_progress = json.loads((int_ckpt_dir / "progress.json").read_text())
        print(f"  Checkpoint rows: {len(resume_rows)}")
        print(f"  Progress: {json.dumps(resume_progress)}")

        resume_keys = sorted([(r["commit_sha"], r["file_path"]) for r in resume_rows])
        print(f"  Unique keys: {len(set(resume_keys))}")
        print(f"  Duplicates: {len(resume_keys) - len(set(resume_keys))}")

        resume_merged = {}
        for split in ["train", "validation", "test"]:
            p = interrupt_out / "combined-v3" / f"{split}.jsonl"
            resume_merged[split] = _read_jsonl(p) if p.exists() else []
        resume_total = sum(len(v) for v in resume_merged.values())
        print(f"  Merged total: {resume_total}")

        # ── Step 6: Compare ───────────────────────────────────────────
        print("\n" + "=" * 60)
        print("STEP 5: Compare clean vs resumed output")
        print("=" * 60)

        clean_keys = sorted([(r["commit_sha"], r["file_path"]) for r in clean_rows])
        print(f"  Clean checkpoint rows: {len(clean_keys)}")
        print(f"  Resume checkpoint rows: {len(resume_keys)}")
        print(f"  Checkpoint keys match: {clean_keys == resume_keys}")

        clean_feat = {
            (r["commit_sha"], r["file_path"]): r["experimental_features"]
            for r in clean_rows
        }
        resume_feat = {
            (r["commit_sha"], r["file_path"]): r["experimental_features"]
            for r in resume_rows
        }
        feat_mismatches = 0
        for k in clean_keys:
            if clean_feat.get(k) != resume_feat.get(k):
                feat_mismatches += 1
                print(f"    MISMATCH: {k[:8]}...{k[1][:30]}")
                print(f"      clean={clean_feat.get(k)}")
                print(f"      resume={resume_feat.get(k)}")
        print(f"  Feature mismatches: {feat_mismatches}")

        clean_mkeys = sorted([(r["commit_sha"], r["file_path"])
                              for rows in clean_merged.values() for r in rows])
        resume_mkeys = sorted([(r["commit_sha"], r["file_path"])
                               for rows in resume_merged.values() for r in rows])
        print(f"  Merged clean: {len(clean_mkeys)}, resume: {len(resume_mkeys)}")
        print(f"  Merged keys match: {clean_mkeys == resume_mkeys}")

        clean_mfeat = {}
        for rows in clean_merged.values():
            for r in rows:
                clean_mfeat[(r["commit_sha"], r["file_path"])] = r["experimental_features"]
        resume_mfeat = {}
        for rows in resume_merged.values():
            for r in rows:
                resume_mfeat[(r["commit_sha"], r["file_path"])] = r["experimental_features"]
        merge_mismatches = sum(1 for k in clean_mkeys if clean_mfeat.get(k) != resume_mfeat.get(k))
        print(f"  Merged feature mismatches: {merge_mismatches}")

        # Final verdict
        print("\n" + "=" * 60)
        all_ok = (
            len(resume_rows) == len(clean_rows)
            and clean_keys == resume_keys
            and feat_mismatches == 0
            and len(resume_keys) == len(set(resume_keys))
            and clean_mkeys == resume_mkeys
            and merge_mismatches == 0
            and resume_progress.get("testrepo", {}).get("done") is True
        )
        print(f"RESULT: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
        if not all_ok:
            print(f"  Rows: clean={len(clean_rows)}, resume={len(resume_rows)}")
            print(f"  Duplicates: {len(resume_keys) - len(set(resume_keys))}")
            print(f"  Checkpoint match: {clean_keys == resume_keys}")
            print(f"  Feature mismatches: {feat_mismatches}")
            print(f"  Merged match: {clean_mkeys == resume_mkeys}")
            print(f"  Repo done: {resume_progress.get('testrepo', {}).get('done')}")
        print("=" * 60)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
