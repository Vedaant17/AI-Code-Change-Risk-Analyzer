"""Phase 4.7: Label & Attribution Distribution Audit.

Reads the frozen combined-v3 JSONL files and performs a comprehensive
analysis of the label and attribution distribution to determine
suitability for file-level risk ranking.

All operations are READ-ONLY on frozen data.

Key analyses:
1. Positive/negative/ambiguous commit composition
2. All-positive vs mixed positive commits
3. Positive-file ratios
4. Attribution-source distribution
5. Attribution-source x commit-composition distribution
6. File-level vs commit-level attribution behavior
7. Repository-level positive/mixed distribution
8. Commit-size relationship
9. Original split vs Phase 4.6 repo split distribution
10. Suitability of the frozen labels for the intended file-level ranking task
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path("backend/data/datasets/combined-v3")
OUTPUT_DIR = Path("backend/data/models/v0.1.0-combined-v3-phase4.7")

VALID_LABEL_SOURCES = frozenset({
    "explicit_sha_reference",
    "revert",
    "line_overlap",
    "none",
    "ambiguous",
})

COMMIT_LEVEL_SOURCES = frozenset({
    "explicit_sha_reference",
    "revert",
})

FILE_LEVEL_SOURCES = frozenset({
    "line_overlap",
})


def _read_all_rows(data_dir: Path) -> list[dict]:
    """Read all rows from the frozen JSONL files.

    Returns raw dicts (not DatasetRow objects) to avoid unnecessary
    dependency on Pydantic validation during audit.
    """
    all_rows: list[dict] = []
    for split_name in ("train", "validation", "test"):
        path = data_dir / f"{split_name}.jsonl"
        if not path.exists():
            logger.warning("File not found: %s", path)
            continue
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON at line {line_no} in {path}: {e}"
                    ) from e
                all_rows.append(obj)
    logger.info("Loaded %d total rows from %s", len(all_rows), data_dir)
    return all_rows


def _group_by_commit(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Group rows by (repo_name, commit_sha)."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["repo_name"], row["commit_sha"])
        groups[key].append(row)
    return dict(groups)


def classify_commit(file_rows: list[dict]) -> dict:
    """Classify a commit's label composition.

    Returns dict with:
        total_files: int
        positive_files: int
        negative_files: int
        ambiguous_files: int
        classification: "all_positive" | "mixed" | "all_negative" | "all_ambiguous"
        positive_ratio: float
        attribution_sources: list[str] (sorted unique sources for positive files)
    """
    total = len(file_rows)
    pos = sum(1 for r in file_rows if r["defect_label"] == 1)
    neg = sum(1 for r in file_rows if r["defect_label"] == 0)
    amb = sum(1 for r in file_rows if r["defect_label"] == -1)

    if pos > 0 and neg == 0:
        classification = "all_positive"
    elif pos > 0 and neg > 0:
        classification = "mixed"
    elif pos == 0 and neg > 0:
        classification = "all_negative"
    else:
        classification = "all_ambiguous"

    ratio = pos / total if total > 0 else 0.0

    sources = sorted({
        r.get("label_source", "none")
        for r in file_rows
        if r["defect_label"] == 1
    })

    return {
        "total_files": total,
        "positive_files": pos,
        "negative_files": neg,
        "ambiguous_files": amb,
        "classification": classification,
        "positive_ratio": ratio,
        "attribution_sources": sources,
    }


def analyze_positive_commit_composition(all_rows: list[dict]) -> dict:
    """Analyze positive commit composition across the dataset."""
    commit_groups = _group_by_commit(all_rows)

    stats = {
        "total_commits": len(commit_groups),
        "positive_commits": 0,
        "negative_commits": 0,
        "ambiguous_commits": 0,
        "all_positive_commits": 0,
        "mixed_commits": 0,
        "all_positive_pct": 0.0,
        "mixed_pct": 0.0,
        "total_rows": len(all_rows),
        "positive_rows": sum(1 for r in all_rows if r["defect_label"] == 1),
        "negative_rows": sum(1 for r in all_rows if r["defect_label"] == 0),
        "ambiguous_rows": sum(1 for r in all_rows if r["defect_label"] == -1),
    }

    tf_list: list[int] = []
    pf_list: list[int] = []
    pr_list: list[float] = []

    for _key, rows in commit_groups.items():
        info = classify_commit(rows)
        cls = info["classification"]
        if cls == "all_positive":
            stats["positive_commits"] += 1
            stats["all_positive_commits"] += 1
        elif cls == "mixed":
            stats["positive_commits"] += 1
            stats["mixed_commits"] += 1
        elif cls == "all_negative":
            stats["negative_commits"] += 1
        else:
            stats["ambiguous_commits"] += 1

        if info["positive_files"] > 0:
            tf_list.append(info["total_files"])
            pf_list.append(info["positive_files"])
            pr_list.append(info["positive_ratio"])

    n_pos = stats["positive_commits"]
    stats["all_positive_pct"] = (
        stats["all_positive_commits"] / n_pos * 100 if n_pos else 0.0
    )
    stats["mixed_pct"] = (
        stats["mixed_commits"] / n_pos * 100 if n_pos else 0.0
    )

    def _dist(vals: list) -> dict:
        if not vals:
            return {
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0,
            }
        a = np.array(vals, dtype=np.float64)
        return {
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "min": float(a.min()),
            "max": float(a.max()),
            "std": float(a.std()),
        }

    return {
        "overall": stats,
        "positive_commit_file_distribution": {
            "total_files": _dist(tf_list),
            "positive_files": _dist(pf_list),
            "positive_ratio": _dist(pr_list),
        },
    }


def analyze_by_split(all_rows: list[dict]) -> dict:
    """Analyze positive commit composition per split."""
    split_rows: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        split_rows[row["split"]].append(row)

    result: dict[str, dict] = {}
    for split_name, rows in split_rows.items():
        commit_groups = _group_by_commit(rows)
        pos = 0
        all_pos = 0
        mixed = 0
        for _key, cr in commit_groups.items():
            info = classify_commit(cr)
            if info["classification"] == "all_positive":
                pos += 1
                all_pos += 1
            elif info["classification"] == "mixed":
                pos += 1
                mixed += 1

        result[split_name] = {
            "total_rows": len(rows),
            "positive_rows": sum(1 for r in rows if r["defect_label"] == 1),
            "total_commits": len(commit_groups),
            "positive_commits": pos,
            "all_positive_commits": all_pos,
            "mixed_commits": mixed,
            "all_positive_pct": all_pos / pos * 100 if pos else 0.0,
            "mixed_pct": mixed / pos * 100 if pos else 0.0,
        }

    return result


def analyze_by_repo(all_rows: list[dict]) -> dict:
    """Analyze per-repository label distribution."""
    repo_rows: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        repo_rows[row["repo_name"]].append(row)

    repos: list[dict] = []
    for repo_name in sorted(repo_rows):
        rows = repo_rows[repo_name]
        commit_groups = _group_by_commit(rows)
        pos_commits = 0
        all_pos = 0
        mixed = 0
        pos_file_rows = 0
        total_file_rows = len(rows)
        all_sources: set[str] = set()

        for _key, cr in commit_groups.items():
            info = classify_commit(cr)
            if info["classification"] == "all_positive":
                pos_commits += 1
                all_pos += 1
            elif info["classification"] == "mixed":
                pos_commits += 1
                mixed += 1
            pos_file_rows += info["positive_files"]
            all_sources.update(info["attribution_sources"])

        repos.append({
            "repo_name": repo_name,
            "positive_commits": pos_commits,
            "all_positive_commits": all_pos,
            "mixed_commits": mixed,
            "positive_file_rows": pos_file_rows,
            "total_file_rows": total_file_rows,
            "positive_file_ratio": (
                pos_file_rows / total_file_rows if total_file_rows else 0.0
            ),
            "primary_attribution_sources": sorted(all_sources - {"none"}),
        })

    repos.sort(key=lambda r: -r["positive_commits"])
    return {"repos": repos}


def analyze_attribution_sources(all_rows: list[dict]) -> dict:
    """Analyze attribution source distribution."""
    commit_groups = _group_by_commit(all_rows)

    by_source: dict[str, dict] = {}
    for source in VALID_LABEL_SOURCES:
        by_source[source] = {
            "positive_commits": 0,
            "positive_file_rows": 0,
            "all_positive_commits": 0,
            "mixed_commits": 0,
        }

    granularity = {
        "commit_level_commits": 0,
        "file_level_commits": 0,
    }

    for _key, rows in commit_groups.items():
        info = classify_commit(rows)
        if info["positive_files"] == 0:
            continue

        sources = set(info["attribution_sources"])
        has_commit_level = bool(sources & COMMIT_LEVEL_SOURCES)
        has_file_level = bool(sources & FILE_LEVEL_SOURCES)

        if has_commit_level:
            granularity["commit_level_commits"] += 1
        if has_file_level:
            granularity["file_level_commits"] += 1

        for source in sources:
            if source in by_source:
                by_source[source]["positive_commits"] += 1
                if info["classification"] == "all_positive":
                    by_source[source]["all_positive_commits"] += 1
                elif info["classification"] == "mixed":
                    by_source[source]["mixed_commits"] += 1

        pos_file_rows = info["positive_files"]
        for source in sources:
            if source in by_source:
                by_source[source]["positive_file_rows"] += pos_file_rows

    total_pos_commits = sum(
        1 for rows in commit_groups.values()
        if any(r["defect_label"] == 1 for r in rows)
    )
    pct_commit = (
        granularity["commit_level_commits"] / total_pos_commits * 100
        if total_pos_commits else 0.0
    )

    return {
        "by_source": by_source,
        "labeling_granularity_summary": {
            "commit_level_commits": granularity["commit_level_commits"],
            "file_level_commits": granularity["file_level_commits"],
            "pct_commit_level": pct_commit,
            "total_positive_commits": total_pos_commits,
        },
    }


def analyze_commit_size(all_rows: list[dict]) -> dict:
    """Analyze commit-size relationship with all-positive/mixed labeling."""
    commit_groups = _group_by_commit(all_rows)

    buckets: dict[str, list[dict]] = {
        "1": [],
        "2-3": [],
        "4-5": [],
        "6-10": [],
        "11-20": [],
        ">20": [],
    }

    def _bucket(n: int) -> str:
        if n == 1:
            return "1"
        if n <= 3:
            return "2-3"
        if n <= 5:
            return "4-5"
        if n <= 10:
            return "6-10"
        if n <= 20:
            return "11-20"
        return ">20"

    for _key, rows in commit_groups.items():
        info = classify_commit(rows)
        if info["positive_files"] == 0:
            continue
        buckets[_bucket(info["total_files"])].append(info)

    result: dict[str, dict] = {}
    for bname, infos in buckets.items():
        if not infos:
            result[bname] = {
                "count": 0,
                "all_positive": 0,
                "mixed": 0,
                "mean_ratio": 0.0,
            }
            continue
        ratios = [i["positive_ratio"] for i in infos]
        result[bname] = {
            "count": len(infos),
            "all_positive": sum(
                1 for i in infos if i["classification"] == "all_positive"
            ),
            "mixed": sum(1 for i in infos if i["classification"] == "mixed"),
            "mean_ratio": float(np.mean(ratios)),
        }

    return {"buckets": result}


def assess_suitability(
    composition: dict,
    attribution: dict,
) -> dict:
    """Assess suitability for file-level risk ranking."""
    overall = composition["overall"]
    gran = attribution["labeling_granularity_summary"]

    mixed_exists = overall["mixed_commits"] > 0
    pct_commit = gran["pct_commit_level"]
    all_pos_pct = overall["all_positive_pct"]

    if not mixed_exists and pct_commit == 100.0:
        suitability = "not_suitable"
        reasoning = (
            f"All {overall['positive_commits']} positive commits are "
            f"all-positive ({all_pos_pct:.1f}%). "
            f"{gran['commit_level_commits']}/{gran['total_positive_commits']} "
            f"positive commits use commit-level attribution (explicit SHA "
            f"reference or revert). {gran['file_level_commits']} commits use "
            f"file-level attribution (line overlap). The labeling methodology "
            f"cannot distinguish implicated files from unrelated files within "
            f"the same commit. File-level risk ranking requires mixed commits "
            f"where some files are positive and others are negative."
        )
    elif mixed_exists and pct_commit > 80.0:
        suitability = "conditionally_suitable"
        reasoning = (
            f"Mixed commits exist ({overall['mixed_commits']}, "
            f"{overall['mixed_pct']:.1f}%) but most positive commits "
            f"({pct_commit:.1f}%) use commit-level attribution."
        )
    elif mixed_exists:
        suitability = "conditionally_suitable"
        reasoning = (
            f"Mixed commits exist ({overall['mixed_commits']}, "
            f"{overall['mixed_pct']:.1f}%). File-level attribution provides "
            f"some supervision for ranking."
        )
    else:
        suitability = "not_suitable"
        reasoning = "No mixed commits found in the dataset."

    q_a = {
        "answer": "The model can distinguish positive from negative file rows globally.",
        "evidence": (
            f"Dataset contains {overall['positive_rows']} positive and "
            f"{overall['negative_rows']} negative file rows. Global "
            f"classification metrics measure this ability."
        ),
    }

    q_b_answer = (
        "The dataset provides NO supervision for ranking files within a commit."
        if not mixed_exists else
        "The dataset provides limited supervision for ranking files within a commit."
    )
    q_b_evidence = (
        f"{'Zero' if not mixed_exists else str(overall['mixed_commits'])} "
        f"mixed commits exist. "
        f"{'All' if pct_commit == 100.0 else f'{pct_commit:.0f}% of'} "
        f"positive commits label every file as positive, providing no signal "
        f"to distinguish risky from safe files within a commit."
    )
    root_cause = (
        "The labeling methodology uses commit-level attribution for all "
        "positive labels in the dataset. No file-level attribution "
        "(line overlap) was successfully applied."
        if not mixed_exists else
        "Mixed commits exist but are rare."
    )

    directions: list[dict] = []
    if not mixed_exists:
        directions.append({
            "direction": "Construct mixed-commit evaluation subset",
            "evidence_needed": (
                "Commits where bug-fixes touch a subset of changed files, "
                "with reliable file-level attribution."
            ),
            "assumptions": (
                "That some bug-fix commits exist where only specific files "
                "are implicated and others are collateral."
            ),
            "changes_frozen_semantics": False,
            "requires_regeneration": True,
            "supports_ranking": True,
        })
        directions.append({
            "direction": "Improve file-level defect attribution",
            "evidence_needed": (
                "Implementation of line-overlap or other file-level "
                "attribution that can identify specific files implicated "
                "by a defect."
            ),
            "assumptions": (
                "That git diff hunk analysis can reliably identify which "
                "files in a bug-fix commit are directly related to the "
                "defect."
            ),
            "changes_frozen_semantics": True,
            "requires_regeneration": True,
            "supports_ranking": True,
        })
        directions.append({
            "direction": "Use post-hoc bug-fix evidence for labeling",
            "evidence_needed": (
                "Later commits that fix bugs introduced by earlier commits, "
                "where the fix touches specific files."
            ),
            "assumptions": (
                "That bug-introducing commits can be identified through "
                "follow-up bug-fix analysis."
            ),
            "changes_frozen_semantics": True,
            "requires_regeneration": True,
            "supports_ranking": True,
        })

    return {
        "suitability": suitability,
        "reasoning": reasoning,
        "question_a_classification": q_a,
        "question_b_ranking": {"answer": q_b_answer, "evidence": q_b_evidence},
        "root_cause": root_cause,
        "possible_directions": directions,
    }


def _json_default(obj: object) -> object:
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default, sort_keys=True)
        f.write("\n")


def run_audit(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """Run the complete Phase 4.7 label audit.

    Returns a dict with all analysis results for programmatic use.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Phase 4.7: Label & Attribution Distribution Audit")
    logger.info("Reading frozen dataset from %s", data_dir)

    all_rows = _read_all_rows(data_dir)

    logger.info("Analyzing positive commit composition...")
    composition = analyze_positive_commit_composition(all_rows)
    by_split = analyze_by_split(all_rows)
    _write_json(output_dir / "label_distribution.json", {
        "overall": composition["overall"],
        "positive_commit_file_distribution": (
            composition["positive_commit_file_distribution"]
        ),
        "by_split": by_split,
    })

    logger.info("Analyzing attribution sources...")
    attribution = analyze_attribution_sources(all_rows)
    _write_json(output_dir / "attribution_analysis.json", attribution)

    logger.info("Analyzing repository distribution...")
    repo_analysis = analyze_by_repo(all_rows)
    _write_json(output_dir / "repository_analysis.json", repo_analysis)

    logger.info("Analyzing commit-size relationship...")
    size_analysis = analyze_commit_size(all_rows)
    _write_json(output_dir / "commit_size_analysis.json", size_analysis)

    logger.info("Assessing evaluation suitability...")
    suitability = assess_suitability(composition, attribution)
    _write_json(output_dir / "evaluation_suitability.json", suitability)

    logger.info("Generating report...")
    _write_report(
        composition,
        attribution,
        repo_analysis,
        size_analysis,
        suitability,
        by_split,
        output_dir,
    )

    logger.info("Phase 4.7 audit complete. Artifacts: %s", output_dir)

    return {
        "composition": composition,
        "attribution": attribution,
        "repo_analysis": repo_analysis,
        "size_analysis": size_analysis,
        "suitability": suitability,
    }


def _write_report(
    composition: dict,
    attribution: dict,
    repo_analysis: dict,
    size_analysis: dict,
    suitability: dict,
    by_split: dict,
    output_dir: Path,
) -> None:
    """Write the Phase 4.7 audit report."""
    ov = composition["overall"]
    gran = attribution["labeling_granularity_summary"]
    dist = composition["positive_commit_file_distribution"]

    lines: list[str] = [
        "# Phase 4.7: Label & Attribution Distribution Audit",
        "",
        "## 1. Executive Finding",
        "",
    ]

    if ov["mixed_commits"] == 0:
        lines.extend([
            (f"The frozen combined-v3 dataset contains "
             f"**{ov['positive_commits']:,} positive commits** and "
             f"**{ov['mixed_commits']:,} mixed commits**."),
            "",
            (f"**{gran['pct_commit_level']:.0f}% of positive commits** use "
             f"commit-level attribution (explicit SHA reference or revert), "
             f"which labels **every changed file** as positive."),
            "",
            (f"**{gran['file_level_commits']} commits** use file-level "
             f"attribution (line overlap), which can distinguish implicated "
             f"files."),
            "",
            ("**The dataset provides no supervision for file-level risk "
             "ranking within commits.** The product objective requires mixed "
             "commits where some files are positive and others are negative."),
            "",
            "**Assessment: NOT SUITABLE** for training/evaluating file-level "
            "risk ranking.",
        ])
    else:
        lines.extend([
            (f"The dataset contains {ov['positive_commits']:,} positive "
             f"commits, of which {ov['mixed_commits']:,} "
             f"({ov['mixed_pct']:.1f}%) are mixed."),
        ])

    lines.extend([
        "",
        "---",
        "",
        "## 2. Positive Commit Composition",
        "",
        "### Overall",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total commits | {ov['total_commits']:,} |",
        f"| Positive commits | {ov['positive_commits']:,} |",
        f"| Negative commits | {ov['negative_commits']:,} |",
        f"| Ambiguous commits | {ov['ambiguous_commits']:,} |",
        (f"| All-positive commits | {ov['all_positive_commits']:,} "
         f"({ov['all_positive_pct']:.1f}%) |"),
        (f"| Mixed commits | {ov['mixed_commits']:,} "
         f"({ov['mixed_pct']:.1f}%) |"),
        f"| Total rows | {ov['total_rows']:,} |",
        f"| Positive rows | {ov['positive_rows']:,} |",
        f"| Negative rows | {ov['negative_rows']:,} |",
        f"| Ambiguous rows | {ov['ambiguous_rows']:,} |",
        "",
        "### Positive Commit File Distribution",
        "",
        "| Metric | Mean | Median | Min | Max |",
        "|--------|------|--------|-----|-----|",
        (f"| Total files | {dist['total_files']['mean']:.1f} "
         f"| {dist['total_files']['median']:.1f} "
         f"| {dist['total_files']['min']:.0f} "
         f"| {dist['total_files']['max']:.0f} |"),
        (f"| Positive files | {dist['positive_files']['mean']:.1f} "
         f"| {dist['positive_files']['median']:.1f} "
         f"| {dist['positive_files']['min']:.0f} "
         f"| {dist['positive_files']['max']:.0f} |"),
        (f"| Positive ratio | {dist['positive_ratio']['mean']:.3f} "
         f"| {dist['positive_ratio']['median']:.3f} "
         f"| {dist['positive_ratio']['min']:.3f} "
         f"| {dist['positive_ratio']['max']:.3f} |"),
        "",
        "### Per-Split Breakdown",
        "",
        ("| Split | Commits | Pos Commits | All-Pos | Mixed "
         "| All-Pos % | Mixed % |"),
        ("|-------|---------|-------------|---------|-------"
         "|-----------|---------|"),
    ])

    for sn in ("train", "validation", "test"):
        s = by_split.get(sn, {})
        lines.append(
            f"| {sn} | {s.get('total_commits', 0):,} | "
            f"{s.get('positive_commits', 0):,} | "
            f"{s.get('all_positive_commits', 0):,} | "
            f"{s.get('mixed_commits', 0):,} | "
            f"{s.get('all_positive_pct', 0):.1f}% | "
            f"{s.get('mixed_pct', 0):.1f}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Attribution-Source Analysis",
        "",
        ("| Source | Pos Commits | Pos File Rows | All-Pos "
         "| Mixed | Granularity |"),
        ("|--------|-------------|---------------|---------"
         "|-------|-------------|"),
    ])

    for src in ("explicit_sha_reference", "revert", "line_overlap",
                "none", "ambiguous"):
        s = attribution["by_source"].get(src, {})
        glbl = (
            "commit" if src in COMMIT_LEVEL_SOURCES
            else ("file" if src == "line_overlap" else "n/a")
        )
        lines.append(
            f"| {src} | {s.get('positive_commits', 0):,} | "
            f"{s.get('positive_file_rows', 0):,} | "
            f"{s.get('all_positive_commits', 0):,} | "
            f"{s.get('mixed_commits', 0):,} | {glbl} |"
        )

    lines.extend([
        "",
        (f"**Granularity summary:** "
         f"{gran['commit_level_commits']}/{gran['total_positive_commits']} "
         f"positive commits ({gran['pct_commit_level']:.0f}%) use "
         f"commit-level attribution."),
        "",
        "---",
        "",
        "## 4. File-Level Label Construction Analysis",
        "",
        "### A. Implementation Fact: What the Frozen Code Does",
        "",
        ("The labeling is implemented in `labeling.py` "
         "(`DefectLabeler.attribute_defects()`) and resolved in "
         "`builder.py` "
         "(`CombinedDatasetBuilder._build_single_repo()`)."),
        "",
        "**Commit-level attribution (high confidence):**",
        ("- `explicit_sha_reference`: A bug-fix commit message contains an "
         "explicit SHA reference to the candidate commit. The entire commit "
         "is labeled positive. Evidence: "
         "\"commit-level attribution propagated to all changed files.\""),
        ("- `revert`: The candidate commit was reverted. The entire commit "
         "is labeled positive. Evidence: "
         "\"commit-level attribution propagated to all changed files.\""),
        "",
        "**File-level attribution (medium confidence):**",
        ("- `line_overlap`: Line-level overlap between a bug-fix and the "
         "candidate commit. Only the specific matching file is labeled "
         "positive."),
        "",
        "**Builder resolution (builder.py lines 125-137):**",
        "```python",
        "if ff.file_path in commit_file_attrs:",
        "    # File has its own medium-confidence (line-overlap) attribution",
        "    fa = commit_file_attrs[ff.file_path]",
        "    file_defect_label = fa[1]  # = 1",
        "else:",
        "    # Fall back to commit-level attribution",
        "    file_defect_label = defect_label  # inherited from results[c.sha]",
        "```",
        "",
        "### B. Observed Dataset Fact: What the Frozen Data Contains",
        "",
        (f"- Total positive commits: {ov['positive_commits']:,}"),
        (f"- All-positive commits: {ov['all_positive_commits']:,} "
         f"({ov['all_positive_pct']:.1f}%)"),
        (f"- Mixed commits: {ov['mixed_commits']:,} "
         f"({ov['mixed_pct']:.1f}%)"),
        (f"- File-level attributed commits: "
         f"{gran['file_level_commits']:,}"),
        "",
        "### C. Inference",
        "",
    ])

    if ov["mixed_commits"] == 0 and gran["file_level_commits"] == 0:
        lines.extend([
            ("The observed data is fully consistent with the implementation: "
             "all positive labels come from commit-level attribution, and no "
             "file-level attribution was applied. The labeling methodology "
             "cannot distinguish implicated files from unrelated files "
             "within the same commit."),
            "",
            ("Files unrelated to the defect (refactors, formatting, tests, "
             "docs) that happen to be changed in the same commit receive "
             "`defect_label=1` with the same confidence as the actually "
             "implicated file."),
        ])
    elif ov["mixed_commits"] > 0:
        lines.extend([
            (f"Some mixed commits exist ({ov['mixed_commits']:,}, "
             f"{ov['mixed_pct']:.1f}%), indicating file-level attribution "
             f"is partially effective."),
        ])

    lines.extend([
        "",
        "---",
        "",
        "## 5. Repository-Level Distribution",
        "",
        ("| Repo | Pos Commits | All-Pos | Mixed | Pos Files "
         "| Total Files | Pos Ratio | Sources |"),
        ("|------|-------------|---------|-------|-----------"
         "|-------------|-----------|---------|"),
    ])

    for r in repo_analysis["repos"]:
        if r["positive_commits"] == 0:
            continue
        srcs = (
            ", ".join(r["primary_attribution_sources"])
            if r["primary_attribution_sources"] else "none"
        )
        lines.append(
            f"| {r['repo_name']} | {r['positive_commits']} | "
            f"{r['all_positive_commits']} | {r['mixed_commits']} | "
            f"{r['positive_file_rows']} | {r['total_file_rows']} | "
            f"{r['positive_file_ratio']:.3f} | {srcs} |"
        )

    mixed_repos = [
        r for r in repo_analysis["repos"] if r["mixed_commits"] > 0
    ]
    lines.extend([
        "",
        f"**Repos with mixed commits: {len(mixed_repos)}**",
        "",
        "---",
        "",
        "## 6. Commit-Size Analysis",
        "",
        "| Bucket | Pos Commits | All-Pos | Mixed | Mean Ratio |",
        "|--------|-------------|---------|-------|------------|",
    ])

    for bucket in ("1", "2-3", "4-5", "6-10", "11-20", ">20"):
        b = size_analysis["buckets"].get(bucket, {})
        lines.append(
            f"| {bucket} | {b.get('count', 0)} | "
            f"{b.get('all_positive', 0)} | {b.get('mixed', 0)} | "
            f"{b.get('mean_ratio', 0):.3f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 7. Why TEST Has Zero Mixed Commits",
        "",
    ])

    if ov["mixed_commits"] == 0:
        lines.extend([
            ("The zero-mixed-commits finding is **systemic**, not a "
             "repository-split artifact:"),
            "",
            ("1. **Root cause:** No file-level attribution (line overlap) "
             "was successfully applied in the frozen dataset."),
            ("2. **Mechanism:** All positive labels come from "
             "`explicit_sha_reference` and/or `revert`, both of which "
             "are commit-level and label every file as positive."),
            ("3. **Evidence:** The same all-positive pattern appears in "
             "TRAIN, VALIDATION, and TEST. The repository split is "
             "irrelevant."),
            ("4. **Quantification:** See per-split breakdown above and "
             "repository table above."),
        ])
    else:
        lines.extend([
            (f"Mixed commits exist ({ov['mixed_commits']:,}, "
             f"{ov['mixed_pct']:.1f}%). The zero-mixed finding in TEST "
             f"may be partly a repository-split effect."),
        ])

    lines.extend([
        "",
        "---",
        "",
        "## 8. Ranking-Supervision Assessment",
        "",
        ("**Question A (Classification):** "
         f"{suitability['question_a_classification']['answer']}"),
        "",
        ("**Question B (Ranking):** "
         f"{suitability['question_b_ranking']['answer']}"),
        "",
        f"**Root cause:** {suitability['root_cause']}",
        "",
        (f"**Suitability: "
         f"{suitability['suitability'].upper()}**"),
        "",
        f"**Reasoning:** {suitability['reasoning']}",
        "",
        "---",
        "",
        "## 9. Recommended Next Methodological Directions",
        "",
    ])

    for i, d in enumerate(
        suitability.get("possible_directions", []), 1
    ):
        lines.extend([
            f"### Direction {i}: {d['direction']}",
            "",
            f"- **Evidence needed:** {d['evidence_needed']}",
            f"- **Assumptions:** {d['assumptions']}",
            (f"- **Changes frozen semantics:** "
             f"{d['changes_frozen_semantics']}"),
            f"- **Requires regeneration:** {d['requires_regeneration']}",
            f"- **Supports ranking objective:** {d['supports_ranking']}",
            "",
        ])

    if not suitability.get("possible_directions"):
        lines.append("No additional directions identified from this audit.")

    lines.extend([
        "",
        "---",
        "",
        "## 10. Frozen Data Integrity",
        "",
        "- No frozen dataset files (JSONL) were modified.",
        "- No frozen labeling logic (`labeling.py`) was modified.",
        "- No frozen builder logic (`builder.py`) was modified.",
        "- No frozen schema definitions (`schemas.py`) were modified.",
        "- No Phase 4/4.5/4.6 model artifacts were modified.",
        "- No dataset regeneration was performed.",
        "- All analyses were performed by reading frozen JSONL files.",
        "",
    ])

    report_path = output_dir / "phase47_label_audit.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Report written to %s", report_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_audit()
