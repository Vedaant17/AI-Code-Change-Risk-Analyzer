"""Investigate positive example distribution across splits."""

from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path("backend/data/datasets/flask")

for split in ["train", "validation", "test"]:
    lines = (DATA_DIR / f"{split}.jsonl").read_text(encoding="utf-8").strip().split("\n")
    positives = [json.loads(line) for line in lines if json.loads(line)["defect_label"] == 1]
    if positives:
        print(f"{split}: {len(positives)} positive")
        for p in positives:
            sha = p["commit_sha"][:12]
            fp = p["file_path"]
            src = p["label_source"]
            ev = p["label_evidence"][:100]
            print(f"  SHA: {sha}... file: {fp}")
            print(f"  source: {src}  evidence: {ev}")
    else:
        print(f"{split}: 0 positive")
