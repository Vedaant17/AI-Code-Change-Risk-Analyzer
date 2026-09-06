"""Quick smoke test: exercise the commit endpoint against a real public repo."""

from __future__ import annotations

import json

import httpx


def main() -> None:
    resp = httpx.post(
        "http://127.0.0.1:8000/analysis/commit",
        json={
            "repo_url": "https://github.com/pallets/markupsafe",
            "commit_sha": "297fc8e356e6836a62087949245d09a28e9f1b13",
        },
        timeout=120,
    )
    print("Status:", resp.status_code)
    data = resp.json()
    if resp.status_code == 200:
        print("SHA:", data["sha"][:12])
        print("Author:", data["author"])
        print("Message:", data["message"][:80])
        print("Files changed:", len(data["files"]))
        print("Stats:", json.dumps(data["stats"], indent=2))
        for f in data["files"][:10]:
            status = f["status"]
            added = f["lines_added"]
            deleted = f["lines_deleted"]
            path = f["path"]
            print(f"  {status:8s} +{added:3d} -{deleted:3d}  {path}")
    else:
        print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
