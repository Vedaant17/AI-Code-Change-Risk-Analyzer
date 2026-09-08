"""Generate the combined multi-repository dataset (Phase 3.7 v3).

Extends Phase 3.6 with additional repositories for dataset expansion.
Outputs to backend/data/datasets/combined-v3/.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))  # noqa: E402

from backend.app.dataset.combined_builder import CombinedDatasetBuilder  # noqa: E402
from backend.app.dataset.config import DatasetConfig  # noqa: E402
from backend.app.dataset.multi_repo_config import MultiRepoConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_BASE = project_root / "backend" / "data" / "datasets"


def main() -> None:
    """Generate the Phase 3.7 expanded multi-repo dataset."""
    logger.info("Phase 3.7: Combined multi-repository dataset generation (v3)")

    multi_config = MultiRepoConfig(
        dataset_version="v3-multi-phase3.7",
        combined_dir_name="combined-v3",
    )
    dataset_config = DatasetConfig()

    logger.info("Repositories (%d total):", len(multi_config.repositories))
    for repo in multi_config.repositories:
        logger.info("  %s @ %s (max %d commits)",
                     repo.repo_name, repo.repo_revision[:12], repo.max_commits)

    builder = CombinedDatasetBuilder(
        multi_config=multi_config,
        dataset_config=dataset_config,
    )

    manifests = builder.build_all(OUTPUT_BASE)

    # Print summary
    print("\n" + "=" * 70)
    print("PHASE 3.7 DATASET GENERATION COMPLETE (v3)")
    print("=" * 70)

    if "combined-v3" in manifests:
        m = manifests["combined-v3"]
        print("\nCombined Dataset (v3):")
        print(f"  Total commits:     {m.total_commits_included}")
        print(f"  Total file examples: {m.total_file_examples}")
        print(f"  Positive:          {m.positive_examples}")
        print(f"  Negative:          {m.negative_examples}")
        print(f"  Ambiguous:         {m.ambiguous_examples}")
        print(f"  Train:             {m.train_examples} files")
        print(f"  Validation:        {m.validation_examples} files")
        print(f"  Test:              {m.test_examples} files")

        print(f"\n  Explicit SHA attributions: {m.explicit_sha_attributions}")
        print(f"  Revert attributions:       {m.revert_attributions}")
        print(f"  Line overlap attributions: {m.line_overlap_attributions}")
        print(f"  Ambiguous attributions:    {m.ambiguous_attributions}")

    print("\nPer-repo summaries:")
    for name, m in manifests.items():
        if name == "combined-v3":
            continue
        print(f"\n  {name}:")
        print(f"    Commits: {m.total_commits_included}")
        print(f"    Files:   {m.total_file_examples}")
        print(f"    Pos:     {m.positive_examples}")
        print(f"    Neg:     {m.negative_examples}")
        print(f"    Amb:     {m.ambiguous_examples}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
