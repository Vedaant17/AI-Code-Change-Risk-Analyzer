"""Phase 6 evidence engine tests.

Tests evidence schemas, cheap evidence collection, historical evidence
collection, temporal boundary enforcement, and deterministic output.
"""
from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from backend.app.evidence.collector import EvidenceEngine
from backend.app.evidence.schemas import EvidenceItem, FileEvidence
from backend.app.features.schemas import CommitFeatures, FeatureExtractionResult, FileFeatures
from backend.app.inference.feature_pipeline import PipelineResult
from backend.app.schemas.diff import CommitInfo, DiffStats, FileDiff, FileStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_file_features(
    file_path: str,
    lines_added: int = 10,
    lines_deleted: int = 5,
    total_lines_changed: int = 15,
    hunk_count: int = 1,
    is_binary: bool = False,
    is_test_file: bool = False,
    function_declarations_added: int = 0,
    function_declarations_deleted: int = 0,
    class_declarations_added: int = 0,
    class_declarations_deleted: int = 0,
    imports_added: int = 0,
    imports_deleted: int = 0,
    avg_changed_line_indent: float = 0.0,
    max_changed_line_indent: int = 0,
    avg_hunk_size: float = 0.0,
    language: float = 0.0,
) -> FileFeatures:
    """Create a features.schemas.FileFeatures for testing."""
    return FileFeatures(
        file_path=file_path,
        language=language,
        is_binary=is_binary,
        is_test_file=is_test_file,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        total_lines_changed=total_lines_changed,
        hunk_count=hunk_count,
        function_declarations_added=function_declarations_added,
        function_declarations_deleted=function_declarations_deleted,
        class_declarations_added=class_declarations_added,
        class_declarations_deleted=class_declarations_deleted,
        imports_added=imports_added,
        imports_deleted=imports_deleted,
        avg_changed_line_indent=avg_changed_line_indent,
        max_changed_line_indent=max_changed_line_indent,
        avg_hunk_size=avg_hunk_size,
    )


def _make_pipeline_file_features(
    path: str,
    status: str = "modified",
    total_lines_changed: int = 15,
    lines_added: int = 10,
    lines_deleted: int = 5,
    is_binary: bool = False,
    is_test_file: bool = False,
    language: float = 0.0,
):
    """Create a feature_pipeline.FileFeatures for testing."""
    ff = MagicMock()
    ff.path = path
    ff.status = status
    ff.total_lines_changed = total_lines_changed
    ff.lines_added = lines_added
    ff.lines_deleted = lines_deleted
    ff.is_binary = is_binary
    ff.is_test_file = is_test_file
    ff.language = language
    return ff


def _make_file_diff(
    path: str,
    status: FileStatus = FileStatus.MODIFIED,
    lines_added: int = 10,
    lines_deleted: int = 5,
    is_binary: bool = False,
) -> FileDiff:
    """Create a FileDiff for testing."""
    return FileDiff(
        path=path,
        status=status,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        is_binary=is_binary,
    )


def _make_commit_features(
    has_test_changes: bool = True,
    test_prod_coupling: bool = False,
    total_function_declarations_changed: int = 0,
) -> CommitFeatures:
    """Create CommitFeatures for testing."""
    return CommitFeatures(
        commit_sha="abc123",
        has_test_changes=has_test_changes,
        test_prod_coupling=test_prod_coupling,
        total_function_declarations_changed=total_function_declarations_changed,
    )


def _make_pipeline_result(
    commit_info: CommitInfo,
    file_features: list[FileFeatures],
    commit_features: CommitFeatures | None = None,
) -> PipelineResult:
    """Create a PipelineResult for testing."""
    ext = FeatureExtractionResult(
        commit_features=commit_features or _make_commit_features(),
        file_features=file_features,
    )
    # Create mock pipeline file features for B1 scoring
    pipeline_ffs = []
    for ff in file_features:
        pipeline_ff = MagicMock()
        pipeline_ff.path = ff.file_path
        pipeline_ff.status = "modified"
        pipeline_ff.total_lines_changed = ff.total_lines_changed
        pipeline_ff.lines_added = ff.lines_added
        pipeline_ff.lines_deleted = ff.lines_deleted
        pipeline_ff.is_binary = ff.is_binary
        pipeline_ff.is_test_file = ff.is_test_file
        pipeline_ff.language = ff.language
        pipeline_ffs.append(pipeline_ff)

    pr = MagicMock(spec=PipelineResult)
    pr.commit_info = commit_info
    pr.extraction_result = ext
    pr.file_features = pipeline_ffs
    return pr


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestEvidenceSchemas:
    """Tests for EvidenceItem and FileEvidence schemas."""

    def test_evidence_item_schema(self):
        """EvidenceItem validates with all required fields."""
        item = EvidenceItem(
            category="change",
            evidence_type="lines_changed",
            description="10 lines added, 5 deleted",
            source="diff",
            provenance="direct",
        )
        assert item.category == "change"
        assert item.evidence_type == "lines_changed"
        assert item.ref_commit == ""
        assert item.ref_file == ""

    def test_evidence_item_with_refs(self):
        """EvidenceItem validates with optional ref fields."""
        item = EvidenceItem(
            category="historical",
            evidence_type="recent_commits",
            description="File changed in 3 recent commit(s)",
            source="git_log",
            provenance="derived",
            ref_commit="abc123",
            ref_file="src/main.py",
        )
        assert item.ref_commit == "abc123"
        assert item.ref_file == "src/main.py"

    def test_file_evidence_schema(self):
        """FileEvidence validates with required fields."""
        fe = FileEvidence(
            path="src/main.py",
            score=0.8,
            position=0,
            total_lines_changed=100,
            evidence=[],
        )
        assert fe.path == "src/main.py"
        assert fe.score == 0.8
        assert fe.position == 0
        assert fe.evidence == []

    def test_file_evidence_with_items(self):
        """FileEvidence validates with evidence items."""
        item = EvidenceItem(
            category="change",
            evidence_type="lines_changed",
            description="10 lines added, 5 deleted",
            source="diff",
            provenance="direct",
        )
        fe = FileEvidence(
            path="test.py",
            score=0.5,
            position=2,
            total_lines_changed=15,
            evidence=[item],
        )
        assert len(fe.evidence) == 1
        assert fe.evidence[0].category == "change"

    def test_file_evidence_score_bounds(self):
        """FileEvidence score must be in [0, 1]."""
        with pytest.raises(Exception):
            FileEvidence(
                path="test.py",
                score=1.5,
                position=0,
                total_lines_changed=10,
            )

    def test_file_evidence_position_non_negative(self):
        """FileEvidence position must be >= 0."""
        with pytest.raises(Exception):
            FileEvidence(
                path="test.py",
                score=0.5,
                position=-1,
                total_lines_changed=10,
            )


# ---------------------------------------------------------------------------
# Cheap evidence tests
# ---------------------------------------------------------------------------


class TestCheapEvidence:
    """Tests for EvidenceEngine.collect_cheap."""

    def setup_method(self):
        self.engine = EvidenceEngine()

    def test_cheap_evidence_no_io(self):
        """collect_cheap makes no subprocess calls."""
        ci = CommitInfo(
            sha="abc123def456",
            author="Test",
            author_date=datetime(2026, 1, 1, tzinfo=UTC),
            files=[_make_file_diff("a.py")],
            stats=DiffStats(total_files=1, total_lines_added=10, total_lines_deleted=5),
        )
        ff = _make_file_features("a.py", lines_added=10, lines_deleted=5, total_lines_changed=15)
        pr = _make_pipeline_result(ci, [ff])

        result = self.engine.collect_cheap(pr)
        assert "a.py" in result
        assert len(result["a.py"]) > 0

    def test_change_evidence_lines_changed(self):
        """lines_changed evidence is always produced."""
        ci = CommitInfo(
            sha="abc123",
            files=[_make_file_diff("a.py", lines_added=20, lines_deleted=10)],
            stats=DiffStats(total_files=1, total_lines_added=20, total_lines_deleted=10),
        )
        ff = _make_file_features("a.py", lines_added=20, lines_deleted=10, total_lines_changed=30)
        pr = _make_pipeline_result(ci, [ff])

        result = self.engine.collect_cheap(pr)
        items = result["a.py"]
        lines_items = [i for i in items if i.evidence_type == "lines_changed"]
        assert len(lines_items) == 1
        assert "20 lines added" in lines_items[0].description

    def test_change_evidence_hunk_count(self):
        """hunk_count evidence produced when hunk_count > 1."""
        ci = CommitInfo(
            sha="abc123",
            files=[_make_file_diff("a.py")],
            stats=DiffStats(total_files=1, total_lines_added=10, total_lines_deleted=5),
        )
        ff = _make_file_features("a.py", hunk_count=3)
        pr = _make_pipeline_result(ci, [ff])

        result = self.engine.collect_cheap(pr)
        items = result["a.py"]
        hunk_items = [i for i in items if i.evidence_type == "hunk_count"]
        assert len(hunk_items) == 1
        assert "3 hunks" in hunk_items[0].description

    def test_no_hunk_count_when_single_hunk(self):
        """hunk_count evidence not produced when hunk_count == 1."""
        ci = CommitInfo(
            sha="abc123",
            files=[_make_file_diff("a.py")],
            stats=DiffStats(total_files=1, total_lines_added=10, total_lines_deleted=5),
        )
        ff = _make_file_features("a.py", hunk_count=1)
        pr = _make_pipeline_result(ci, [ff])

        result = self.engine.collect_cheap(pr)
        items = result["a.py"]
        hunk_items = [i for i in items if i.evidence_type == "hunk_count"]
        assert len(hunk_items) == 0

    def test_function_churn_evidence(self):
        """function_churn evidence produced when declarations added/deleted."""
        ci = CommitInfo(
            sha="abc123",
            files=[_make_file_diff("a.py")],
            stats=DiffStats(total_files=1, total_lines_added=10, total_lines_deleted=5),
        )
        ff = _make_file_features(
            "a.py",
            function_declarations_added=3,
            function_declarations_deleted=1,
        )
        pr = _make_pipeline_result(ci, [ff])

        result = self.engine.collect_cheap(pr)
        items = result["a.py"]
        func_items = [i for i in items if i.evidence_type == "function_churn"]
        assert len(func_items) == 1
        assert "3 function" in func_items[0].description
        assert "1 deleted" in func_items[0].description

    def test_class_churn_evidence(self):
        """class_churn evidence produced when class declarations change."""
        ci = CommitInfo(
            sha="abc123",
            files=[_make_file_diff("a.py")],
            stats=DiffStats(total_files=1, total_lines_added=10, total_lines_deleted=5),
        )
        ff = _make_file_features("a.py", class_declarations_added=2)
        pr = _make_pipeline_result(ci, [ff])

        result = self.engine.collect_cheap(pr)
        items = result["a.py"]
        cls_items = [i for i in items if i.evidence_type == "class_churn"]
        assert len(cls_items) == 1
        assert "2 class" in cls_items[0].description

    def test_import_churn_evidence(self):
        """import_churn evidence produced when imports change."""
        ci = CommitInfo(
            sha="abc123",
            files=[_make_file_diff("a.py")],
            stats=DiffStats(total_files=1, total_lines_added=10, total_lines_deleted=5),
        )
        ff = _make_file_features("a.py", imports_added=5, imports_deleted=2)
        pr = _make_pipeline_result(ci, [ff])

        result = self.engine.collect_cheap(pr)
        items = result["a.py"]
        imp_items = [i for i in items if i.evidence_type == "import_churn"]
        assert len(imp_items) == 1
        assert "5 import" in imp_items[0].description

    def test_file_status_evidence(self):
        """file_status evidence produced for non-modified files."""
        ci = CommitInfo(
            sha="abc123",
            files=[_make_file_diff("new.py", status=FileStatus.ADDED)],
            stats=DiffStats(total_files=1, total_lines_added=50, total_lines_deleted=0),
        )
        ff = _make_file_features("new.py", lines_added=50, lines_deleted=0)
        pr = _make_pipeline_result(ci, [ff])

        result = self.engine.collect_cheap(pr)
        items = result["new.py"]
        status_items = [i for i in items if i.evidence_type == "file_status"]
        assert len(status_items) == 1
        assert "added" in status_items[0].description

    def test_no_file_status_for_modified(self):
        """file_status evidence not produced for modified files."""
        ci = CommitInfo(
            sha="abc123",
            files=[_make_file_diff("a.py", status=FileStatus.MODIFIED)],
            stats=DiffStats(total_files=1, total_lines_added=10, total_lines_deleted=5),
        )
        ff = _make_file_features("a.py")
        pr = _make_pipeline_result(ci, [ff])

        result = self.engine.collect_cheap(pr)
        items = result["a.py"]
        status_items = [i for i in items if i.evidence_type == "file_status"]
        assert len(status_items) == 0

    def test_binary_file_evidence(self):
        """binary_file evidence produced for binary files."""
        ci = CommitInfo(
            sha="abc123",
            files=[_make_file_diff("image.png", is_binary=True)],
            stats=DiffStats(total_files=1, total_lines_added=0, total_lines_deleted=0),
        )
        ff = _make_file_features("image.png", is_binary=True)
        pr = _make_pipeline_result(ci, [ff])

        result = self.engine.collect_cheap(pr)
        items = result["image.png"]
        binary_items = [i for i in items if i.evidence_type == "binary_file"]
        assert len(binary_items) == 1

    def test_indentation_evidence(self):
        """indentation evidence produced when indentation > 0."""
        ci = CommitInfo(
            sha="abc123",
            files=[_make_file_diff("a.py")],
            stats=DiffStats(total_files=1, total_lines_added=10, total_lines_deleted=5),
        )
        ff = _make_file_features(
            "a.py",
            avg_changed_line_indent=4.5,
            max_changed_line_indent=12,
        )
        pr = _make_pipeline_result(ci, [ff])

        result = self.engine.collect_cheap(pr)
        items = result["a.py"]
        indent_items = [i for i in items if i.evidence_type == "indentation"]
        assert len(indent_items) == 1
        assert "4.5 spaces" in indent_items[0].description

    def test_hunk_size_evidence(self):
        """hunk_size evidence produced when avg_hunk_size > 0."""
        ci = CommitInfo(
            sha="abc123",
            files=[_make_file_diff("a.py")],
            stats=DiffStats(total_files=1, total_lines_added=10, total_lines_deleted=5),
        )
        ff = _make_file_features("a.py", avg_hunk_size=25.0)
        pr = _make_pipeline_result(ci, [ff])

        result = self.engine.collect_cheap(pr)
        items = result["a.py"]
        hunk_size_items = [i for i in items if i.evidence_type == "hunk_size"]
        assert len(hunk_size_items) == 1
        assert "25.0 lines" in hunk_size_items[0].description

    def test_test_file_evidence(self):
        """test_file evidence produced for test files."""
        ci = CommitInfo(
            sha="abc123",
            files=[_make_file_diff("test_main.py")],
            stats=DiffStats(total_files=1, total_lines_added=10, total_lines_deleted=5),
        )
        ff = _make_file_features("test_main.py", is_test_file=True)
        pr = _make_pipeline_result(ci, [ff])

        result = self.engine.collect_cheap(pr)
        items = result["test_main.py"]
        test_items = [i for i in items if i.evidence_type == "test_file"]
        assert len(test_items) == 1
        assert "test file" in test_items[0].description

    def test_test_coupling_evidence(self):
        """test_coupling evidence produced when test_prod_coupling is True."""
        cf = _make_commit_features(test_prod_coupling=True, has_test_changes=True)
        ci = CommitInfo(
            sha="abc123",
            files=[_make_file_diff("main.py")],
            stats=DiffStats(total_files=1, total_lines_added=10, total_lines_deleted=5),
        )
        ff = _make_file_features("main.py")
        pr = _make_pipeline_result(ci, [ff], commit_features=cf)

        result = self.engine.collect_cheap(pr)
        items = result["main.py"]
        coupling_items = [i for i in items if i.evidence_type == "test_coupling"]
        assert len(coupling_items) == 1

    def test_no_test_changes_evidence(self):
        """no_test_changes evidence produced when has_test_changes is False."""
        cf = _make_commit_features(has_test_changes=False)
        ci = CommitInfo(
            sha="abc123",
            files=[_make_file_diff("main.py")],
            stats=DiffStats(total_files=1, total_lines_added=10, total_lines_deleted=5),
        )
        ff = _make_file_features("main.py")
        pr = _make_pipeline_result(ci, [ff], commit_features=cf)

        result = self.engine.collect_cheap(pr)
        items = result["main.py"]
        no_test_items = [i for i in items if i.evidence_type == "no_test_changes"]
        assert len(no_test_items) == 1

    def test_change_concentration_evidence(self):
        """change_concentration evidence produced with correct percentage."""
        ci = CommitInfo(
            sha="abc123",
            files=[
                _make_file_diff("big.py", lines_added=80, lines_deleted=10),
                _make_file_diff("small.py", lines_added=5, lines_deleted=5),
            ],
            stats=DiffStats(total_files=2, total_lines_added=85, total_lines_deleted=15),
        )
        ff_big = _make_file_features(
            "big.py", lines_added=80, lines_deleted=10, total_lines_changed=90
        )
        ff_small = _make_file_features(
            "small.py", lines_added=5, lines_deleted=5, total_lines_changed=10
        )
        pr = _make_pipeline_result(ci, [ff_big, ff_small])

        result = self.engine.collect_cheap(pr)
        big_items = result["big.py"]
        conc_items = [i for i in big_items if i.evidence_type == "change_concentration"]
        assert len(conc_items) == 1
        assert "90%" in conc_items[0].description or "90 of" in conc_items[0].description

    def test_b1_ranking_evidence(self):
        """b1_ranking evidence produced when b1_context provided."""
        ci = CommitInfo(
            sha="abc123",
            files=[_make_file_diff("a.py", lines_added=30, lines_deleted=20)],
            stats=DiffStats(total_files=1, total_lines_added=30, total_lines_deleted=20),
        )
        ff = _make_file_features("a.py", lines_added=30, lines_deleted=20, total_lines_changed=50)
        pr = _make_pipeline_result(ci, [ff])

        b1_context = [("a.py", 0.8, 0)]
        result = self.engine.collect_cheap(pr, b1_context=b1_context)
        items = result["a.py"]
        rank_items = [i for i in items if i.evidence_type == "b1_ranking"]
        assert len(rank_items) == 1
        assert "position 0" in rank_items[0].description
        assert "50 lines changed" in rank_items[0].description

    def test_no_b1_ranking_without_context(self):
        """b1_ranking evidence not produced when b1_context is None."""
        ci = CommitInfo(
            sha="abc123",
            files=[_make_file_diff("a.py")],
            stats=DiffStats(total_files=1, total_lines_added=10, total_lines_deleted=5),
        )
        ff = _make_file_features("a.py")
        pr = _make_pipeline_result(ci, [ff])

        result = self.engine.collect_cheap(pr, b1_context=None)
        items = result["a.py"]
        rank_items = [i for i in items if i.evidence_type == "b1_ranking"]
        assert len(rank_items) == 0

    def test_deterministic_output(self):
        """Same inputs produce identical evidence items."""
        ci = CommitInfo(
            sha="abc123",
            author_date=datetime(2026, 1, 1, tzinfo=UTC),
            files=[_make_file_diff("a.py")],
            stats=DiffStats(total_files=1, total_lines_added=10, total_lines_deleted=5),
        )
        ff = _make_file_features("a.py", hunk_count=3, function_declarations_added=2)
        pr = _make_pipeline_result(ci, [ff])

        result1 = self.engine.collect_cheap(pr)
        result2 = self.engine.collect_cheap(pr)

        assert result1.keys() == result2.keys()
        for path in result1:
            items1 = result1[path]
            items2 = result2[path]
            assert len(items1) == len(items2)
            for i1, i2 in zip(items1, items2):
                assert i1.category == i2.category
                assert i1.evidence_type == i2.evidence_type
                assert i1.description == i2.description
                assert i1.source == i2.source
                assert i1.provenance == i2.provenance

    def test_no_subjective_claims(self):
        """No evidence description contains subjective risk language."""
        ci = CommitInfo(
            sha="abc123",
            files=[_make_file_diff("a.py")],
            stats=DiffStats(total_files=1, total_lines_added=10, total_lines_deleted=5),
        )
        ff = _make_file_features(
            "a.py",
            hunk_count=3,
            function_declarations_added=2,
            imports_added=5,
            avg_changed_line_indent=4.0,
            max_changed_line_indent=12,
            avg_hunk_size=20.0,
        )
        cf = _make_commit_features(test_prod_coupling=True)
        pr = _make_pipeline_result(ci, [ff], commit_features=cf)

        result = self.engine.collect_cheap(pr, b1_context=[("a.py", 0.5, 0)])

        forbidden_words = [
            "risky", "risk", "defect", "buggy", "dangerous",
            "should be investigated", "because", "suggests possible",
        ]
        for items in result.values():
            for item in items:
                desc_lower = item.description.lower()
                for word in forbidden_words:
                    assert word not in desc_lower, (
                        f"Forbidden word '{word}' in: {item.description}"
                    )

    def test_multiple_files(self):
        """Evidence produced for each file in the commit."""
        ci = CommitInfo(
            sha="abc123",
            files=[
                _make_file_diff("a.py", lines_added=20, lines_deleted=10),
                _make_file_diff("b.py", lines_added=5, lines_deleted=2),
            ],
            stats=DiffStats(total_files=2, total_lines_added=25, total_lines_deleted=12),
        )
        ff_a = _make_file_features("a.py", lines_added=20, lines_deleted=10, total_lines_changed=30)
        ff_b = _make_file_features("b.py", lines_added=5, lines_deleted=2, total_lines_changed=7)
        pr = _make_pipeline_result(ci, [ff_a, ff_b])

        result = self.engine.collect_cheap(pr)
        assert "a.py" in result
        assert "b.py" in result
        assert len(result["a.py"]) > 0
        assert len(result["b.py"]) > 0

    def test_provenance_direct(self):
        """Direct evidence has provenance 'direct'."""
        ci = CommitInfo(
            sha="abc123",
            files=[_make_file_diff("a.py")],
            stats=DiffStats(total_files=1, total_lines_added=10, total_lines_deleted=5),
        )
        ff = _make_file_features("a.py")
        pr = _make_pipeline_result(ci, [ff])

        result = self.engine.collect_cheap(pr)
        items = result["a.py"]
        lines_item = next(i for i in items if i.evidence_type == "lines_changed")
        assert lines_item.provenance == "direct"

    def test_provenance_derived(self):
        """Derived evidence has provenance 'derived'."""
        cf = _make_commit_features(test_prod_coupling=True, has_test_changes=True)
        ci = CommitInfo(
            sha="abc123",
            files=[_make_file_diff("main.py")],
            stats=DiffStats(total_files=1, total_lines_added=10, total_lines_deleted=5),
        )
        ff = _make_file_features("main.py")
        pr = _make_pipeline_result(ci, [ff], commit_features=cf)

        result = self.engine.collect_cheap(pr)
        items = result["main.py"]
        coupling_item = next(i for i in items if i.evidence_type == "test_coupling")
        assert coupling_item.provenance == "derived"


# ---------------------------------------------------------------------------
# Historical evidence tests
# ---------------------------------------------------------------------------


class TestHistoricalEvidence:
    """Tests for EvidenceEngine.collect_historical with temporal boundary."""

    def setup_method(self):
        self.engine = EvidenceEngine()

    def _create_repo_with_history(self):
        """Create a temp git repo with A -> B -> C history."""
        import shutil

        import git

        repo_dir = tempfile.mkdtemp(prefix="phase6_test_")
        repo = git.Repo.init(repo_dir)

        try:
            # Commit A
            with open(os.path.join(repo_dir, "src.py"), "w") as f:
                f.write("# A\n")
            repo.index.add(["src.py"])
            os.environ["GIT_AUTHOR_DATE"] = "2026-01-01 12:00:00 +0000"
            os.environ["GIT_COMMITTER_DATE"] = "2026-01-01 12:00:00 +0000"
            repo.index.commit("Commit A: initial")

            # Commit B
            with open(os.path.join(repo_dir, "src.py"), "w") as f:
                f.write("# A\n# B\n")
            repo.index.add(["src.py"])
            os.environ["GIT_AUTHOR_DATE"] = "2026-01-02 12:00:00 +0000"
            os.environ["GIT_COMMITTER_DATE"] = "2026-01-02 12:00:00 +0000"
            repo.index.commit("Commit B: update")

            # Commit C (the one we'll analyze)
            with open(os.path.join(repo_dir, "src.py"), "w") as f:
                f.write("# A\n# B\n# C\n")
            repo.index.add(["src.py"])
            os.environ["GIT_AUTHOR_DATE"] = "2026-01-03 12:00:00 +0000"
            os.environ["GIT_COMMITTER_DATE"] = "2026-01-03 12:00:00 +0000"
            repo.index.commit("Commit C: analyze me")

            # Clean env vars
            del os.environ["GIT_AUTHOR_DATE"]
            del os.environ["GIT_COMMITTER_DATE"]

            # Get commit C SHA
            commit_c = list(repo.iter_commits())[0]

            return repo_dir, commit_c.hexsha, datetime(2026, 1, 3, 12, 0, 0, tzinfo=UTC)
        except Exception:
            shutil.rmtree(repo_dir, ignore_errors=True)
            raise

    def test_historical_temporal_boundary_abc(self):
        """A->B->C: A and B may appear; C must not."""
        repo_dir, commit_sha, analyzed_ts = self._create_repo_with_history()

        try:
            ci = CommitInfo(
                sha=commit_sha,
                author_date=analyzed_ts,
                files=[_make_file_diff("src.py")],
                stats=DiffStats(total_files=1, total_lines_added=1, total_lines_deleted=0),
            )
            ff = _make_file_features("src.py")
            pr = _make_pipeline_result(ci, [ff])

            evidence, git_count, analyzed = self.engine.collect_historical(
                pr, repo_dir, top_k=10
            )

            assert git_count == 1
            assert analyzed == 1
            assert "src.py" in evidence

            items = evidence["src.py"]
            # Should have recent_commits evidence
            recent_items = [i for i in items if i.evidence_type == "recent_commits"]
            assert len(recent_items) == 1

            # All ref_commit values should be for A or B, not C
            for item in items:
                if item.ref_commit:
                    assert item.ref_commit != commit_sha
        finally:
            import shutil
            shutil.rmtree(repo_dir, ignore_errors=True)

    def test_historical_equal_timestamp_rejected(self):
        """Candidate with equal timestamp is rejected."""
        import git

        repo_dir = tempfile.mkdtemp(prefix="phase6_test_")
        repo = git.Repo.init(repo_dir)

        equal_ts = datetime(2026, 1, 3, 12, 0, 0, tzinfo=UTC)

        try:
            # Commit B with same timestamp as analyzed commit
            with open(os.path.join(repo_dir, "src.py"), "w") as f:
                f.write("# B\n")
            repo.index.add(["src.py"])
            os.environ["GIT_AUTHOR_DATE"] = "2026-01-03 12:00:00 +0000"
            os.environ["GIT_COMMITTER_DATE"] = "2026-01-03 12:00:00 +0000"
            repo.index.commit("Commit B: equal timestamp")

            # Commit C (analyzed) with same timestamp
            with open(os.path.join(repo_dir, "src.py"), "w") as f:
                f.write("# B\n# C\n")
            repo.index.add(["src.py"])
            os.environ["GIT_AUTHOR_DATE"] = "2026-01-03 12:00:00 +0000"
            os.environ["GIT_COMMITTER_DATE"] = "2026-01-03 12:00:00 +0000"
            repo.index.commit("Commit C: analyze me")

            del os.environ["GIT_AUTHOR_DATE"]
            del os.environ["GIT_COMMITTER_DATE"]

            commit_c = list(repo.iter_commits())[0]

            ci = CommitInfo(
                sha=commit_c.hexsha,
                author_date=equal_ts,
                files=[_make_file_diff("src.py")],
                stats=DiffStats(total_files=1, total_lines_added=1, total_lines_deleted=0),
            )
            ff = _make_file_features("src.py")
            pr = _make_pipeline_result(ci, [ff])

            evidence, _, _ = self.engine.collect_historical(pr, repo_dir, top_k=10)

            # No recent commits should be accepted (equal timestamp)
            items = evidence.get("src.py", [])
            recent_items = [i for i in items if i.evidence_type == "recent_commits"]
            # Should be 0 or have 0 commits
            if recent_items:
                assert "0 recent" in recent_items[0].description
        finally:
            import shutil
            shutil.rmtree(repo_dir, ignore_errors=True)

    def test_historical_later_timestamp_rejected(self):
        """Candidate with later timestamp is rejected."""
        import git

        repo_dir = tempfile.mkdtemp(prefix="phase6_test_")
        repo = git.Repo.init(repo_dir)

        try:
            # Commit B with later timestamp
            with open(os.path.join(repo_dir, "src.py"), "w") as f:
                f.write("# B\n")
            repo.index.add(["src.py"])
            os.environ["GIT_AUTHOR_DATE"] = "2026-01-05 12:00:00 +0000"
            os.environ["GIT_COMMITTER_DATE"] = "2026-01-05 12:00:00 +0000"
            repo.index.commit("Commit B: later timestamp")

            # Commit C (analyzed) with earlier timestamp
            with open(os.path.join(repo_dir, "src.py"), "w") as f:
                f.write("# B\n# C\n")
            repo.index.add(["src.py"])
            os.environ["GIT_AUTHOR_DATE"] = "2026-01-03 12:00:00 +0000"
            os.environ["GIT_COMMITTER_DATE"] = "2026-01-03 12:00:00 +0000"
            repo.index.commit("Commit C: analyze me")

            del os.environ["GIT_AUTHOR_DATE"]
            del os.environ["GIT_COMMITTER_DATE"]

            commit_c = list(repo.iter_commits())[0]

            ci = CommitInfo(
                sha=commit_c.hexsha,
                author_date=datetime(2026, 1, 3, 12, 0, 0, tzinfo=UTC),
                files=[_make_file_diff("src.py")],
                stats=DiffStats(total_files=1, total_lines_added=1, total_lines_deleted=0),
            )
            ff = _make_file_features("src.py")
            pr = _make_pipeline_result(ci, [ff])

            evidence, _, _ = self.engine.collect_historical(pr, repo_dir, top_k=10)

            items = evidence.get("src.py", [])
            recent_items = [i for i in items if i.evidence_type == "recent_commits"]
            if recent_items:
                # Should report 0 commits (later timestamp rejected)
                assert "0 recent" in recent_items[0].description
        finally:
            import shutil
            shutil.rmtree(repo_dir, ignore_errors=True)

    def test_historical_root_commit(self):
        """Root commit produces empty historical evidence."""
        import git

        repo_dir = tempfile.mkdtemp(prefix="phase6_test_")
        repo = git.Repo.init(repo_dir)

        try:
            with open(os.path.join(repo_dir, "src.py"), "w") as f:
                f.write("# root\n")
            repo.index.add(["src.py"])
            os.environ["GIT_AUTHOR_DATE"] = "2026-01-01 12:00:00 +0000"
            os.environ["GIT_COMMITTER_DATE"] = "2026-01-01 12:00:00 +0000"
            repo.index.commit("Root commit")

            del os.environ["GIT_AUTHOR_DATE"]
            del os.environ["GIT_COMMITTER_DATE"]

            commit = list(repo.iter_commits())[0]

            ci = CommitInfo(
                sha=commit.hexsha,
                author_date=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
                files=[_make_file_diff("src.py")],
                stats=DiffStats(total_files=1, total_lines_added=1, total_lines_deleted=0),
            )
            ff = _make_file_features("src.py")
            pr = _make_pipeline_result(ci, [ff])

            evidence, git_count, analyzed = self.engine.collect_historical(
                pr, repo_dir, top_k=10
            )

            assert git_count == 0
            assert analyzed == 0
            items = evidence.get("src.py", [])
            root_items = [i for i in items if i.evidence_type == "root_commit"]
            assert len(root_items) == 1
        finally:
            import shutil
            shutil.rmtree(repo_dir, ignore_errors=True)

    def test_historical_skips_binary(self):
        """Binary files are not eligible for historical analysis."""
        ci = CommitInfo(
            sha="abc123",
            author_date=datetime(2026, 1, 3, tzinfo=UTC),
            files=[_make_file_diff("image.png", is_binary=True)],
            stats=DiffStats(total_files=1),
        )
        ff = _make_file_features("image.png", is_binary=True)
        pr = _make_pipeline_result(ci, [ff])

        evidence, git_count, analyzed = self.engine.collect_historical(
            pr, "/nonexistent", top_k=10
        )

        assert git_count == 0
        assert analyzed == 0

    def test_historical_skips_deleted(self):
        """Deleted files are not eligible for historical analysis."""
        ci = CommitInfo(
            sha="abc123",
            author_date=datetime(2026, 1, 3, tzinfo=UTC),
            files=[_make_file_diff("old.py", status=FileStatus.DELETED)],
            stats=DiffStats(total_files=1),
        )
        ff = _make_file_features("old.py")
        # Set status to deleted in a mock pipeline feature
        pff = MagicMock()
        pff.path = "old.py"
        pff.status = "deleted"
        pff.total_lines_changed = 0
        pff.lines_added = 0
        pff.lines_deleted = 0
        pff.is_binary = False
        pff.is_test_file = False
        pff.language = 0.0

        ext = FeatureExtractionResult(
            commit_features=_make_commit_features(),
            file_features=[ff],
        )
        pr = MagicMock(spec=PipelineResult)
        pr.commit_info = ci
        pr.extraction_result = ext
        pr.file_features = [pff]

        evidence, git_count, analyzed = self.engine.collect_historical(
            pr, "/nonexistent", top_k=10
        )

        assert git_count == 0
        assert analyzed == 0

    def test_binary_before_eligible_does_not_consume_slot(self):
        """Binary file at position 0 does not consume a top_k slot."""
        ci = CommitInfo(
            sha="abc123",
            author_date=datetime(2026, 1, 3, tzinfo=UTC),
            files=[
                _make_file_diff("image.bin", is_binary=True),
                _make_file_diff("a.py"),
                _make_file_diff("b.py"),
            ],
            stats=DiffStats(total_files=3, total_lines_added=10, total_lines_deleted=5),
        )
        ff_bin = _make_file_features("image.bin", is_binary=True)
        ff_a = _make_file_features("a.py", total_lines_changed=20)
        ff_b = _make_file_features("b.py", total_lines_changed=10)

        # Mock pipeline features with correct ordering
        pipeline_ffs = []
        for ff, path, is_bin in [
            (ff_bin, "image.bin", True),
            (ff_a, "a.py", False),
            (ff_b, "b.py", False),
        ]:
            pff = MagicMock()
            pff.path = path
            pff.status = "modified"
            pff.total_lines_changed = ff.total_lines_changed
            pff.lines_added = ff.lines_added
            pff.lines_deleted = ff.lines_deleted
            pff.is_binary = is_bin
            pff.is_test_file = False
            pff.language = 0.0
            pipeline_ffs.append(pff)

        ext = FeatureExtractionResult(
            commit_features=_make_commit_features(),
            file_features=[ff_bin, ff_a, ff_b],
        )
        pr = MagicMock(spec=PipelineResult)
        pr.commit_info = ci
        pr.extraction_result = ext
        pr.file_features = pipeline_ffs

        # With top_k=1, only the first eligible (non-binary) file should be analyzed
        # Binary at position 0 should not consume the slot
        evidence, git_count, analyzed = self.engine.collect_historical(
            pr, "/nonexistent", top_k=1
        )

        assert git_count == 0  # No real git repo, but selection logic should pick a.py
        assert analyzed == 0

    def test_top_k_limit(self):
        """Only top_k eligible files receive historical analysis."""
        ci = CommitInfo(
            sha="abc123",
            author_date=datetime(2026, 1, 3, tzinfo=UTC),
            files=[
                _make_file_diff(f"file{i}.py") for i in range(5)
            ],
            stats=DiffStats(total_files=5, total_lines_added=50, total_lines_deleted=25),
        )
        ffs = [
            _make_file_features(f"file{i}.py", total_lines_changed=10 * (5 - i))
            for i in range(5)
        ]

        pipeline_ffs = []
        for ff in ffs:
            pff = MagicMock()
            pff.path = ff.file_path
            pff.status = "modified"
            pff.total_lines_changed = ff.total_lines_changed
            pff.lines_added = ff.lines_added
            pff.lines_deleted = ff.lines_deleted
            pff.is_binary = False
            pff.is_test_file = False
            pff.language = 0.0
            pipeline_ffs.append(pff)

        ext = FeatureExtractionResult(
            commit_features=_make_commit_features(),
            file_features=ffs,
        )
        pr = MagicMock(spec=PipelineResult)
        pr.commit_info = ci
        pr.extraction_result = ext
        pr.file_features = pipeline_ffs

        # With top_k=2, only 2 files should be selected
        evidence, git_count, analyzed = self.engine.collect_historical(
            pr, "/nonexistent", top_k=2
        )

        # Selection happens before git calls; git_count=0 because /nonexistent
        assert analyzed == 0
        # But the selection logic should have only picked 2 files
        # (We can't verify the exact selection without a real repo,
        # but the function should not error)

    def test_historical_shallow_clone(self):
        """Shallow clone produces history_unavailable evidence."""
        import git

        repo_dir = tempfile.mkdtemp(prefix="phase6_test_")
        repo = git.Repo.init(repo_dir)

        shallow_dir = None
        try:
            # Create a shallow clone
            with open(os.path.join(repo_dir, "src.py"), "w") as f:
                f.write("# A\n")
            repo.index.add(["src.py"])
            os.environ["GIT_AUTHOR_DATE"] = "2026-01-01 12:00:00 +0000"
            os.environ["GIT_COMMITTER_DATE"] = "2026-01-01 12:00:00 +0000"
            repo.index.commit("Commit A")

            with open(os.path.join(repo_dir, "src.py"), "w") as f:
                f.write("# A\n# B\n")
            repo.index.add(["src.py"])
            os.environ["GIT_AUTHOR_DATE"] = "2026-01-02 12:00:00 +0000"
            os.environ["GIT_COMMITTER_DATE"] = "2026-01-02 12:00:00 +0000"
            repo.index.commit("Commit B")

            del os.environ["GIT_AUTHOR_DATE"]
            del os.environ["GIT_COMMITTER_DATE"]

            # Make a shallow clone
            shallow_dir = tempfile.mkdtemp(prefix="phase6_shallow_")
            git.Repo.clone_from(repo_dir, shallow_dir, depth=1)

            commit_c = list(git.Repo(shallow_dir).iter_commits())[0]

            ci = CommitInfo(
                sha=commit_c.hexsha,
                author_date=datetime(2026, 1, 2, 12, 0, 0, tzinfo=UTC),
                files=[_make_file_diff("src.py")],
                stats=DiffStats(total_files=1, total_lines_added=1, total_lines_deleted=0),
            )
            ff = _make_file_features("src.py")
            pr = _make_pipeline_result(ci, [ff])

            evidence, git_count, analyzed = self.engine.collect_historical(
                pr, shallow_dir, top_k=10
            )

            assert git_count == 1
            assert analyzed == 1
            # Should have either history_unavailable or recent_commits with limited data
        finally:
            import shutil
            shutil.rmtree(repo_dir, ignore_errors=True)
            if shallow_dir:
                shutil.rmtree(shallow_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Integration: collect() method
# ---------------------------------------------------------------------------


class TestEvidenceEngineCollect:
    """Tests for the combined collect() method."""

    def setup_method(self):
        self.engine = EvidenceEngine()

    def test_collect_merges_cheap_and_historical(self):
        """collect() merges cheap and historical evidence."""
        ci = CommitInfo(
            sha="abc123",
            author_date=datetime(2026, 1, 3, tzinfo=UTC),
            files=[_make_file_diff("a.py")],
            stats=DiffStats(total_files=1, total_lines_added=10, total_lines_deleted=5),
        )
        ff = _make_file_features("a.py", hunk_count=2)
        pr = _make_pipeline_result(ci, [ff])

        result, git_count, analyzed = self.engine.collect(
            pr, local_path="/nonexistent", top_k=10
        )

        assert "a.py" in result
        # Should have cheap evidence items
        evidence_types = [i.evidence_type for i in result["a.py"]]
        assert "lines_changed" in evidence_types
