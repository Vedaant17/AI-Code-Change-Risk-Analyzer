"""Phase 8.0: Deterministic investigation-priority presentation tests.

Tests priority-band assignment, binary/test-file derivation, evidence
projection, explanation composition, evidence-gap detection, summary
counts, engine determinism, serialization, API contract, and absence
of post-outcome fields.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.app.api.main import app
from backend.app.decision.engine import DecisionEngine
from backend.app.decision.schemas import (
    DecisionResponse,
    DecisionResult,
    PriorityBand,
)
from backend.app.evidence.schemas import EvidenceItem, FileEvidence
from backend.app.investigation.schemas import InvestigationResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_evidence(
    evidence_type: str = "lines_changed",
    category: str = "change",
    source: str = "diff",
    provenance: str = "direct",
    description: str = "test evidence",
) -> EvidenceItem:
    """Build a minimal EvidenceItem."""
    return EvidenceItem(
        category=category,
        evidence_type=evidence_type,
        description=description,
        source=source,
        provenance=provenance,
    )


def _make_file_evidence(
    path: str = "src/main.py",
    score: float = 0.8,
    position: int = 0,
    total_lines_changed: int = 50,
    evidence: list[EvidenceItem] | None = None,
) -> FileEvidence:
    """Build a minimal FileEvidence."""
    return FileEvidence(
        path=path,
        score=score,
        position=position,
        total_lines_changed=total_lines_changed,
        evidence=evidence if evidence is not None else [],
    )


def _make_investigation_result(
    files: list[FileEvidence] | None = None,
    repo_url: str = "https://github.com/user/repo",
    commit_sha: str = "a" * 40,
    short_sha: str = "a" * 8,
    total_files: int | None = None,
) -> InvestigationResult:
    """Build a minimal InvestigationResult."""
    if files is None:
        files = []
    if total_files is None:
        total_files = len(files)
    return InvestigationResult(
        repo_url=repo_url,
        commit_sha=commit_sha,
        short_sha=short_sha,
        total_files=total_files,
        files=files,
    )


def _binary_evidence() -> list[EvidenceItem]:
    """Evidence list indicating a binary file."""
    return [_make_evidence(evidence_type="binary_file", category="structural")]


def _test_file_evidence() -> list[EvidenceItem]:
    """Evidence list indicating a test file."""
    return [_make_evidence(evidence_type="test_file", category="test")]


def _evidence_with_types(*types: str) -> list[EvidenceItem]:
    """Evidence list with specific evidence_type values."""
    return [_make_evidence(evidence_type=t) for t in types]


# ---------------------------------------------------------------------------
# PriorityBand assignment tests
# ---------------------------------------------------------------------------


class TestPriorityBandAssignment:
    """Tests for _assign_priority_bands."""

    def setup_method(self):
        self.engine = DecisionEngine()

    def test_n_zero_all_low(self):
        """N=0: all files get LOW band."""
        fe = _make_file_evidence(score=0.5, position=0, evidence=_binary_evidence())
        result = _make_investigation_result(files=[fe], total_files=1)
        decision = self.engine.decide(result)
        assert decision.files[0].priority_band == PriorityBand.LOW

    def test_n_one_highest(self):
        """N=1: the single non-binary file gets HIGHEST."""
        fe = _make_file_evidence(score=0.9, position=0)
        result = _make_investigation_result(files=[fe])
        decision = self.engine.decide(result)
        assert decision.files[0].priority_band == PriorityBand.HIGHEST

    def test_n_one_with_binary(self):
        """N=1: non-binary HIGHEST, binary LOW."""
        fe_nonbinary = _make_file_evidence(
            path="a.py", score=0.9, position=0
        )
        fe_binary = _make_file_evidence(
            path="img.png", score=0.0, position=1, evidence=_binary_evidence()
        )
        result = _make_investigation_result(
            files=[fe_nonbinary, fe_binary], total_files=2
        )
        decision = self.engine.decide(result)
        bands = {f.path: f.priority_band for f in decision.files}
        assert bands["a.py"] == PriorityBand.HIGHEST
        assert bands["img.png"] == PriorityBand.LOW

    def test_n_two_positions(self):
        """N=2: position 0 HIGHEST, position 1 MEDIUM."""
        files = [
            _make_file_evidence(path="a.py", score=0.9, position=0),
            _make_file_evidence(path="b.py", score=0.6, position=1),
        ]
        result = _make_investigation_result(files=files)
        decision = self.engine.decide(result)
        bands = {f.path: f.priority_band for f in decision.files}
        assert bands["a.py"] == PriorityBand.HIGHEST
        assert bands["b.py"] == PriorityBand.MEDIUM

    def test_n_four_bands(self):
        """N=4: HIGHEST(1), HIGH(1), MEDIUM(1), LOW(1)."""
        files = [
            _make_file_evidence(path=f"file{i}.py", score=0.9 - i * 0.1, position=i)
            for i in range(4)
        ]
        result = _make_investigation_result(files=files)
        decision = self.engine.decide(result)
        bands = [f.priority_band for f in decision.files]
        assert bands[0] == PriorityBand.HIGHEST
        assert bands[1] == PriorityBand.HIGH
        assert bands[2] == PriorityBand.MEDIUM
        assert bands[3] == PriorityBand.LOW

    def test_n_eight_bands(self):
        """N=8: correct band distribution."""
        files = [
            _make_file_evidence(path=f"f{i}.py", score=0.9 - i * 0.05, position=i)
            for i in range(8)
        ]
        result = _make_investigation_result(files=files)
        decision = self.engine.decide(result)
        bands = [f.priority_band for f in decision.files]
        assert bands[0] == PriorityBand.HIGHEST
        assert bands[1] == PriorityBand.HIGH
        assert bands[2] == PriorityBand.HIGH
        assert bands[3] == PriorityBand.MEDIUM
        assert bands[4] == PriorityBand.MEDIUM
        assert bands[5] == PriorityBand.LOW
        assert bands[6] == PriorityBand.LOW
        assert bands[7] == PriorityBand.LOW

    def test_b1_order_preserved(self):
        """Files appear in B1 output order, not sorted by band."""
        files = [
            _make_file_evidence(path="low.py", score=0.2, position=0),
            _make_file_evidence(path="high.py", score=0.9, position=1),
        ]
        result = _make_investigation_result(files=files)
        decision = self.engine.decide(result)
        # Position 0 → HIGHEST even though score is lower
        assert decision.files[0].path == "low.py"
        assert decision.files[0].priority_band == PriorityBand.HIGHEST
        assert decision.files[1].path == "high.py"
        # N=2: rank 1 is MEDIUM (floor(2/2)=1), not LOW
        assert decision.files[1].priority_band == PriorityBand.MEDIUM

    def test_position_preserved(self):
        """FileEvidence.position is not altered."""
        files = [
            _make_file_evidence(path="a.py", score=0.9, position=5),
            _make_file_evidence(path="b.py", score=0.6, position=12),
        ]
        result = _make_investigation_result(files=files, total_files=20)
        decision = self.engine.decide(result)
        assert decision.files[0].b1_position == 5
        assert decision.files[1].b1_position == 12

    def test_binary_always_low(self):
        """Binary files always get LOW regardless of position."""
        files = [
            _make_file_evidence(
                path="img.bin", score=0.0, position=0, evidence=_binary_evidence()
            ),
            _make_file_evidence(
                path="a.py", score=0.5, position=1
            ),
        ]
        result = _make_investigation_result(files=files)
        decision = self.engine.decide(result)
        bands = {f.path: f.priority_band for f in decision.files}
        assert bands["img.bin"] == PriorityBand.LOW
        assert bands["a.py"] == PriorityBand.HIGHEST


# ---------------------------------------------------------------------------
# Binary derivation tests
# ---------------------------------------------------------------------------


class TestIsBinaryDerivation:
    """Tests for _is_binary derivation from evidence types."""

    def setup_method(self):
        self.engine = DecisionEngine()

    def test_binary_when_evidence_present(self):
        """is_binary True when binary_file evidence type present."""
        fe = _make_file_evidence(evidence=_binary_evidence())
        assert self.engine._is_binary(fe) is True

    def test_not_binary_when_absent(self):
        """is_binary False when no binary_file evidence type."""
        fe = _make_file_evidence(evidence=_evidence_with_types("lines_changed"))
        assert self.engine._is_binary(fe) is False

    def test_not_binary_empty_evidence(self):
        """is_binary False when evidence list is empty."""
        fe = _make_file_evidence(evidence=[])
        assert self.engine._is_binary(fe) is False


# ---------------------------------------------------------------------------
# Test-file derivation tests
# ---------------------------------------------------------------------------


class TestIsTestFileDerivation:
    """Tests for _is_test_file derivation from evidence types."""

    def setup_method(self):
        self.engine = DecisionEngine()

    def test_test_file_when_evidence_present(self):
        """is_test_file True when test_file evidence type present."""
        fe = _make_file_evidence(evidence=_test_file_evidence())
        assert self.engine._is_test_file(fe) is True

    def test_not_test_file_when_absent(self):
        """is_test_file False when no test_file evidence type."""
        fe = _make_file_evidence(evidence=_evidence_with_types("lines_changed"))
        assert self.engine._is_test_file(fe) is False


# ---------------------------------------------------------------------------
# EvidenceSummary projection tests
# ---------------------------------------------------------------------------


class TestEvidenceSummary:
    """Tests for strict field-by-field projection from EvidenceItem."""

    def setup_method(self):
        self.engine = DecisionEngine()

    def test_all_fields_projected(self):
        """EvidenceSummary copies all EvidenceItem fields verbatim."""
        item = EvidenceItem(
            category="change",
            evidence_type="lines_changed",
            description="File has 50 lines changed",
            source="diff",
            provenance="direct",
            ref_commit="abc123",
            ref_file="other.py",
        )
        fe = _make_file_evidence(evidence=[item])
        result = _make_investigation_result(files=[fe])
        decision = self.engine.decide(result)
        summary = decision.files[0].evidence_summaries[0]
        assert summary.category == "change"
        assert summary.evidence_type == "lines_changed"
        assert summary.description == "File has 50 lines changed"
        assert summary.source == "diff"
        assert summary.provenance == "direct"

    def test_no_transformation(self):
        """Description is copied verbatim, not parsed."""
        item = _make_evidence(
            description="Ranked at position 3 in B1 output (120 lines changed)"
        )
        fe = _make_file_evidence(evidence=[item])
        result = _make_investigation_result(files=[fe])
        decision = self.engine.decide(result)
        assert decision.files[0].evidence_summaries[0].description == (
            "Ranked at position 3 in B1 output (120 lines changed)"
        )

    def test_multiple_evidence_items(self):
        """All evidence items are projected."""
        items = [
            _make_evidence(evidence_type="lines_changed"),
            _make_evidence(evidence_type="hunk_count"),
            _make_evidence(evidence_type="function_churn"),
        ]
        fe = _make_file_evidence(evidence=items)
        result = _make_investigation_result(files=[fe])
        decision = self.engine.decide(result)
        assert len(decision.files[0].evidence_summaries) == 3
        types = [s.evidence_type for s in decision.files[0].evidence_summaries]
        assert types == ["lines_changed", "hunk_count", "function_churn"]


# ---------------------------------------------------------------------------
# FileDecision schema tests
# ---------------------------------------------------------------------------


class TestFileDecisionSchema:
    """Tests for FileDecision required fields and B1 field preservation."""

    def setup_method(self):
        self.engine = DecisionEngine()

    def test_b1_fields_copied_verbatim(self):
        """b1_score and b1_position match FileEvidence exactly."""
        fe = _make_file_evidence(score=0.7342, position=3, total_lines_changed=120)
        result = _make_investigation_result(files=[fe])
        decision = self.engine.decide(result)
        fd = decision.files[0]
        assert fd.b1_score == pytest.approx(0.7342)
        assert fd.b1_position == 3
        assert fd.total_lines_changed == 120

    def test_float_equality(self):
        """B1 score float equality preserved (not rounded)."""
        score = 0.123456789012345
        fe = _make_file_evidence(score=score, position=0)
        result = _make_investigation_result(files=[fe])
        decision = self.engine.decide(result)
        assert decision.files[0].b1_score == score

    def test_evidence_count(self):
        """evidence_count matches len(evidence)."""
        items = [_make_evidence(evidence_type=f"t{i}") for i in range(5)]
        fe = _make_file_evidence(evidence=items)
        result = _make_investigation_result(files=[fe])
        decision = self.engine.decide(result)
        assert decision.files[0].evidence_count == 5

    def test_path_preserved(self):
        """File path is copied from FileEvidence."""
        fe = _make_file_evidence(path="deep/nested/path/file.py")
        result = _make_investigation_result(files=[fe])
        decision = self.engine.decide(result)
        assert decision.files[0].path == "deep/nested/path/file.py"


# ---------------------------------------------------------------------------
# Explanation composition tests
# ---------------------------------------------------------------------------


class TestExplanationComposition:
    """Tests for deterministic, evidence-grounded explanations."""

    def setup_method(self):
        self.engine = DecisionEngine()

    def test_deterministic_same_input_same_output(self):
        """Same input produces identical explanation text."""
        fe = _make_file_evidence(
            score=0.8,
            position=0,
            total_lines_changed=30,
            evidence=_evidence_with_types("hunk_count", "function_churn"),
        )
        result = _make_investigation_result(files=[fe])
        d1 = self.engine.decide(result)
        d2 = self.engine.decide(result)
        assert d1.files[0].explanation == d2.files[0].explanation

    def test_nonbinary_with_evidence(self):
        """Non-binary file with evidence includes band and change info."""
        fe = _make_file_evidence(
            score=0.8, position=0, total_lines_changed=30,
            evidence=_evidence_with_types("lines_changed"),
        )
        result = _make_investigation_result(files=[fe])
        decision = self.engine.decide(result)
        exp = decision.files[0].explanation
        assert "Highest-priority file" in exp
        assert "30 line(s) changed" in exp
        assert exp.endswith(".")

    def test_binary_explanation(self):
        """Binary file gets fixed explanation."""
        fe = _make_file_evidence(
            score=0.0, position=0, evidence=_binary_evidence()
        )
        result = _make_investigation_result(files=[fe])
        decision = self.engine.decide(result)
        assert decision.files[0].explanation == (
            "Binary file: no B1 investigation-priority score assigned."
        )

    def test_no_evidence_explanation(self):
        """Non-binary file with no evidence gets fallback explanation."""
        fe = _make_file_evidence(
            score=0.3, position=0, total_lines_changed=10, evidence=[]
        )
        result = _make_investigation_result(files=[fe])
        decision = self.engine.decide(result)
        exp = decision.files[0].explanation
        assert "10 line(s) changed" in exp
        assert "no evidence available" in exp

    def test_evidence_type_statements(self):
        """Known evidence types produce expected statements."""
        fe = _make_file_evidence(
            score=0.8,
            position=0,
            total_lines_changed=50,
            evidence=_evidence_with_types(
                "hunk_count", "function_churn", "class_churn", "import_churn"
            ),
        )
        result = _make_investigation_result(files=[fe])
        decision = self.engine.decide(result)
        exp = decision.files[0].explanation
        assert "across multiple hunks" in exp
        assert "with function declaration changes" in exp
        assert "with class declaration changes" in exp
        assert "with import changes" in exp

    def test_historical_evidence_statements(self):
        """Historical evidence types produce expected statements."""
        fe = _make_file_evidence(
            score=0.8,
            position=0,
            total_lines_changed=20,
            evidence=_evidence_with_types(
                "recent_commits", "recent_bug_fixes"
            ),
        )
        result = _make_investigation_result(files=[fe])
        decision = self.engine.decide(result)
        exp = decision.files[0].explanation
        assert "with prior commit history" in exp
        assert "including prior bug-fix commits" in exp

    def test_history_unavailable_statement(self):
        """history_unavailable evidence type produces expected statement."""
        fe = _make_file_evidence(
            score=0.8,
            position=0,
            total_lines_changed=10,
            evidence=_evidence_with_types("history_unavailable"),
        )
        result = _make_investigation_result(files=[fe])
        decision = self.engine.decide(result)
        assert "no prior commit history available" in decision.files[0].explanation

    def test_root_commit_statement(self):
        """root_commit evidence type produces expected statement."""
        fe = _make_file_evidence(
            score=0.8,
            position=0,
            total_lines_changed=100,
            evidence=_evidence_with_types("root_commit"),
        )
        result = _make_investigation_result(files=[fe])
        decision = self.engine.decide(result)
        assert "analyzed as a root commit" in decision.files[0].explanation

    def test_test_file_statement(self):
        """Test file evidence produces expected statement."""
        fe = _make_file_evidence(
            score=0.8,
            position=0,
            total_lines_changed=10,
            evidence=_evidence_with_types("test_file"),
        )
        result = _make_investigation_result(files=[fe])
        decision = self.engine.decide(result)
        assert "This is a test file" in decision.files[0].explanation

    def test_no_test_changes_statement(self):
        """no_test_changes evidence type produces expected statement."""
        fe = _make_file_evidence(
            score=0.8,
            position=0,
            total_lines_changed=10,
            evidence=_evidence_with_types("no_test_changes"),
        )
        result = _make_investigation_result(files=[fe])
        decision = self.engine.decide(result)
        assert "No test files changed" in decision.files[0].explanation

    def test_test_coupling_statement(self):
        """test_coupling evidence type produces expected statement."""
        fe = _make_file_evidence(
            score=0.8,
            position=0,
            total_lines_changed=10,
            evidence=_evidence_with_types("test_coupling"),
        )
        result = _make_investigation_result(files=[fe])
        decision = self.engine.decide(result)
        assert "Both test and production files changed" in decision.files[0].explanation


# ---------------------------------------------------------------------------
# Evidence-gap tests
# ---------------------------------------------------------------------------


class TestEvidenceGaps:
    """Tests for evidence-gap detection."""

    def setup_method(self):
        self.engine = DecisionEngine()

    def test_binary_gap(self):
        """Binary file has binary gap."""
        fe = _make_file_evidence(evidence=_binary_evidence())
        result = _make_investigation_result(files=[fe])
        decision = self.engine.decide(result)
        assert "Binary file: no B1 ranking score assigned" in decision.files[0].evidence_gaps

    def test_no_evidence_gap(self):
        """Non-binary file with empty evidence has no-evidence gap."""
        fe = _make_file_evidence(score=0.5, position=0, evidence=[])
        result = _make_investigation_result(files=[fe])
        decision = self.engine.decide(result)
        assert "No evidence generated for this file" in decision.files[0].evidence_gaps

    def test_history_unavailable_gap(self):
        """history_unavailable evidence type produces gap."""
        fe = _make_file_evidence(
            score=0.5, position=0, evidence=_evidence_with_types("history_unavailable")
        )
        result = _make_investigation_result(files=[fe])
        decision = self.engine.decide(result)
        assert any(
            "Historical commit data not available" in g
            for g in decision.files[0].evidence_gaps
        )

    def test_root_commit_gap(self):
        """root_commit evidence type produces gap."""
        fe = _make_file_evidence(
            score=0.5, position=0, evidence=_evidence_with_types("root_commit")
        )
        result = _make_investigation_result(files=[fe])
        decision = self.engine.decide(result)
        assert any(
            "Root commit: no parent history available" in g
            for g in decision.files[0].evidence_gaps
        )

    def test_no_gap_when_evidence_present(self):
        """File with evidence and no special types has no gaps."""
        fe = _make_file_evidence(
            score=0.5, position=0, evidence=_evidence_with_types("lines_changed")
        )
        result = _make_investigation_result(files=[fe])
        decision = self.engine.decide(result)
        assert decision.files[0].evidence_gaps == []


# ---------------------------------------------------------------------------
# PrioritySummary tests
# ---------------------------------------------------------------------------


class TestPrioritySummary:
    """Tests for commit-level summary counts."""

    def setup_method(self):
        self.engine = DecisionEngine()

    def test_files_ranked_plus_binary_equals_total(self):
        """files_ranked + files_binary == total_files."""
        files = [
            _make_file_evidence(path="a.py", score=0.9, position=0),
            _make_file_evidence(
                path="b.bin", score=0.0, position=1, evidence=_binary_evidence()
            ),
            _make_file_evidence(path="c.py", score=0.5, position=2),
        ]
        result = _make_investigation_result(files=files, total_files=3)
        decision = self.engine.decide(result)
        s = decision.summary
        assert s.files_ranked + s.files_binary == s.total_files

    def test_band_counts_sum(self):
        """Sum of band counts equals total_files."""
        files = [
            _make_file_evidence(path=f"f{i}.py", score=0.9 - i * 0.1, position=i)
            for i in range(8)
        ]
        result = _make_investigation_result(files=files)
        decision = self.engine.decide(result)
        s = decision.summary
        total_banded = s.highest_count + s.high_count + s.medium_count + s.low_count
        assert total_banded == s.total_files

    def test_evidence_available_unavailable_sum(self):
        """evidence_available + evidence_unavailable == total_files."""
        files = [
            _make_file_evidence(
                path="a.py", score=0.9, position=0,
                evidence=_evidence_with_types("t"),
            ),
            _make_file_evidence(path="b.py", score=0.5, position=1, evidence=[]),
            _make_file_evidence(
                path="c.bin", score=0.0, position=2, evidence=_binary_evidence()
            ),
        ]
        result = _make_investigation_result(files=files, total_files=3)
        decision = self.engine.decide(result)
        s = decision.summary
        assert s.evidence_available + s.evidence_unavailable == s.total_files

    def test_binary_count_correct(self):
        """files_binary counts only files with binary_file evidence."""
        files = [
            _make_file_evidence(path="a.py", score=0.9, position=0),
            _make_file_evidence(
                path="b.bin", score=0.0, position=1, evidence=_binary_evidence()
            ),
            _make_file_evidence(path="c.py", score=0.5, position=2),
        ]
        result = _make_investigation_result(files=files, total_files=3)
        decision = self.engine.decide(result)
        assert decision.summary.files_binary == 1
        assert decision.summary.files_ranked == 2


# ---------------------------------------------------------------------------
# DecisionEngine tests
# ---------------------------------------------------------------------------


class TestDecisionEngine:
    """Tests for full DecisionEngine pipeline and determinism."""

    def setup_method(self):
        self.engine = DecisionEngine()

    def test_deterministic_model_dump(self):
        """Same input produces identical model_dump() output."""
        files = [
            _make_file_evidence(path="a.py", score=0.9, position=0),
            _make_file_evidence(path="b.py", score=0.5, position=1),
        ]
        result = _make_investigation_result(files=files)
        d1 = self.engine.decide(result).model_dump()
        d2 = self.engine.decide(result).model_dump()
        assert d1 == d2

    def test_b1_order_preserved_in_output(self):
        """Output files list preserves B1 output order."""
        files = [
            _make_file_evidence(path="z.py", score=0.9, position=0),
            _make_file_evidence(path="a.py", score=0.5, position=1),
            _make_file_evidence(path="m.py", score=0.3, position=2),
        ]
        result = _make_investigation_result(files=files)
        decision = self.engine.decide(result)
        paths = [f.path for f in decision.files]
        assert paths == ["z.py", "a.py", "m.py"]

    def test_full_pipeline_with_mixed_files(self):
        """Full pipeline handles mix of binary, test, and normal files."""
        files = [
            _make_file_evidence(
                path="img.bin", score=0.0, position=0, evidence=_binary_evidence()
            ),
            _make_file_evidence(
                path="test_foo.py", score=0.8, position=1,
                evidence=_evidence_with_types("test_file", "lines_changed"),
            ),
            _make_file_evidence(
                path="src/main.py", score=0.6, position=2,
                evidence=_evidence_with_types("function_churn", "hunk_count"),
            ),
            _make_file_evidence(path="readme.md", score=0.1, position=3, evidence=[]),
        ]
        result = _make_investigation_result(files=files, total_files=4)
        decision = self.engine.decide(result)
        assert len(decision.files) == 4
        assert decision.summary.total_files == 4
        assert decision.summary.files_binary == 1
        assert decision.summary.files_ranked == 3

    def test_warnings_propagated(self):
        """Warnings from InvestigationResult appear in DecisionResult."""
        result = _make_investigation_result()
        result.warnings = ["Test warning"]
        decision = self.engine.decide(result)
        assert "Test warning" in decision.warnings

    def test_empty_files(self):
        """Empty file list produces empty DecisionResult."""
        result = _make_investigation_result(files=[], total_files=0)
        decision = self.engine.decide(result)
        assert decision.files == []
        assert decision.summary.total_files == 0


# ---------------------------------------------------------------------------
# DecisionResult schema tests
# ---------------------------------------------------------------------------


class TestDecisionResultSchema:
    """Tests for DecisionResult schema validation."""

    def setup_method(self):
        self.engine = DecisionEngine()

    def test_required_fields(self):
        """DecisionResult has all required fields."""
        result = _make_investigation_result()
        decision = self.engine.decide(result)
        d = decision.model_dump()
        assert "repo_url" in d
        assert "commit_sha" in d
        assert "short_sha" in d
        assert "strategy" in d
        assert "summary" in d
        assert "files" in d
        assert "limitations" in d
        assert "warnings" in d

    def test_limitations_non_empty(self):
        """DecisionResult contains hardcoded limitations."""
        result = _make_investigation_result()
        decision = self.engine.decide(result)
        assert len(decision.limitations) > 0
        assert any("rank-position" in lim for lim in decision.limitations)

    def test_no_runtime_fields(self):
        """DecisionResult does not contain analyzed_at or elapsed_ms."""
        result = _make_investigation_result()
        decision = self.engine.decide(result)
        d = decision.model_dump()
        assert "analyzed_at" not in d
        assert "elapsed_ms" not in d


# ---------------------------------------------------------------------------
# DecisionResponse envelope tests
# ---------------------------------------------------------------------------


class TestDecisionResponseEnvelope:
    """Tests for DecisionResponse API envelope."""

    def setup_method(self):
        self.engine = DecisionEngine()

    def test_contains_decision_and_runtime(self):
        """DecisionResponse wraps DecisionResult with runtime metadata."""
        result = _make_investigation_result()
        decision = self.engine.decide(result)
        response = self.engine.wrap_response(
            decision, analyzed_at="2026-01-01T00:00:00+00:00", elapsed_ms=42.5
        )
        assert response.decision == decision
        assert response.analyzed_at == "2026-01-01T00:00:00+00:00"
        assert response.elapsed_ms == 42.5

    def test_analyzed_at_is_iso(self):
        """analyzed_at is ISO format string."""
        result = _make_investigation_result()
        decision = self.engine.decide(result)
        response = self.engine.wrap_response(
            decision, analyzed_at="2026-09-22T12:00:00+00:00", elapsed_ms=1.0
        )
        # ISO 8601 contains T separator
        assert "T" in response.analyzed_at


# ---------------------------------------------------------------------------
# API contract tests
# ---------------------------------------------------------------------------


class TestDecisionAPIContract:
    """Tests for POST /analysis/decision endpoint contract."""

    def setup_method(self):
        self.client = TestClient(app)

    def test_endpoint_exists(self):
        """POST /analysis/decision accepts valid request."""
        mock_result = _make_investigation_result(
            files=[
                _make_file_evidence(path="a.py", score=0.8, position=0),
            ],
        )
        with patch(
            "backend.app.api.main.investigation_service.investigate",
            return_value=mock_result,
        ):
            resp = self.client.post(
                "/analysis/decision",
                json={
                    "repo_url": "https://github.com/user/repo",
                    "commit_sha": "a" * 40,
                },
            )
            assert resp.status_code == 200
            body = resp.json()
            assert "decision" in body
            assert "analyzed_at" in body
            assert "elapsed_ms" in body

    def test_422_invalid_request(self):
        """POST /analysis/decision returns 422 for invalid request."""
        resp = self.client.post(
            "/analysis/decision",
            json={"repo_url": "", "commit_sha": "abc"},
        )
        assert resp.status_code == 422

    def test_404_missing_commit(self):
        """POST /analysis/decision returns 404 for missing commit."""
        from backend.app.inference.errors import CommitNotFoundError

        with patch(
            "backend.app.api.main.investigation_service.investigate",
            side_effect=CommitNotFoundError("Commit not found"),
        ):
            resp = self.client.post(
                "/analysis/decision",
                json={
                    "repo_url": "https://github.com/user/repo",
                    "commit_sha": "nonexistent",
                },
            )
            assert resp.status_code == 404

    def test_response_schema_valid(self):
        """Response body validates against DecisionResponse."""
        mock_result = _make_investigation_result()
        with patch(
            "backend.app.api.main.investigation_service.investigate",
            return_value=mock_result,
        ):
            resp = self.client.post(
                "/analysis/decision",
                json={
                    "repo_url": "https://github.com/user/repo",
                    "commit_sha": "a" * 40,
                },
            )
            assert resp.status_code == 200
            # Validate by constructing DecisionResponse from JSON
            dr = DecisionResponse(**resp.json())
            assert dr.decision.repo_url == "https://github.com/user/repo"


# ---------------------------------------------------------------------------
# No post-outcome fields tests
# ---------------------------------------------------------------------------


class TestNoFutureInformation:
    """Tests for absence of post-outcome or future-information fields."""

    def setup_method(self):
        self.engine = DecisionEngine()

    def test_no_defect_label(self):
        """DecisionResult does not contain defect_label."""
        result = _make_investigation_result(
            files=[_make_file_evidence(path="a.py", score=0.8, position=0)]
        )
        decision = self.engine.decide(result)
        d = decision.model_dump()
        assert "defect_label" not in d
        for f in d["files"]:
            assert "defect_label" not in f

    def test_no_post_outcome_fields(self):
        """DecisionResult does not contain corrective_sha, label_source, etc."""
        result = _make_investigation_result()
        decision = self.engine.decide(result)
        d = decision.model_dump()
        forbidden = [
            "corrective_sha",
            "label_source",
            "content_correspondence",
            "confidence",
            "risk_score",
            "probability",
        ]
        for field in forbidden:
            assert field not in d


# ---------------------------------------------------------------------------
# Deterministic output tests
# ---------------------------------------------------------------------------


class TestDeterministicOutput:
    """Tests for byte-for-byte deterministic output."""

    def test_model_dump_equality(self):
        """Two DecisionEngine instances produce identical model_dump()."""
        files = [
            _make_file_evidence(path="a.py", score=0.9, position=0),
            _make_file_evidence(path="b.py", score=0.5, position=1),
        ]
        result = _make_investigation_result(files=files)
        e1 = DecisionEngine()
        e2 = DecisionEngine()
        d1 = e1.decide(result).model_dump()
        d2 = e2.decide(result).model_dump()
        assert d1 == d2

    def test_json_roundtrip(self):
        """DecisionResult survives JSON serialization roundtrip."""
        files = [
            _make_file_evidence(path="a.py", score=0.9, position=0),
            _make_file_evidence(path="b.py", score=0.5, position=1),
        ]
        result = _make_investigation_result(files=files)
        decision = self.engine.decide(result)
        json_str = decision.model_dump_json()
        restored = DecisionResult.model_validate_json(json_str)
        assert restored.model_dump() == decision.model_dump()

    def setup_method(self):
        self.engine = DecisionEngine()
