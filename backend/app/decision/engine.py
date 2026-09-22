"""Deterministic investigation-priority presentation engine (Phase 8.0).

Projects B1 investigation priority and Phase 6 evidence into a structured,
deterministic format. Operates exclusively on InvestigationResult output;
never calls B1 scoring, evidence collection, or feature extraction.
"""
from __future__ import annotations

from backend.app.decision.schemas import (
    DecisionResponse,
    DecisionResult,
    EvidenceSummary,
    FileDecision,
    PriorityBand,
    PrioritySummary,
)
from backend.app.investigation.schemas import InvestigationResult

_DECISION_LIMITATIONS: list[str] = [
    "Priority bands are rank-position groupings, not risk categories.",
    "B1 scores are investigation-priority ranking signals, not probabilities.",
    "Explanations cite observed evidence; they do not assert causation.",
    "Absence of evidence is not evidence of absence.",
    "Historical evidence covers only the most recent changes before the analyzed commit.",
    "Binary files receive no B1 ranking score and are placed in the lowest band.",
    "This analysis does not use machine learning or probabilistic models.",
]

_BAND_LABELS: dict[PriorityBand, str] = {
    PriorityBand.HIGHEST: "Highest-priority file",
    PriorityBand.HIGH: "High-priority file",
    PriorityBand.MEDIUM: "Medium-priority file",
    PriorityBand.LOW: "Low-priority file",
}


class DecisionEngine:
    """Deterministic presentation engine for investigation priority."""

    def decide(self, investigation_result: InvestigationResult) -> DecisionResult:
        """Project InvestigationResult into a deterministic DecisionResult.

        Parameters
        ----------
        investigation_result:
            Output of InvestigationService.investigate(). Frozen Phase 6 pipeline.

        Returns
        -------
        DecisionResult
            Deterministic investigation-priority presentation.
        """
        band_assignments = self._assign_priority_bands(investigation_result.files)
        band_counts = self._count_bands(band_assignments)
        summary = self._compute_summary(investigation_result, band_assignments, band_counts)

        file_decisions: list[FileDecision] = []
        for file_evidence in investigation_result.files:
            band = band_assignments[file_evidence.position]
            is_binary = self._is_binary(file_evidence)
            is_test = self._is_test_file(file_evidence)
            evidence_summaries = self._project_evidence(file_evidence)
            evidence_gaps = self._detect_evidence_gaps(file_evidence, is_binary)
            explanation = self._compose_explanation(
                file_evidence, band, is_binary, is_test, evidence_gaps
            )

            file_decisions.append(
                FileDecision(
                    path=file_evidence.path,
                    priority_band=band,
                    b1_score=file_evidence.score,
                    b1_position=file_evidence.position,
                    total_lines_changed=file_evidence.total_lines_changed,
                    is_binary=is_binary,
                    is_test_file=is_test,
                    evidence_count=len(file_evidence.evidence),
                    evidence_summaries=evidence_summaries,
                    explanation=explanation,
                    evidence_gaps=evidence_gaps,
                )
            )

        return DecisionResult(
            repo_url=investigation_result.repo_url,
            commit_sha=investigation_result.commit_sha,
            short_sha=investigation_result.short_sha,
            strategy=investigation_result.strategy,
            strategy_version="1.0.0",
            feature_version="v1",
            evidence_version=investigation_result.evidence_version,
            decision_version="1.0.0",
            summary=summary,
            files=file_decisions,
            limitations=list(_DECISION_LIMITATIONS),
            warnings=list(investigation_result.warnings),
        )

    def wrap_response(
        self,
        decision: DecisionResult,
        analyzed_at: str,
        elapsed_ms: float,
    ) -> DecisionResponse:
        """Wrap DecisionResult in API envelope with runtime metadata."""
        return DecisionResponse(
            decision=decision,
            analyzed_at=analyzed_at,
            elapsed_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Priority-band assignment
    # ------------------------------------------------------------------

    def _assign_priority_bands(
        self,
        files: list,
    ) -> dict[int, PriorityBand]:
        """Assign priority bands from B1 output order. Purely positional.

        Parameters
        ----------
        files:
            List of FileEvidence in frozen B1 output order.

        Returns
        -------
        dict[int, PriorityBand]
            Mapping from FileEvidence.position to PriorityBand.
        """
        if not files:
            return {}

        non_binary_positions = [
            f.position for f in files if not self._is_binary(f)
        ]
        n_non_binary = len(non_binary_positions)

        bands: dict[int, PriorityBand] = {}

        if n_non_binary == 0:
            for f in files:
                bands[f.position] = PriorityBand.LOW
        elif n_non_binary == 1:
            for f in files:
                if self._is_binary(f):
                    bands[f.position] = PriorityBand.LOW
                else:
                    bands[f.position] = PriorityBand.HIGHEST
        else:
            q1 = n_non_binary // 4
            med = n_non_binary // 2

            for rank, pos in enumerate(non_binary_positions):
                if rank == 0:
                    bands[pos] = PriorityBand.HIGHEST
                elif rank <= q1:
                    bands[pos] = PriorityBand.HIGH
                elif rank <= med:
                    bands[pos] = PriorityBand.MEDIUM
                else:
                    bands[pos] = PriorityBand.LOW

            for f in files:
                if f.position not in bands:
                    bands[f.position] = PriorityBand.LOW

        return bands

    def _count_bands(
        self, band_assignments: dict[int, PriorityBand]
    ) -> dict[PriorityBand, int]:
        """Count files in each priority band."""
        counts = {b: 0 for b in PriorityBand}
        for band in band_assignments.values():
            counts[band] += 1
        return counts

    # ------------------------------------------------------------------
    # Summary computation
    # ------------------------------------------------------------------

    def _compute_summary(
        self,
        investigation_result: InvestigationResult,
        band_assignments: dict[int, PriorityBand],
        band_counts: dict[PriorityBand, int],
    ) -> PrioritySummary:
        """Compute commit-level summary counts."""
        total = investigation_result.total_files
        files_binary = sum(
            1 for f in investigation_result.files if self._is_binary(f)
        )
        files_ranked = total - files_binary
        evidence_available = sum(
            1 for f in investigation_result.files if len(f.evidence) > 0
        )
        evidence_unavailable = total - evidence_available

        return PrioritySummary(
            total_files=total,
            files_ranked=files_ranked,
            files_binary=files_binary,
            highest_count=band_counts[PriorityBand.HIGHEST],
            high_count=band_counts[PriorityBand.HIGH],
            medium_count=band_counts[PriorityBand.MEDIUM],
            low_count=band_counts[PriorityBand.LOW],
            evidence_available=evidence_available,
            evidence_unavailable=evidence_unavailable,
        )

    # ------------------------------------------------------------------
    # Evidence projection
    # ------------------------------------------------------------------

    @staticmethod
    def _project_evidence(file_evidence) -> list[EvidenceSummary]:
        """Project FileEvidence.evidence into EvidenceSummary list.

        Strict field-by-field copy. No transformation.
        """
        return [
            EvidenceSummary(
                category=item.category,
                evidence_type=item.evidence_type,
                description=item.description,
                source=item.source,
                provenance=item.provenance,
            )
            for item in file_evidence.evidence
        ]

    # ------------------------------------------------------------------
    # Binary / test-file derivation
    # ------------------------------------------------------------------

    @staticmethod
    def _is_binary(file_evidence) -> bool:
        """Derive binary status from evidence type presence."""
        return any(
            item.evidence_type == "binary_file" for item in file_evidence.evidence
        )

    @staticmethod
    def _is_test_file(file_evidence) -> bool:
        """Derive test-file status from evidence type presence."""
        return any(
            item.evidence_type == "test_file" for item in file_evidence.evidence
        )

    # ------------------------------------------------------------------
    # Explanation composition
    # ------------------------------------------------------------------

    @staticmethod
    def _compose_explanation(
        file_evidence,
        band: PriorityBand,
        is_binary: bool,
        is_test: bool,
        evidence_gaps: list[str],
    ) -> str:
        """Compose a deterministic, evidence-grounded explanation.

        Sources: FileEvidence structured fields, EvidenceItem presence checks,
        deterministic derivations. No description text parsing.
        """
        if is_binary:
            return "Binary file: no B1 investigation-priority score assigned."

        evidence_types = {item.evidence_type for item in file_evidence.evidence}

        parts: list[str] = []

        # Priority band opening
        parts.append(_BAND_LABELS[band])

        # Change magnitude (from structured field)
        parts.append(
            f"with {file_evidence.total_lines_changed} line(s) changed"
        )

        # Evidence-type-derived statements (presence checks only)
        if "hunk_count" in evidence_types:
            parts.append("across multiple hunks")

        if "function_churn" in evidence_types:
            parts.append("with function declaration changes")

        if "class_churn" in evidence_types:
            parts.append("with class declaration changes")

        if "import_churn" in evidence_types:
            parts.append("with import changes")

        if "recent_commits" in evidence_types:
            parts.append("with prior commit history")

        if "recent_bug_fixes" in evidence_types:
            parts.append("including prior bug-fix commits")

        if "history_unavailable" in evidence_types:
            parts.append("with no prior commit history available")

        if "root_commit" in evidence_types:
            parts.append("analyzed as a root commit")

        # Test context
        if is_test:
            parts.append("This is a test file")

        if "no_test_changes" in evidence_types:
            parts.append("No test files changed in this commit")

        if "test_coupling" in evidence_types:
            parts.append(
                "Both test and production files changed in this commit"
            )

        # No-evidence fallback
        if not file_evidence.evidence:
            return (
                f"Low-priority file with "
                f"{file_evidence.total_lines_changed} line(s) changed; "
                f"no evidence available."
            )

        return "; ".join(parts) + "."

    # ------------------------------------------------------------------
    # Evidence-gap detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_evidence_gaps(file_evidence, is_binary: bool) -> list[str]:
        """Detect evidence gaps for a single file."""
        gaps: list[str] = []

        if is_binary:
            gaps.append("Binary file: no B1 ranking score assigned")

        if not file_evidence.evidence and not is_binary:
            gaps.append("No evidence generated for this file")

        evidence_types = {item.evidence_type for item in file_evidence.evidence}

        if "history_unavailable" in evidence_types:
            gaps.append(
                "Historical commit data not available for this file"
            )

        if "root_commit" in evidence_types:
            gaps.append("Root commit: no parent history available")

        return gaps
