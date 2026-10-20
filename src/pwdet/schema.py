"""Report schema: Evidence, Finding, lineage, verdict and the top-level Report.

Everything else in pwdet produces or consumes these models; the JSON schema is exported to
schema/report.v2.json by `pwdet report schema` (report.v1.json is kept for 0.1 reports).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Severity = Literal["info", "low", "medium", "high", "critical"]
SEVERITY_ORDER: list[str] = ["info", "low", "medium", "high", "critical"]
Status = Literal["pass", "warn", "fail", "skipped", "error"]
Layer = Literal["static", "provenance", "blackbox", "whitebox"]
Hop = Literal["suspect", "parent", "base", "unknown"]
EvidenceKind = Literal[
    "file",
    "tensor",
    "config_key",
    "template_snippet",
    "prompt",
    "response",
    "metric",
    "hub_record",
    "diff",
    "note",
]


def severity_rank(s: str) -> int:
    return SEVERITY_ORDER.index(s)


def lower_severity(s: Severity) -> Severity:
    """One step down, never below info."""
    return SEVERITY_ORDER[max(0, severity_rank(s) - 1)]  # type: ignore[return-value]


class Evidence(BaseModel):
    kind: EvidenceKind
    locator: str = Field(description="file#tensor[row=..], config key path, probe id, URL ...")
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)
    reference_locator: str | None = None


class Finding(BaseModel):
    check_id: str
    title: str
    status: Status
    severity: Severity = "info"
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    declared: bool = Field(False, description="publisher disclosed this modification")
    hop: Hop | None = Field(None, description="lineage hop that introduced the observation")
    evidence: list[Evidence] = Field(default_factory=list)
    remediation: str | None = None
    tags: list[str] = Field(default_factory=list)

    @property
    def effective_severity(self) -> Severity:
        """Declared modifications are downgraded one step but never hidden."""
        return lower_severity(self.severity) if self.declared else self.severity

    # Convenience constructors -------------------------------------------------
    @classmethod
    def passed(cls, check_id: str, title: str, **kw: Any) -> Finding:
        return cls(check_id=check_id, title=title, status="pass", severity="info", **kw)

    @classmethod
    def skipped(cls, check_id: str, reason: str, **kw: Any) -> Finding:
        return cls(
            check_id=check_id,
            title=f"skipped: {reason}",
            status="skipped",
            severity="info",
            confidence=0.0,
            evidence=[Evidence(kind="note", locator=check_id, summary=reason)],
            **kw,
        )

    @classmethod
    def errored(cls, check_id: str, exc: BaseException, **kw: Any) -> Finding:
        import traceback

        frames = traceback.extract_tb(exc.__traceback__)
        where = (
            f"{frames[-1].filename.rsplit('/', 1)[-1]}:{frames[-1].lineno}" if frames else check_id
        )
        return cls(
            check_id=check_id,
            title=f"check raised {type(exc).__name__}",
            status="error",
            severity="info",
            confidence=0.0,
            evidence=[
                Evidence(
                    kind="note",
                    locator=where,
                    summary=(str(exc) or type(exc).__name__)[:1500],
                    data={"traceback": "".join(traceback.format_tb(exc.__traceback__))[-3000:]},
                )
            ],
            **kw,
        )


class ModelRef(BaseModel):
    key: str = Field(description="config key or user-supplied identifier")
    kind: Literal["hf-cache", "local", "hf-remote", "hf-mirror", "endpoint"]
    repo: str | None = None
    revision: str | None = None
    path: str | None = Field(
        None, description="snapshot directory for hf-cache/local; sparse mirror for hf-mirror"
    )
    endpoint: str | None = None
    served_model: str | None = None
    expect: list[str] = Field(default_factory=list, description="declared-deviation profiles")

    def label(self) -> str:
        return self.repo or self.served_model or self.path or self.key


class LineageHop(BaseModel):
    hop: Hop
    ref: ModelRef
    relation: Literal["finetune", "quantized", "merge", "adapter", "repack", "unknown"] = "unknown"
    verified: bool = False
    source: str = Field(
        "", description="where the claim came from: config._derivative_of, README, SOURCE.json, hub"
    )
    notes: list[str] = Field(default_factory=list)


VerdictLabel = Literal["clean", "suspicious", "likely_tampered", "malicious", "inconclusive"]


class Verdict(BaseModel):
    label: VerdictLabel
    rationale: list[str] = Field(default_factory=list)
    coverage: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description=(
            "per layer, counted in checks (not findings): eligible, run, pass/warn/fail (the "
            "worst status of each check that ran), partial (ran, part skipped), skipped, "
            "not_applicable, error; plus findings, the number of findings"
        ),
    )


ThreatState = Literal["tested", "partial", "not_tested", "not_applicable"]
AssuranceLevel = Literal["basic", "standard", "thorough", "deep"]


class ThreatCoverage(BaseModel):
    id: str
    title: str
    state: ThreatState
    ran: list[str] = Field(default_factory=list, description="checks that ran for this threat")
    missing: list[str] = Field(
        default_factory=list,
        description="checks that did not (fully) run, why, and how to close it",
    )


class Assurance(BaseModel):
    """What the scan tested, so a verdict is never read as more than it is."""

    level: AssuranceLevel
    summary: str
    threats: list[ThreatCoverage] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


class Report(BaseModel):
    schema_version: Literal["1", "2"] = "2"  # 1: read-only, reports from pwdet 0.1
    tool_version: str
    started: datetime
    finished: datetime | None = None
    host: dict[str, Any] = Field(default_factory=dict)
    profile: str | None = None
    subject: ModelRef
    references: list[ModelRef] = Field(default_factory=list)
    lineage: list[LineageHop] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    verdict: Verdict | None = None
    assurance: Assurance | None = None
    manifest_path: str | None = None
    notes: list[str] = Field(default_factory=list)

    @staticmethod
    def now() -> datetime:
        return datetime.now(UTC)

    def max_effective_severity(self) -> Severity:
        worst: Severity = "info"
        for f in self.findings:
            if f.status in ("warn", "fail") and severity_rank(f.effective_severity) > severity_rank(
                worst
            ):
                worst = f.effective_severity
        return worst
