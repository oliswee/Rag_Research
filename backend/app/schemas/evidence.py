"""Public M6 provenance and verification contracts; frozen original evidence."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SupportStatus = Literal[
    "supported", "contradicted", "insufficient_evidence", "unsupported", "unknown"
]
EvidenceType = Literal["full_text", "abstract_only", "metadata_only"]


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(frozen=True)
    paper_id: str
    document_id: str
    document_version: str
    block_id: str
    raw_start: int = Field(ge=0, strict=True)
    raw_end: int = Field(gt=0, strict=True)
    page: int | None = Field(None, ge=1, strict=True)
    bbox: tuple[float, float, float, float] | None = None

    @model_validator(mode="after")
    def ordered(self):
        if self.raw_start >= self.raw_end:
            raise ValueError("Empty original span")
        return self


class EvidenceSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)
    source_id: str
    paper_id: str
    title: str
    doi: str | None = None
    source_origin: Literal["local", "web"]
    parser_origin: str | None = None
    evidence_type: EvidenceType
    source_url: str
    raw_text: str
    spans: tuple[EvidenceSpan, ...]
    version: str
    pdf_pages: tuple[int, ...] = ()
    retrieved_at: str | None = None
    license: str | None = None
    year: int | None = None
    parent_id: str | None = None
    parent_text: str | None = None
    parent_spans: tuple[EvidenceSpan, ...] = ()


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)
    citation_id: str
    source_id: str
    paper_id: str
    doi: str | None = None
    source_origin: Literal["local", "web"]
    evidence_type: EvidenceType
    source_url: str
    raw_quote: str
    span: tuple[EvidenceSpan, ...]
    pdf_page: int | None
    version: str
    support_status: SupportStatus = "supported"
    retrieved_at: str | None = None
    license: str | None = None


class Characteristic(BaseModel):
    value: str | None = None
    raw_quote: str | None = None
    citation_ids: list[str] = Field(default_factory=list)
    support_status: SupportStatus = "unknown"


class StudyCharacteristics(BaseModel):
    population: Characteristic = Field(default_factory=Characteristic)
    study_design: Characteristic = Field(default_factory=Characteristic)
    outcome: Characteristic = Field(default_factory=Characteristic)
    exposure: Characteristic = Field(default_factory=Characteristic)
    effect_measure: Characteristic = Field(default_factory=Characteristic)
    adjustment: Characteristic = Field(default_factory=Characteristic)
    direction: Characteristic = Field(default_factory=Characteristic)
    year: int | None = None
    version: str | None = None
    doi: str | None = None


class AtomicClaim(BaseModel):
    claim_id: str
    text: str = ""
    paper_id: str | None = None
    dimension: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    support_status: SupportStatus = "unknown"
    uncertainty: list[str] = Field(default_factory=list)
    characteristics: StudyCharacteristics = Field(default_factory=StudyCharacteristics)


class GroundedAnswer(BaseModel):
    text: str
    claims: list[AtomicClaim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    evidence: list[dict] = Field(default_factory=list)
    support_status: Literal[
        "supported", "partial", "evidence_only", "abstain", "clarification"
    ] = "abstain"
    uncertainty: list[str] = Field(default_factory=list)


class ComparisonAssessment(BaseModel):
    paper_ids: tuple[str, str]
    status: Literal[
        "comparable_agreement",
        "comparable_conflict",
        "not_comparable",
        "insufficient_evidence",
    ]
    reasons: list[str]
