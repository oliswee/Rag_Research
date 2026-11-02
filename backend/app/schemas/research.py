"""Verified comparison request and matrix response."""

from pydantic import BaseModel, Field, model_validator

from app.schemas.evidence import AtomicClaim, Citation, ComparisonAssessment
from app.schemas.query import QueryRequest


class CompareRequest(BaseModel):
    paper_ids: list[str] = Field(min_length=2, max_length=5)
    question: str | None = Field(None, max_length=2000)
    local_only: bool = False
    only_uploaded: bool = False
    latest: bool = False
    year_from: int | None = Field(None, ge=1800, le=2100)
    year_to: int | None = Field(None, ge=1800, le=2100)
    provider: str | None = None

    @model_validator(mode="after")
    def scope(self):
        if len(set(self.paper_ids)) != len(self.paper_ids):
            raise ValueError("Choose distinct papers")
        QueryRequest(
            query="validate scope",
            provider=self.provider,
            year_from=self.year_from,
            year_to=self.year_to,
        )
        if self.year_from and self.year_to and self.year_from > self.year_to:
            raise ValueError("From year must not exceed to year")
        return self


class ComparisonCell(BaseModel):
    paper_id: str
    text: str
    evidence: list[dict] = Field(default_factory=list)
    claims: list[AtomicClaim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    support_status: str = "unknown"
    uncertainty: list[str] = Field(default_factory=list)


class ComparisonRow(BaseModel):
    label: str
    cells: list[ComparisonCell]
    comparisons: list[ComparisonAssessment] = Field(default_factory=list)


class ComparisonResponse(BaseModel):
    run_id: str
    papers: list[dict]
    rows: list[ComparisonRow]
    coverage: dict = Field(default_factory=dict)
    notices: list[str] = Field(default_factory=list)
    web_records: list[dict] = Field(default_factory=list)
    verification: dict = Field(default_factory=dict)
