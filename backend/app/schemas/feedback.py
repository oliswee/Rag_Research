from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FeedbackReason = Literal[
    "wrong_paper",
    "missed_evidence",
    "unsupported_citation",
    "incorrect_interpretation",
    "outdated_conflicting",
    "web_result_irrelevant",
]


class FeedbackCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str = Field(min_length=1, max_length=64)
    reason: FeedbackReason
    source_id: str | None = Field(None, max_length=256)
    expected_paper_id: str | None = Field(None, max_length=36)


class RegressionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    synthetic: bool = False


class FixReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reference: str = Field(min_length=1, max_length=160)
