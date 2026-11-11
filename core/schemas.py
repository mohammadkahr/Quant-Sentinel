"""Validated execution interface, Table II; concise analytical summaries."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

class Steps(BaseModel):
    model_config = ConfigDict(extra="forbid")
    momentum: str
    liquidity: str
    risk_inv: str
    @field_validator("momentum", "liquidity", "risk_inv")
    @classmethod
    def limit_words(cls, value):
        if not value.strip() or len(value.split()) > 150:
            raise ValueError("Step must contain 1-150 words")
        return value

class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: Steps
    direction: Literal["LONG", "SHORT", "NO-TRADE"]
    confidence: float = Field(ge=.50, le=.95)
    rationale: str
    @field_validator("rationale")
    @classmethod
    def limit_words(cls, value):
        if not value.strip() or len(value.split()) > 100:
            raise ValueError("Rationale must contain 1-100 words")
        return value
