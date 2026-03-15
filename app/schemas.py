"""Request and response models for the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class MeasurementResponse(BaseModel):
    height_cm: float = Field(..., description="Estimated body height in centimeters.")
    shoulder_width_cm: float = Field(
        ...,
        description="Estimated shoulder width in centimeters.",
    )
    chest_cm: float = Field(..., description="Estimated chest circumference in centimeters.")
    waist_cm: float = Field(..., description="Estimated waist circumference in centimeters.")
    hip_cm: float = Field(..., description="Estimated hip circumference in centimeters.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Heuristic confidence score for the estimate.",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Key assumptions and limitations used to produce the estimate.",
    )


class HealthResponse(BaseModel):
    status: str = Field(..., description="Simple service health status.")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Human-readable error message.")
