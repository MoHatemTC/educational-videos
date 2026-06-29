"""Request/response schemas for the educational-video pipeline API."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

Language = Literal["en", "egyptian_arabic"]
Mode = Literal["code_tutorial", "web_explainer"]


class JobRequest(BaseModel):
    """Body for ``POST /videos/jobs``."""

    topic: str = Field(min_length=3, max_length=300, description="Topic/title for the video.")
    language: Language = Field(default="egyptian_arabic", description="Narration language.")
    mode: Mode = Field(
        default="code_tutorial", description="code_tutorial = code video; web_explainer = explain a URL."
    )
    url: Optional[str] = Field(default=None, description="Page to explain (required when mode=web_explainer).")

    @model_validator(mode="after")
    def _require_url_for_web(self) -> "JobRequest":
        """web_explainer mode needs a URL to navigate."""
        if self.mode == "web_explainer" and not (self.url or "").strip():
            raise ValueError("url is required when mode='web_explainer'")
        return self


class JobCreateResponse(BaseModel):
    """Response for ``POST /videos/jobs``."""

    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    """Response for ``GET /videos/jobs/{job_id}``."""

    job_id: str
    topic: str
    language: str
    mode: str = "code_tutorial"
    url: Optional[str] = None
    status: str
    current_step: str
    awaiting_approval: bool
    review_status: str
    error: Optional[str] = None
    artifacts: dict[str, Any] = Field(default_factory=dict)


class ReviewArtifact(BaseModel):
    """Response for ``GET /videos/jobs/{job_id}/review`` — the editable artifacts."""

    job_id: str
    topic: str
    language: str
    mode: str = "code_tutorial"
    url: Optional[str] = None
    status: str
    script: Optional[str] = None
    code: Optional[str] = None
    timeline: Optional[dict[str, Any]] = None
    research: Optional[str] = None
    screenshots: Optional[list[str]] = None


class ApprovalRequest(BaseModel):
    """Body for ``POST /videos/jobs/{job_id}/approve`` — optional edits."""

    script: Optional[str] = None
    code: Optional[str] = None


class RejectionRequest(BaseModel):
    """Body for ``POST /videos/jobs/{job_id}/reject``."""

    reason: Optional[str] = None


class TraceRow(BaseModel):
    """One observability record for a pipeline LLM/API call."""

    stage: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    est_cost_usd: float = 0.0
    latency_ms: int = 0


class RagIngestRequest(BaseModel):
    """Body for ``POST /rag/ingest``."""

    docs_dir: Optional[str] = Field(default=None, description="Server-side directory of .md/.txt docs to ingest.")
    source: str = Field(default="manual", description="Logical source tag stored in chunk metadata.")
    version: str = Field(default="latest", description="Version tag stored in chunk metadata.")
    doc_type: str = Field(default="documentation", description="Doc type tag stored in chunk metadata.")


class RagQueryRequest(BaseModel):
    """Body for ``POST /rag/query``."""

    query: str = Field(min_length=2)
    top_k: int = Field(default=5, ge=1, le=20)
    source: Optional[str] = None
    version: Optional[str] = None
    doc_type: Optional[str] = None
