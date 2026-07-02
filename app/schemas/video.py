"""Request/response schemas for the educational-video pipeline API."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

Language = Literal["en", "egyptian_arabic"]
Mode = Literal["code_tutorial", "web_explainer"]
VisionActionKind = Literal["click", "scroll", "type", "wait"]


class TTSVoiceSettings(BaseModel):
    """Provider-neutral voice controls accepted by video jobs."""

    stability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    similarity_boost: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    style: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    use_speaker_boost: Optional[bool] = None
    emotion: Optional[str] = Field(default=None, max_length=80)


class TTSSegment(BaseModel):
    """One narration segment with optional voice and emotion/style controls."""

    text: str = Field(min_length=1, max_length=3000)
    voice_id: Optional[str] = Field(default=None, max_length=120)
    language: Optional[Language] = None
    voice_settings: Optional[TTSVoiceSettings] = None


class VisionAction(BaseModel):
    """DOM-independent browser action selected by a vision/browser agent."""

    action: VisionActionKind
    x: Optional[int] = Field(default=None, ge=0)
    y: Optional[int] = Field(default=None, ge=0)
    text: Optional[str] = Field(default=None, max_length=1000)
    delta_y: Optional[int] = Field(default=None)
    wait_ms: Optional[int] = Field(default=None, ge=0, le=10_000)


class JobRequest(BaseModel):
    """Body for ``POST /videos/jobs``."""

    topic: Optional[str] = Field(default=None, min_length=3, max_length=300, description="Topic/title for the video.")
    raw_script: Optional[str] = Field(
        default=None,
        min_length=20,
        max_length=8000,
        description="Optional finished narration script. When supplied, research and script generation are skipped.",
    )
    script: Optional[str] = Field(
        default=None,
        min_length=20,
        max_length=8000,
        description="Alias for raw_script, kept for clients that call it script.",
    )
    language: Language = Field(default="egyptian_arabic", description="Narration language.")
    mode: Mode = Field(
        default="code_tutorial", description="code_tutorial = code video; web_explainer = explain a URL."
    )
    url: Optional[str] = Field(default=None, description="Page to explain (required when mode=web_explainer).")
    tts_settings: Optional[TTSVoiceSettings] = Field(
        default=None, description="Default TTS settings for the narration."
    )
    tts_segments: Optional[list[TTSSegment]] = Field(
        default=None, description="Optional per-segment narration text and TTS settings."
    )
    vision_actions: Optional[list[VisionAction]] = Field(
        default=None, description="Optional coordinate-based actions for web_explainer capture."
    )

    @model_validator(mode="after")
    def _validate_video_input(self) -> "JobRequest":
        """Require a topic or raw script and validate mode-specific fields."""
        if not self.resolved_topic_source() and not self.resolved_raw_script():
            raise ValueError("topic or raw_script is required")
        if self.mode == "web_explainer" and not (self.url or "").strip():
            raise ValueError("url is required when mode='web_explainer'")
        return self

    def resolved_raw_script(self) -> str | None:
        """Return raw narration supplied through either accepted field name."""
        script = (self.raw_script or self.script or "").strip()
        return script or None

    def resolved_topic_source(self) -> str | None:
        """Return the explicitly supplied topic, if any."""
        topic = (self.topic or "").strip()
        return topic or None

    def resolved_topic(self) -> str:
        """Return the topic stored in the job row, deriving one for raw-script jobs."""
        topic = self.resolved_topic_source()
        if topic:
            return topic
        raw_script = self.resolved_raw_script() or "Raw script video"
        return raw_script.splitlines()[0][:80].strip() or "Raw script video"


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
    timeline: Optional[dict[str, Any]] = None
    tts_settings: Optional[TTSVoiceSettings] = None
    tts_segments: Optional[list[TTSSegment]] = None


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
