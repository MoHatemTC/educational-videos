"""Video-generation job model.

Persisted in a dedicated SQLite database (see ``app/services/video_store.py``),
kept separate from the template's Postgres-backed user/session tables so the
educational-video pipeline boots without Postgres.
"""

from datetime import UTC, datetime
from typing import Any, Optional

from sqlalchemy import JSON, Column
from sqlmodel import Field

from app.models.base import BaseModel


class VideoJob(BaseModel, table=True):
    """A single topic-to-video generation job and its lifecycle state.

    Attributes:
        id: UUID primary key returned to the client for polling.
        topic: Natural-language topic the video explains.
        language: Narration language (``en`` or ``egyptian_arabic``).
        status: Coarse lifecycle state (pending, running, awaiting_approval,
            approved, rendering, done, error, rejected).
        current_step: Human-readable name of the stage currently executing.
        awaiting_approval: True while the pipeline is paused at the HITL gate.
        review_status: Review decision (none, pending, approved, rejected).
        error_message: Populated when ``status == 'error'``.
        artifacts: Free-form JSON map of produced artifacts (script, code,
            timeline, audio paths, video path, traces).
        updated_at: Last mutation timestamp (UTC).
    """

    __tablename__: Any = "video_jobs"

    id: str = Field(primary_key=True)
    topic: str
    language: str = Field(default="en")
    mode: str = Field(default="code_tutorial", index=True)
    url: Optional[str] = Field(default=None)
    status: str = Field(default="pending", index=True)
    current_step: str = Field(default="queued")
    awaiting_approval: bool = Field(default=False)
    review_status: str = Field(default="none")
    error_message: Optional[str] = Field(default=None)
    artifacts: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
