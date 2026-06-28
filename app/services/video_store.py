"""SQLite-backed persistence for video-generation jobs.

Deliberately separate from ``app/services/database.py`` (Postgres): the
educational-video pipeline uses a local SQLite file so it runs without a
Postgres instance. Access is synchronous but fast (local file); methods are
safe to call from FastAPI background tasks because the engine is created with
``check_same_thread=False``.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

from sqlmodel import Session, create_engine, select

from app.core.config import settings
from app.core.logging import logger
from app.models.video_job import VideoJob


class VideoStore:
    """Thin repository over the ``video_jobs`` SQLite table."""

    def __init__(self) -> None:
        """Create the SQLite engine (table creation is deferred to ``init_db``)."""
        db_path = Path(settings.CHECKPOINT_DB_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )

    def init_db(self) -> None:
        """Create the ``video_jobs`` table if it does not yet exist."""
        VideoJob.__table__.create(self.engine, checkfirst=True)  # type: ignore[attr-defined]
        logger.info("video_store_initialized", db_path=settings.CHECKPOINT_DB_PATH)

    def create_job(
        self,
        job_id: str,
        topic: str,
        language: str = "en",
        mode: str = "code_tutorial",
        url: str | None = None,
    ) -> VideoJob:
        """Insert a new pending job and return it."""
        job = VideoJob(
            id=job_id,
            topic=topic,
            language=language,
            mode=mode,
            url=url,
            status="pending",
            current_step="queued",
        )
        with Session(self.engine) as session:
            session.add(job)
            session.commit()
            session.refresh(job)
        logger.info("video_job_created", job_id=job_id, topic=topic, language=language, mode=mode)
        return job

    def get_job(self, job_id: str) -> Optional[VideoJob]:
        """Return the job by id, or None if it does not exist."""
        with Session(self.engine) as session:
            return session.get(VideoJob, job_id)

    def list_jobs(self, limit: int = 20) -> list[VideoJob]:
        """Return the most recently created jobs (newest first)."""
        with Session(self.engine) as session:
            statement = select(VideoJob).order_by(VideoJob.created_at.desc()).limit(limit)  # type: ignore[attr-defined]
            return list(session.exec(statement).all())

    def update_job(
        self,
        job_id: str,
        *,
        artifacts_merge: Optional[dict[str, Any]] = None,
        **fields: Any,
    ) -> Optional[VideoJob]:
        """Patch a job's columns and/or merge keys into its artifacts map.

        Args:
            job_id: Job to update.
            artifacts_merge: Keys merged into the existing artifacts dict
                (a fresh dict is assigned so SQLAlchemy detects the change).
            **fields: Column values to overwrite (e.g. ``status='running'``).

        Returns:
            The refreshed job, or None if the job was not found.
        """
        with Session(self.engine) as session:
            job = session.get(VideoJob, job_id)
            if job is None:
                logger.warning("video_job_update_missing", job_id=job_id)
                return None

            for key, value in fields.items():
                setattr(job, key, value)

            if artifacts_merge:
                merged = dict(job.artifacts or {})
                merged.update(artifacts_merge)
                job.artifacts = merged

            job.updated_at = datetime.now(UTC)
            session.add(job)
            session.commit()
            session.refresh(job)
            return job


# Singleton used by routes and the pipeline.
video_store = VideoStore()
