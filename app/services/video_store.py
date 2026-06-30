"""SQLite-backed persistence for video-generation jobs.

This store deliberately avoids the main PostgreSQL service so the educational
video pipeline can run locally without Docker/Postgres.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator

from app.core.config import settings
from app.core.logging import logger
from app.models.video_job import VideoJob


_ALLOWED_UPDATE_COLUMNS = {
    "topic",
    "language",
    "mode",
    "url",
    "status",
    "current_step",
    "awaiting_approval",
    "review_status",
    "error_message",
    "artifacts",
    "created_at",
    "updated_at",
}


class VideoStore:
    """Repository over the local ``video_jobs`` SQLite table."""

    def __init__(self) -> None:
        """Initialize the store path without opening a long-lived DB handle."""
        self.db_path = Path(settings.CHECKPOINT_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a short-lived SQLite connection with safe local defaults."""
        connection = sqlite3.connect(str(self.db_path), timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def init_db(self) -> None:
        """Create and migrate the ``video_jobs`` table."""
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS video_jobs (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    language TEXT NOT NULL DEFAULT 'en',
                    mode TEXT NOT NULL DEFAULT 'code_tutorial',
                    url TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    current_step TEXT NOT NULL DEFAULT 'queued',
                    awaiting_approval INTEGER NOT NULL DEFAULT 0,
                    review_status TEXT NOT NULL DEFAULT 'none',
                    error_message TEXT,
                    artifacts TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS ix_video_jobs_created_at ON video_jobs(created_at)")
            connection.execute("CREATE INDEX IF NOT EXISTS ix_video_jobs_status ON video_jobs(status)")
            self._ensure_columns(connection)

        logger.info("video_store_initialized", db_path=str(self.db_path))

    def _ensure_columns(self, connection: sqlite3.Connection) -> None:
        """Add columns missing from older local DB files."""
        existing = {str(row["name"]) for row in connection.execute("PRAGMA table_info(video_jobs)").fetchall()}
        migrations = {
            "language": "ALTER TABLE video_jobs ADD COLUMN language TEXT NOT NULL DEFAULT 'en'",
            "mode": "ALTER TABLE video_jobs ADD COLUMN mode TEXT NOT NULL DEFAULT 'code_tutorial'",
            "url": "ALTER TABLE video_jobs ADD COLUMN url TEXT",
            "status": "ALTER TABLE video_jobs ADD COLUMN status TEXT NOT NULL DEFAULT 'pending'",
            "current_step": "ALTER TABLE video_jobs ADD COLUMN current_step TEXT NOT NULL DEFAULT 'queued'",
            "awaiting_approval": "ALTER TABLE video_jobs ADD COLUMN awaiting_approval INTEGER NOT NULL DEFAULT 0",
            "review_status": "ALTER TABLE video_jobs ADD COLUMN review_status TEXT NOT NULL DEFAULT 'none'",
            "error_message": "ALTER TABLE video_jobs ADD COLUMN error_message TEXT",
            "artifacts": "ALTER TABLE video_jobs ADD COLUMN artifacts TEXT NOT NULL DEFAULT '{}'",
            "created_at": "ALTER TABLE video_jobs ADD COLUMN created_at TEXT NOT NULL DEFAULT ''",
            "updated_at": "ALTER TABLE video_jobs ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''",
        }
        for column, statement in migrations.items():
            if column not in existing:
                connection.execute(statement)
                logger.info("video_store_column_added", column=column)

    def create_job(
        self,
        job_id: str,
        topic: str,
        language: str = "en",
        mode: str = "code_tutorial",
        url: str | None = None,
    ) -> VideoJob:
        """Insert a new pending job and return it."""
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO video_jobs (
                    id, topic, language, mode, url, status, current_step,
                    awaiting_approval, review_status, error_message, artifacts,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, 'pending', 'queued', 0, 'none', NULL, '{}', ?, ?)
                """,
                (job_id, topic, language, mode, url, now, now),
            )

        job = self.get_job(job_id)
        if job is None:
            raise RuntimeError(f"failed to create video job {job_id}")

        logger.info("video_job_created", job_id=job_id, topic=topic, language=language, mode=mode)
        return job

    def get_job(self, job_id: str) -> VideoJob | None:
        """Return the job by id, or ``None`` if it does not exist."""
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM video_jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row is not None else None

    def list_jobs(self, limit: int = 20) -> list[VideoJob]:
        """Return recent jobs without loading large artifact payloads."""
        safe_limit = max(1, min(int(limit), 100))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    id, topic, language, mode, url, status, current_step,
                    awaiting_approval, review_status, error_message,
                    '{}' AS artifacts, created_at, updated_at
                FROM video_jobs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def update_job(
        self,
        job_id: str,
        *,
        artifacts_merge: dict[str, Any] | None = None,
        **fields: Any,
    ) -> VideoJob | None:
        """Patch a job and optionally merge artifact keys."""
        job = self.get_job(job_id)
        if job is None:
            logger.warning("video_job_update_missing", job_id=job_id)
            return None

        updates: list[str] = []
        values: list[Any] = []

        for key, value in fields.items():
            if key not in _ALLOWED_UPDATE_COLUMNS:
                raise ValueError(f"unknown video_jobs column: {key}")
            updates.append(f"{key} = ?")
            values.append(self._to_storage_value(value))

        if artifacts_merge:
            merged = dict(job.artifacts or {})
            merged.update(artifacts_merge)
            updates.append("artifacts = ?")
            values.append(json.dumps(merged, ensure_ascii=False))

        updates.append("updated_at = ?")
        values.append(datetime.now(UTC).isoformat())
        values.append(job_id)

        with self._connect() as connection:
            connection.execute(f"UPDATE video_jobs SET {', '.join(updates)} WHERE id = ?", values)

        return self.get_job(job_id)

    def _to_storage_value(self, value: Any) -> Any:
        """Convert Python values to SQLite-safe values."""
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return value

    def _row_to_job(self, row: sqlite3.Row) -> VideoJob:
        """Convert a SQLite row to a ``VideoJob`` model."""
        artifacts_raw = str(row["artifacts"] or "{}")
        try:
            parsed_artifacts = json.loads(artifacts_raw)
        except json.JSONDecodeError:
            parsed_artifacts = {}

        artifacts: dict[str, Any] = parsed_artifacts if isinstance(parsed_artifacts, dict) else {}

        return VideoJob(
            id=str(row["id"]),
            topic=str(row["topic"]),
            language=str(row["language"]),
            mode=str(row["mode"]),
            url=row["url"],
            status=str(row["status"]),
            current_step=str(row["current_step"]),
            awaiting_approval=bool(row["awaiting_approval"]),
            review_status=str(row["review_status"]),
            error_message=row["error_message"],
            artifacts=artifacts,
            created_at=self._parse_datetime(row["created_at"]),
            updated_at=self._parse_datetime(row["updated_at"]),
        )

    def _parse_datetime(self, value: object) -> datetime:
        """Parse SQLite datetime text."""
        if not value:
            return datetime.now(UTC)

        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed


video_store = VideoStore()
