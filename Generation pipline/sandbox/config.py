"""Sandbox configuration — resource limits, timeouts, Docker settings."""

from __future__ import annotations

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)


class SandboxConfig(BaseModel):
    """All tuneable parameters for the sandbox execution environment."""

    # ── Docker settings ────────────────────────────────────────────────
    use_docker: bool = Field(
        default=False,
        description="Prefer Docker isolation; falls back to subprocess if unavailable.",
    )
    docker_image: str = Field(
        default="python:3.11-slim",
        description="Docker image used when Docker isolation is active.",
    )
    docker_network_disabled: bool = Field(
        default=True,
        description="Disable network access inside the container.",
    )
    docker_mem_limit: str = Field(
        default="256m",
        description="Memory limit for Docker container (Docker notation).",
    )

    # ── Subprocess / resource limits ───────────────────────────────────
    timeout_seconds: int = Field(
        default=15,
        ge=1,
        le=120,
        description="Wall-clock timeout for code execution.",
    )
    max_output_bytes: int = Field(
        default=65_536,
        description="Maximum combined stdout+stderr bytes captured.",
    )

    # ── Self-healing loop ──────────────────────────────────────────────
    max_correction_attempts: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum LLM correction rounds before giving up.",
    )
    anthropic_model: str = Field(
        default="claude-sonnet-4-6",
        description="Anthropic model used for self-correction prompts.",
    )
    correction_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="LLM temperature for code correction calls.",
    )

    # ── Logging ────────────────────────────────────────────────────────
    log_path: str = Field(
        default="logs/execution_log.jsonl",
        description="Path to the JSONL execution log.",
    )

    @field_validator("docker_mem_limit")
    @classmethod
    def _validate_mem_limit(cls, v: str) -> str:
        v = v.strip()
        if v[-1].lower() not in ("m", "g", "b"):
            raise ValueError("docker_mem_limit must end with m/g/b (e.g. '256m')")
        return v

    class Config:
        """Configuration settings for the generation pipeline."""

        populate_by_name = True
