"""pipeline.py — End-to-end content-generation pipeline.

Orchestrates:
  1. Secure sandbox execution of LLM-generated code
  2. Bounded self-correction loop (up to N LLM retries)
  3. Multi-lingual TTS synthesis (ElevenLabs or Stub)
  4. Real audio duration measurement
  5. Timeline stretch/alignment
  6. Output file persistence (master_timeline.json, segment_timings.json)
  7. Execution log (logs/execution_log.jsonl)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
)

from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    Field,
)
from sandbox import (
    HealingResult,
    SandboxConfig,
    SelfHealingLoop,
)
from tts.timeline_sync import (
    MasterTimeline,
    NarrationSegment,
    TimelineSyncer,
    make_demo_segments,
)
from tts.tts_client import (
    StubTTSClient,
    TTSClient,
    TTSConfig,
)

# loads .env from project root before any other imports


load_dotenv()
# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline config
# ─────────────────────────────────────────────────────────────────────────────


class PipelineConfig(BaseModel):
    """Configuration for the generation pipeline."""

    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    output_dir: str = "output"
    log_dir: str = "logs"
    use_stub_tts: bool = Field(
        default_factory=lambda: not bool(os.getenv("ELEVENLABS_API_KEY", "").strip()),
        description="Use StubTTSClient when no ElevenLabs key is available.",
    )
    inter_segment_gap: float = 0.3


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline input / output contracts
# ─────────────────────────────────────────────────────────────────────────────


class PipelineInput(BaseModel):
    """Caller supplies code + narration segments."""

    code: str = Field(description="Python code string to execute.")
    segments: List[NarrationSegment] = Field(
        default_factory=list,
        description="Narration segments to synthesize and align.",
    )


class PipelineResult(BaseModel):
    """Full pipeline output."""

    code_healed: bool
    final_code: str
    correction_attempts: int
    master_timeline: Optional[Dict] = None
    output_files: Dict[str, str] = Field(default_factory=dict)
    pipeline_duration_seconds: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline implementation
# ─────────────────────────────────────────────────────────────────────────────


class Pipeline:
    """Orchestrates the full generation pipeline.

    Usage
    -----
    ```python
    pipe = Pipeline()
    result = pipe.run(PipelineInput(code="print('hello')", segments=[...]))
    ```
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        """Initialize the pipeline with the provided configuration."""
        self.config = config or PipelineConfig()
        self._setup_dirs()

        # Sandbox
        self.healing_loop = SelfHealingLoop(self.config.sandbox)

        # TTS
        if self.config.use_stub_tts:
            logger.warning("No ELEVENLABS_API_KEY found — using StubTTSClient.")
            self.tts_client: TTSClient = StubTTSClient(self.config.tts)
        else:
            self.tts_client = TTSClient(self.config.tts)

        # Timeline
        self.syncer = TimelineSyncer(
            output_dir=self.config.output_dir,
            inter_segment_gap=self.config.inter_segment_gap,
        )

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def run(self, pipeline_input: PipelineInput) -> PipelineResult:
        """Synchronous end-to-end pipeline execution."""
        t0 = time.perf_counter()
        logger.info("═" * 60)
        logger.info(
            "Pipeline starting — code len=%d chars, segments=%d",
            len(pipeline_input.code),
            len(pipeline_input.segments),
        )

        # ── Phase 1: Self-healing code execution ─────────────────────
        healing: HealingResult = self._phase_code_execution(pipeline_input.code)

        # ── Phase 2: TTS synthesis ────────────────────────────────────
        segments = self._phase_tts_synthesis(pipeline_input.segments, healing)

        # ── Phase 3: Timeline sync ────────────────────────────────────
        master, output_files = self._phase_timeline_sync(segments)

        elapsed = time.perf_counter() - t0
        logger.info("Pipeline complete in %.2fs — healed=%s", elapsed, healing.healed)
        logger.info("═" * 60)

        return PipelineResult(
            code_healed=healing.healed,
            final_code=healing.final_code,
            correction_attempts=healing.attempts,
            master_timeline=json.loads(master.model_dump_json()),
            output_files={k: str(v) for k, v in output_files.items()},
            pipeline_duration_seconds=round(elapsed, 3),
        )

    async def run_async(self, pipeline_input: PipelineInput) -> PipelineResult:
        """Async entry-point — runs phases in executor where needed."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, pipeline_input)

    # ──────────────────────────────────────────────────────────────────
    # Phases
    # ──────────────────────────────────────────────────────────────────

    def _phase_code_execution(self, code: str) -> HealingResult:
        """Phase 1: run code; self-correct until clean or exhausted."""
        logger.info("─── Phase 1: Self-Healing Code Execution ───")
        result = self.healing_loop.run(code)
        if result.healed:
            logger.info("✓ Code healed after %d attempt(s)", result.attempts)
        else:
            logger.warning(
                "✗ Code NOT healed after %d attempt(s) — continuing with best effort",
                result.attempts,
            )
            # Log final failure details
            if result.final_result.errors:
                for err in result.final_result.errors:
                    logger.warning("  Unresolved: %s — %s", err.error_type, err.error_message)
        return result

    def _phase_tts_synthesis(self, segments: List[NarrationSegment], healing: HealingResult) -> List[NarrationSegment]:
        """Phase 2: synthesize each segment's narration text.

        Uses the VALIDATED code from the healing phase for any code-derived
        narration content.  Skips segments with no text.
        """
        logger.info("─── Phase 2: Multi-Lingual TTS Synthesis ───")
        enriched: List[NarrationSegment] = []

        for seg in segments:
            if not seg.text.strip():
                logger.warning("Segment %s has no text — skipping TTS.", seg.segment_id)
                enriched.append(seg)
                continue

            logger.info("  Synthesizing segment '%s' [%s] len=%d chars", seg.segment_id, seg.lang_code, len(seg.text))
            try:
                audio_path = self.tts_client.synthesize(
                    text=seg.text,
                    lang_code=seg.lang_code,
                )
                new_seg = seg.model_copy(deep=True)
                new_seg.audio_path = str(audio_path)
                enriched.append(new_seg)
                logger.info("  ✓ Audio → %s", audio_path)
            except Exception as exc:
                logger.error("  TTS failed for segment '%s': %s", seg.segment_id, exc)
                enriched.append(seg)  # proceed without audio

        return enriched

    def _phase_timeline_sync(self, segments: List[NarrationSegment]) -> tuple[MasterTimeline, dict]:
        """Phase 3: measure audio durations, stretch events, build master timeline."""
        logger.info("─── Phase 3: Timeline Synchronisation ───")
        master = self.syncer.build_master_timeline(segments)
        output_files = self.syncer.save(master)
        logger.info("  ✓ Master timeline: %.2fs total, %d segments", master.total_duration, len(master.segments))
        return master, output_files

    # ──────────────────────────────────────────────────────────────────
    # Housekeeping
    # ──────────────────────────────────────────────────────────────────

    def _setup_dirs(self) -> None:
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI / demo entry-point
# ─────────────────────────────────────────────────────────────────────────────

DEMO_CODE = """\
def fibonacci(n: int) -> list:
    \"\"\"Return the first n Fibonacci numbers.\"\"\"
    if n <= 0:
        return []
    seq = [0, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq[:n]

result = fibonacci(10)
print("Fibonacci(10):", result)
assert len(result) == 10
assert result[0] == 0
assert result[9] == 34
print("All assertions passed.")
"""

BROKEN_CODE = """\
def fibonacci(n: int) -> list:
    seq = [0, 1]
    while len(seq) < n:
        seq.appnd(seq[-1] + seq[-2])  # typo: appnd
    return seq[:n]

result = fibonacci(10)
print("Result:", result)
"""


def main() -> None:
    """Run the demo generation pipeline."""
    logger.info("Starting demo pipeline run…")

    config = PipelineConfig(
        sandbox=SandboxConfig(max_correction_attempts=3),
    )
    pipe = Pipeline(config)

    # Use broken code to showcase self-healing
    segments = make_demo_segments()
    inp = PipelineInput(code=BROKEN_CODE, segments=segments)

    result = pipe.run(inp)

    print("\n" + "═" * 60)
    print("Pipeline Result Summary")
    print("═" * 60)
    print(f"Code healed      : {result.code_healed}")
    print(f"Correction rounds: {result.correction_attempts}")
    print(f"Wall time        : {result.pipeline_duration_seconds:.2f}s")
    print(f"Output files     : {list(result.output_files.values())}")
    if result.master_timeline:
        total = result.master_timeline.get("total_duration", 0)
        n_segs = len(result.master_timeline.get("segments", []))
        print(f"Timeline         : {total:.2f}s, {n_segs} segment(s)")
    print("═" * 60)


if __name__ == "__main__":
    main()
