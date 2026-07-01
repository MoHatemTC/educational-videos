"""tests/test_pipeline.py — Full test suite for the generation pipeline.

Covers:
  • Sandbox runner (success + failure paths)
  • Error parser (traceback extraction)
  • Prompt builder (content assertions)
  • Self-healing loop (mock LLM)
  • TTS client (stub path + cache)
  • Audio utils (frame counting)
  • Timeline syncer (stretch, RTL, master build)
  • End-to-end pipeline (integration)

All Anthropic and ElevenLabs API calls are mocked — no real keys needed.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import (
    MagicMock,
    patch,
)

import anthropic
import pytest
from pipeline import (
    Pipeline,
    PipelineConfig,
    PipelineInput,
)
from sandbox.config import SandboxConfig
from sandbox.loop import SelfHealingLoop
from sandbox.parser import (
    ExecutionResult,
    parse_execution_output,
)
from sandbox.prompt_builder import build_correction_prompt
from sandbox.runner import SandboxRunner
from tts.audio_utils import (
    _duration_mp3_frames,
    adjust_timestamps,
    compute_stretch_factor,
    get_audio_duration,
)
from tts.timeline_sync import (
    MasterTimeline,
    NarrationSegment,
    TimelineEvent,
    TimelineSyncer,
    is_rtl,
    make_demo_segments,
)
from tts.tts_client import (
    StubTTSClient,
    TTSConfig,
)

_MOCK_ANTHROPIC_RESPONSE = MagicMock()
_MOCK_ANTHROPIC_RESPONSE.content = [MagicMock(text="x = 1\nprint(x)")]


@pytest.fixture(autouse=True)
def mock_anthropic_api(monkeypatch):
    """Patch anthropic.Anthropic so no real API call is ever made in any test.

    Applied automatically to every test via autouse=True.
    """
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _MOCK_ANTHROPIC_RESPONSE
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kwargs: mock_client)


@pytest.fixture
def sandbox_config(tmp_path):
    """Initialize the pipeline with the provided configuration."""
    return SandboxConfig(
        use_docker=False,
        timeout_seconds=10,
        max_correction_attempts=3,
        log_path=str(tmp_path / "execution_log.jsonl"),
    )


@pytest.fixture
def runner(sandbox_config):
    """Initialize the pipeline with the provided configuration."""
    return SandboxRunner(sandbox_config)


@pytest.fixture
def tts_config(tmp_path):
    """Initialize the pipeline with the provided configuration."""
    return TTSConfig(
        api_key="dummy-key-not-real",  # placeholder — ElevenLabs is always stubbed
        cache_dir=str(tmp_path / "tts_cache"),
    )


@pytest.fixture
def stub_tts(tts_config):
    """Initialize the pipeline with the provided configuration."""
    return StubTTSClient(tts_config)


@pytest.fixture
def syncer(tmp_path):
    """Initialize the pipeline with the provided configuration."""
    return TimelineSyncer(output_dir=str(tmp_path / "output"))


@pytest.fixture
def demo_segments():
    """Initialize the pipeline with the provided configuration."""
    return make_demo_segments()


class TestSandboxRunner:
    """Configuration settings for the generation pipeline."""

    def test_run_simple_success(self, runner):
        """Initialize the pipeline with the provided configuration."""
        result = runner.run("x = 1 + 1\nprint(x)")
        assert result.success is True
        assert "2" in result.stdout
        assert result.exit_code == 0
        assert result.duration_seconds >= 0

    def test_run_syntax_error(self, runner):
        """Initialize the pipeline with the provided configuration."""
        result = runner.run("def broken(:\n    pass")
        assert result.success is False
        assert result.exit_code != 0
        assert result.errors

    def test_run_runtime_error(self, runner):
        """Initialize the pipeline with the provided configuration."""
        result = runner.run("raise ValueError('intentional')")
        assert result.success is False
        assert any(e.error_type == "ValueError" for e in result.errors)

    def test_run_name_error(self, runner):
        """Initialize the pipeline with the provided configuration."""
        result = runner.run("print(undefined_var)")
        assert result.success is False
        assert any(e.error_type == "NameError" for e in result.errors)

    def test_run_timeout(self, sandbox_config):
        """Initialize the pipeline with the provided configuration."""
        cfg = sandbox_config.model_copy(update={"timeout_seconds": 2})
        r = SandboxRunner(cfg)
        result = r.run("import time; time.sleep(100)")
        assert result.success is False
        assert result.exit_code != 0

    def test_stdout_captured(self, runner):
        """Initialize the pipeline with the provided configuration."""
        result = runner.run("print('hello world')")
        assert "hello world" in result.stdout

    def test_code_snapshot_preserved(self, runner):
        """Initialize the pipeline with the provided configuration."""
        code = "x = 42\nprint(x)"
        result = runner.run(code)
        assert result.code_snapshot == code

    @pytest.mark.asyncio
    async def test_run_async(self, runner):
        """Initialize the pipeline with the provided configuration."""
        result = await runner.run_async("print('async ok')")
        assert result.success
        assert "async ok" in result.stdout


class TestParser:
    """Configuration settings for the generation pipeline."""

    def _make_result(self, **kwargs) -> ExecutionResult:
        defaults = dict(stdout="", stderr="", exit_code=0, duration_seconds=0.1, code="")
        defaults.update(kwargs)
        return parse_execution_output(**defaults)

    def test_success_path(self):
        """Initialize the pipeline with the provided configuration."""
        r = self._make_result(stdout="ok", exit_code=0)
        assert r.success is True
        assert r.errors == []

    def test_name_error_parsed(self):
        """Initialize the pipeline with the provided configuration."""
        stderr = (
            "Traceback (most recent call last):\n"
            '  File "tmp.py", line 1, in <module>\n'
            "    print(x)\n"
            "NameError: name 'x' is not defined\n"
        )
        r = self._make_result(stderr=stderr, exit_code=1)
        assert not r.success
        assert r.errors[0].error_type == "NameError"
        assert r.errors[0].first_failing_line == 1

    def test_traceback_frames_extracted(self):
        """Initialize the pipeline with the provided configuration."""
        stderr = (
            "Traceback (most recent call last):\n"
            '  File "a.py", line 5, in outer\n'
            "    inner()\n"
            '  File "a.py", line 2, in inner\n'
            "    raise RuntimeError('boom')\n"
            "RuntimeError: boom\n"
        )
        r = self._make_result(stderr=stderr, exit_code=1)
        assert len(r.errors[0].traceback_frames) == 2
        assert r.errors[0].traceback_frames[-1].lineno == 2

    def test_timeout_error(self):
        """Initialize the pipeline with the provided configuration."""
        r = self._make_result(stderr="TimeoutExpired: execution exceeded 15s", exit_code=124)
        assert r.errors[0].error_type == "TimeoutExpired"

    def test_raw_stderr_fallback(self):
        """Initialize the pipeline with the provided configuration."""
        r = self._make_result(stderr="something went wrong", exit_code=1)
        assert r.errors

    def test_duration_stored(self):
        """Initialize the pipeline with the provided configuration."""
        r = self._make_result(exit_code=0, duration_seconds=1.234)
        assert r.duration_seconds == pytest.approx(1.234)


class TestPromptBuilder:
    """Configuration settings for the generation pipeline."""

    def _failed_result(self, code: str = "print(x)") -> ExecutionResult:
        """Initialize the pipeline with the provided configuration."""
        return parse_execution_output(
            stdout="",
            stderr=(
                "Traceback (most recent call last):\n"
                f'  File "t.py", line 1, in <module>\n'
                f"    {code}\n"
                "NameError: name 'x' is not defined\n"
            ),
            exit_code=1,
            duration_seconds=0.05,
            code=code,
        )

    def test_system_prompt_present(self):
        """Initialize the pipeline with the provided configuration."""
        system, _ = build_correction_prompt(self._failed_result(), attempt=1)
        assert "Python debugger" in system

    def test_user_prompt_contains_error(self):
        """Initialize the pipeline with the provided configuration."""
        _, user = build_correction_prompt(self._failed_result(), attempt=1)
        assert "NameError" in user

    def test_user_prompt_contains_code(self):
        """Initialize the pipeline with the provided configuration."""
        code = "print(undefined_variable)"
        _, user = build_correction_prompt(self._failed_result(code), attempt=1)
        assert code in user

    def test_attempt_number_in_prompt(self):
        """Initialize the pipeline with the provided configuration."""
        _, user = build_correction_prompt(self._failed_result(), attempt=3)
        assert "3" in user

    def test_history_included(self):
        """Initialize the pipeline with the provided configuration."""
        result = self._failed_result()
        history = [("old_code = True", result)]
        _, user = build_correction_prompt(result, attempt=2, history=history)
        assert "Prior Correction" in user

    def test_no_history(self):
        """Initialize the pipeline with the provided configuration."""
        result = self._failed_result()
        system, user = build_correction_prompt(result, attempt=1, history=None)
        assert isinstance(system, str)
        assert isinstance(user, str)


class TestSelfHealingLoop:
    """Configuration settings for the generation pipeline."""

    def _make_loop(self, tmp_path, max_attempts=3):
        """Initialize the pipeline with the provided configuration."""
        cfg = SandboxConfig(
            use_docker=False,
            max_correction_attempts=max_attempts,
            log_path=str(tmp_path / "log.jsonl"),
        )
        return SelfHealingLoop(cfg)

    def test_success_on_first_run(self, tmp_path):
        """Initialize the pipeline with the provided configuration."""
        loop = self._make_loop(tmp_path)
        result = loop.run("x = 1\nprint(x)")
        assert result.healed is True
        assert result.attempts == 1

    def test_healing_with_mock_llm(self, tmp_path):
        """Initialize the pipeline with the provided configuration."""
        loop = self._make_loop(tmp_path, max_attempts=2)
        fixed_code = "x = 1\nprint(x)"

        with patch.object(loop, "_request_correction", return_value=fixed_code):
            result = loop.run("print(undefined_var)")

        assert result.healed is True
        assert result.final_code == fixed_code

    def test_exhausted_loop(self, tmp_path):
        """Initialize the pipeline with the provided configuration."""
        loop = self._make_loop(tmp_path, max_attempts=2)

        with patch.object(loop, "_request_correction", return_value="print(still_broken)"):
            result = loop.run("print(broken)")

        assert result.healed is False
        assert result.attempts <= 4

    def test_log_file_written(self, tmp_path):
        """Initialize the pipeline with the provided configuration."""
        loop = self._make_loop(tmp_path)
        loop.run("print('hi')")
        log_path = Path(tmp_path / "log.jsonl")
        assert log_path.exists()
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) >= 1
        record = json.loads(lines[0])
        assert "success" in record
        assert "attempt" in record

    def test_llm_returns_none_stops_loop(self, tmp_path):
        """Initialize the pipeline with the provided configuration."""
        loop = self._make_loop(tmp_path, max_attempts=3)

        with patch.object(loop, "_request_correction", return_value=None):
            result = loop.run("print(broken_var)")

        assert result.healed is False

    @pytest.mark.asyncio
    async def test_run_async(self, tmp_path):
        """Initialize the pipeline with the provided configuration."""
        loop = self._make_loop(tmp_path)
        result = await loop.run_async("print('async heal')")
        assert result.healed is True


class TestStubTTSClient:
    """Configuration settings for the generation pipeline."""

    def test_returns_path(self, stub_tts, tmp_path):
        """Initialize the pipeline with the provided configuration."""
        p = stub_tts.synthesize("Hello world", lang_code="en")
        assert p.exists()
        assert p.suffix == ".mp3"

    def test_cache_hit(self, stub_tts):
        """Initialize the pipeline with the provided configuration."""
        p1 = stub_tts.synthesize("Cache test", lang_code="en")
        t1 = p1.stat().st_mtime
        time.sleep(0.05)
        p2 = stub_tts.synthesize("Cache test", lang_code="en")
        assert p1 == p2
        assert p2.stat().st_mtime == t1  # file not rewritten

    def test_different_texts_different_files(self, stub_tts):
        """Initialize the pipeline with the provided configuration."""
        p1 = stub_tts.synthesize("Text alpha", lang_code="en")
        p2 = stub_tts.synthesize("Text beta", lang_code="en")
        assert p1 != p2

    def test_arabic_voice_selection(self, stub_tts):
        """Initialize the pipeline with the provided configuration."""
        p = stub_tts.synthesize("مرحبا", lang_code="ar")
        assert p.exists()

    @pytest.mark.asyncio
    async def test_synthesize_async(self, stub_tts):
        """Initialize the pipeline with the provided configuration."""
        p = await stub_tts.synthesize_async("Async test", lang_code="en")
        assert p.exists()


# ─────────────────────────────────────────────────────────────────────────────
# tts/audio_utils.py tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAudioUtils:
    """Configuration settings for the generation pipeline."""

    def _write_silent_mp3(self, path: Path, num_frames: int = 20) -> Path:
        """Write a minimal MP3 with MPEG-1 Layer-3 frames."""
        id3 = b"ID3\x03\x00\x00\x00\x00\x00\x00"
        frame_header = b"\xff\xfb\x90\x00"
        frame_body = b"\x00" * 413
        frame = frame_header + frame_body
        path.write_bytes(id3 + frame * num_frames)
        return path

    def test_mp3_frame_duration(self, tmp_path):
        """Initialize the pipeline with the provided configuration."""
        f = self._write_silent_mp3(tmp_path / "test.mp3", num_frames=50)
        dur = _duration_mp3_frames(f)
        assert dur is not None
        assert 1.0 < dur < 2.0

    def test_get_audio_duration_stub_mp3(self, stub_tts, tmp_path):
        """Initialize the pipeline with the provided configuration."""
        p = stub_tts.synthesize("Duration test", lang_code="en")
        try:
            dur = get_audio_duration(p)
            assert dur >= 0
        except RuntimeError:
            pytest.skip("No audio duration backend available in this environment")

    def test_compute_stretch_factor_basic(self):
        """Initialize the pipeline with the provided configuration."""
        factor = compute_stretch_factor(actual_duration=2.0, target_duration=4.0)
        assert factor == pytest.approx(2.0)

    def test_compute_stretch_factor_clamped_high(self):
        """Initialize the pipeline with the provided configuration."""
        factor = compute_stretch_factor(actual_duration=0.5, target_duration=5.0)
        assert factor == 2.0

    def test_compute_stretch_factor_clamped_low(self):
        """Initialize the pipeline with the provided configuration."""
        factor = compute_stretch_factor(actual_duration=10.0, target_duration=1.0)
        assert factor == 0.5

    def test_compute_stretch_factor_zero_guard(self):
        """Initialize the pipeline with the provided configuration."""
        assert compute_stretch_factor(0, 1) == 1.0
        assert compute_stretch_factor(1, 0) == 1.0

    def test_adjust_timestamps(self):
        """Initialize the pipeline with the provided configuration."""
        ts = [0.0, 1.0, 2.0]
        adjusted = adjust_timestamps(ts, stretch_factor=2.0, offset=1.0)
        assert adjusted == [1.0, 3.0, 5.0]

    def test_adjust_timestamps_empty(self):
        """Initialize the pipeline with the provided configuration."""
        assert adjust_timestamps([], 1.5, 0.0) == []

    def test_file_not_found(self, tmp_path):
        """Initialize the pipeline with the provided configuration."""
        with pytest.raises(FileNotFoundError):
            get_audio_duration(tmp_path / "nonexistent.mp3")


# ─────────────────────────────────────────────────────────────────────────────
# tts/timeline_sync.py tests
# ─────────────────────────────────────────────────────────────────────────────


class TestTimelineSync:
    """Configuration settings for the generation pipeline."""

    def test_is_rtl_arabic(self):
        """Initialize the pipeline with the provided configuration."""
        assert is_rtl("ar") is True

    def test_is_rtl_english(self):
        """Initialize the pipeline with the provided configuration."""
        assert is_rtl("en") is False

    def test_is_rtl_hebrew(self):
        """Initialize the pipeline with the provided configuration."""
        assert is_rtl("he") is True

    def test_segment_rtl_flag(self, syncer, stub_tts, tmp_path):
        """Initialize the pipeline with the provided configuration."""
        audio = stub_tts.synthesize("مرحبا", lang_code="ar")
        seg = NarrationSegment(
            segment_id="ar_test",
            lang_code="ar",
            text="مرحبا",
            audio_path=str(audio),
            original_duration_estimate=1.0,
            events=[
                TimelineEvent(event_type="type_char", timestamp=0.0, payload={"char": "م"}),
            ],
        )
        synced = syncer.sync_segment(seg)
        assert synced.rtl is True

    def test_sync_segment_sets_duration(self, syncer, stub_tts):
        """Initialize the pipeline with the provided configuration."""
        audio = stub_tts.synthesize("Hello sync", lang_code="en")
        seg = NarrationSegment(
            segment_id="s1",
            lang_code="en",
            text="Hello sync",
            audio_path=str(audio),
            original_duration_estimate=2.0,
            events=[
                TimelineEvent(event_type="type_char", timestamp=0.0, payload={}),
                TimelineEvent(event_type="pause", timestamp=1.0, duration=0.5, payload={}),
            ],
        )
        try:
            synced = syncer.sync_segment(seg)
            assert synced.audio_duration is not None
            assert synced.audio_duration >= 0
        except RuntimeError:
            pytest.skip("Audio backend unavailable")

    def test_event_timestamps_scaled(self, syncer):
        """Verify event scaling via mocked get_audio_duration — no real file needed."""
        seg = NarrationSegment(
            segment_id="s2",
            lang_code="en",
            text="test",
            audio_path=None,
            original_duration_estimate=2.0,
            events=[
                TimelineEvent(event_type="type_char", timestamp=0.5, payload={}),
                TimelineEvent(event_type="highlight_line", timestamp=1.0, duration=1.0, payload={}),
            ],
        )
        with patch("tts.timeline_sync.get_audio_duration", return_value=4.0):
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
                tf.write(b"\x00" * 10)
                seg.audio_path = tf.name
            try:
                synced = syncer.sync_segment(seg)
                assert synced.stretch_factor == pytest.approx(2.0, abs=0.1)
                assert synced.events[0].timestamp == pytest.approx(1.0, abs=0.05)
            finally:
                os.unlink(tf.name)

    def test_build_master_timeline(self, syncer, demo_segments, stub_tts):
        """Initialize the pipeline with the provided configuration."""
        with_audio = []
        for seg in demo_segments:
            p = stub_tts.synthesize(seg.text, lang_code=seg.lang_code)
            s = seg.model_copy(deep=True)
            s.audio_path = str(p)
            with_audio.append(s)

        try:
            master = syncer.build_master_timeline(with_audio)
        except RuntimeError:
            pytest.skip("Audio backend unavailable")

        assert isinstance(master, MasterTimeline)
        assert master.validated is True
        assert len(master.segments) == len(demo_segments)
        assert master.total_duration >= 0

    def test_save_output_files(self, syncer, demo_segments, stub_tts, tmp_path):
        """Initialize the pipeline with the provided configuration."""
        with_audio = []
        for seg in demo_segments:
            p = stub_tts.synthesize(seg.text, lang_code=seg.lang_code)
            s = seg.model_copy(deep=True)
            s.audio_path = str(p)
            with_audio.append(s)

        try:
            master = syncer.build_master_timeline(with_audio)
        except RuntimeError:
            pytest.skip("Audio backend unavailable")

        files = syncer.save(master)
        assert files["master_timeline"].exists()
        assert files["segment_timings"].exists()

        data = json.loads(files["master_timeline"].read_text(encoding="utf-8"))
        assert "segments" in data
        assert "total_duration" in data


# ─────────────────────────────────────────────────────────────────────────────
# Integration: full pipeline
# ─────────────────────────────────────────────────────────────────────────────


class TestPipelineIntegration:
    """Configuration settings for the generation pipeline."""

    def _make_pipeline(self, tmp_path):
        cfg = PipelineConfig(
            sandbox=SandboxConfig(
                use_docker=False,
                max_correction_attempts=2,
                log_path=str(tmp_path / "logs/execution_log.jsonl"),
            ),
            tts=TTSConfig(
                api_key="dummy-key-not-real",  # placeholder — always stubbed
                cache_dir=str(tmp_path / "tts_cache"),
            ),
            output_dir=str(tmp_path / "output"),
            log_dir=str(tmp_path / "logs"),
            use_stub_tts=True,
        )
        return Pipeline(cfg)

    def test_clean_code_no_healing(self, tmp_path):
        """Initialize the pipeline with the provided configuration."""
        pipe = self._make_pipeline(tmp_path)
        segments = make_demo_segments()
        result = pipe.run(PipelineInput(code="x = 1\nprint(x)", segments=segments))
        assert result.code_healed is True
        assert result.correction_attempts == 1

    def test_broken_code_healed_by_mock_llm(self, tmp_path):
        """Initialize the pipeline with the provided configuration."""
        pipe = self._make_pipeline(tmp_path)
        fixed = "x = 1\nprint(x)"

        with patch.object(pipe.healing_loop, "_request_correction", return_value=fixed):
            result = pipe.run(PipelineInput(code="print(undefined_var)", segments=make_demo_segments()))

        assert result.code_healed is True
        assert result.final_code == fixed

    def test_output_files_created(self, tmp_path):
        """Initialize the pipeline with the provided configuration."""
        pipe = self._make_pipeline(tmp_path)
        pipe.run(PipelineInput(code="print('ok')", segments=make_demo_segments()))
        out_dir = Path(tmp_path / "output")
        assert (out_dir / "master_timeline.json").exists()
        assert (out_dir / "segment_timings.json").exists()

    def test_execution_log_created(self, tmp_path):
        """Initialize the pipeline with the provided configuration."""
        pipe = self._make_pipeline(tmp_path)
        pipe.run(PipelineInput(code="print(1)", segments=[]))
        log = Path(tmp_path / "logs/execution_log.jsonl")
        assert log.exists()

    def test_pipeline_result_schema(self, tmp_path):
        """Initialize the pipeline with the provided configuration."""
        pipe = self._make_pipeline(tmp_path)
        result = pipe.run(PipelineInput(code="print('schema')", segments=[]))
        assert isinstance(result.code_healed, bool)
        assert isinstance(result.final_code, str)
        assert isinstance(result.correction_attempts, int)
        assert result.pipeline_duration_seconds >= 0

    def test_empty_segments_ok(self, tmp_path):
        """Initialize the pipeline with the provided configuration."""
        pipe = self._make_pipeline(tmp_path)
        result = pipe.run(PipelineInput(code="print('no segments')", segments=[]))
        assert result.code_healed is True

    @pytest.mark.asyncio
    async def test_pipeline_async(self, tmp_path):
        """Initialize the pipeline with the provided configuration."""
        pipe = self._make_pipeline(tmp_path)
        result = await pipe.run_async(PipelineInput(code="print('async pipeline')", segments=[]))
        assert result.code_healed is True

    def test_exhausted_loop_result(self, tmp_path):
        """Initialize the pipeline with the provided configuration."""
        pipe = self._make_pipeline(tmp_path)
        always_broken = "print(still_broken_xyz)"

        with patch.object(pipe.healing_loop, "_request_correction", return_value=always_broken):
            result = pipe.run(PipelineInput(code="print(broken_xyz)", segments=[]))

        assert result.code_healed is False
