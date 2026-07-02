"""tests/test_pipeline.py — Full test suite for the integrated pipeline.

Covers:
  • Sandbox runner (success + failure + env-leak regression — #14)
  • Traceback parser (structured error extraction)
  • Self-healing loop (mocked PipelineLLM)
  • Stub TTS client (path + cache + zh→ja bug fix — #17)
  • Audio utils (frame counting, stretch factor)
  • Timeline syncer (stretch, RTL, master build)
  • Timeline adapter (round-trip to shared schema — #15)

All LLM and ElevenLabs API calls are mocked — no real keys needed.

#17 integration: all imports now point at app/services/pipeline/* (no island).
Island's SelfHealingLoop / CorrectionLoopConfig / run_and_parse are NOT used —
main already has self_heal_code() + run_code() wired with PipelineLLM.
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

import pytest

# ── Shared schema (#15) ───────────────────────────────────────────────────────
from app.core.schemas import (
    Timeline,
    validate_timeline_json,
)

# ── Sandbox (main's API) ──────────────────────────────────────────────────────
from app.services.pipeline.sandbox import (
    ExecutionResult,
    SelfHealResult,
    run_code,
    self_heal_code,
)
from app.services.pipeline.sandbox.parser import parse_traceback

# ── TTS (newly ported from island — #17) ─────────────────────────────────────
from app.services.pipeline.tts.audio_utils import (
    _duration_mp3_frames,
    adjust_timestamps,
    compute_stretch_factor,
)
from app.services.pipeline.tts.stub import (
    synthesize_stub,
    voice_id_for_lang,
)
from app.services.pipeline.tts.timeline_adapter import (
    _segment_to_shared_events,
    master_timeline_to_shared,
)
from app.services.pipeline.tts.timeline_sync import (
    MasterTimeline,
    NarrationSegment,
    TimelineEvent,
    TimelineSyncer,
    is_rtl,
    make_demo_segments,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _silent_mp3(path: Path, num_frames: int = 50) -> Path:
    """Write a minimal silent MP3 for tests that need a real audio file."""
    id3 = b"ID3\x03\x00\x00\x00\x00\x00\x00"
    frame = b"\xff\xfb\x90\x00" + b"\x00" * 413
    path.write_bytes(id3 + frame * num_frames)
    return path


def _attach_audio(tmp_path: Path, segments: list) -> list:
    """Attach a silent MP3 to each segment so sync_segment can measure duration."""
    out = []
    for seg in segments:
        s = seg.model_copy(deep=True)
        mp3 = _silent_mp3(tmp_path / f"{s.segment_id}.mp3")
        s.audio_path = str(mp3)
        out.append(s)
    return out


@pytest.fixture
def syncer(tmp_path):
    return TimelineSyncer(output_dir=str(tmp_path / "output"))


@pytest.fixture
def demo_segments():
    return make_demo_segments()


@pytest.fixture
def mock_llm():
    """Minimal PipelineLLM mock — complete() returns clean fixed code."""
    llm = MagicMock()
    llm.complete.return_value = "x = 1\nprint(x)"
    return llm


# ─────────────────────────────────────────────────────────────────────────────
# TestSandboxRunner — uses main's run_code directly
# ─────────────────────────────────────────────────────────────────────────────


class TestSandboxRunner:
    """Sandbox execution via main's secure run_code (no os.environ leak)."""

    def test_run_simple_success(self):
        result = run_code("x = 1 + 1\nprint(x)")
        assert result.ok is True
        assert "2" in result.stdout
        assert result.exit_code == 0

    def test_run_syntax_error(self):
        result = run_code("def broken(:\n    pass")
        assert result.ok is False
        assert result.exit_code != 0

    def test_run_runtime_error(self):
        result = run_code("raise ValueError('intentional')")
        assert result.ok is False
        assert "ValueError" in result.stderr

    def test_run_name_error(self):
        result = run_code("print(undefined_var)")
        assert result.ok is False
        assert "NameError" in result.stderr

    def test_run_timeout(self):
        result = run_code("import time; time.sleep(100)", timeout_s=2)
        assert result.ok is False
        assert result.timed_out is True

    def test_stdout_captured(self):
        result = run_code("print('hello world')")
        assert "hello world" in result.stdout

    def test_no_env_leak(self, monkeypatch):
        """#14 regression: host secrets must never reach sandboxed code."""
        monkeypatch.setenv("FAKE_ANTHROPIC_API_KEY", "sk-leak-me-if-you-can")
        result = run_code(
            "import os; print(repr(os.environ.get('FAKE_ANTHROPIC_API_KEY')))"
        )
        assert result.ok
        assert "sk-leak-me-if-you-can" not in result.stdout
        assert "None" in result.stdout

    def test_path_still_resolves(self, monkeypatch):
        """PATH must be present; unrelated secrets must not leak."""
        monkeypatch.setenv("SOME_UNRELATED_SECRET", "should-not-appear")
        code = (
            "import os\n"
            "print('PATH_present=', bool(os.environ.get('PATH')))\n"
            "print('secret_present=', 'SOME_UNRELATED_SECRET' in os.environ)\n"
        )
        result = run_code(code)
        assert result.ok
        assert "PATH_present= True" in result.stdout
        assert "secret_present= False" in result.stdout


# ─────────────────────────────────────────────────────────────────────────────
# TestParser — uses main's parse_traceback
# ─────────────────────────────────────────────────────────────────────────────


class TestParser:
    def test_returns_none_on_clean_stderr(self):
        assert parse_traceback("") is None
        assert parse_traceback("   ") is None

    def test_name_error_parsed(self):
        stderr = (
            "Traceback (most recent call last):\n"
            '  File "tmp.py", line 1, in <module>\n'
            "    print(x)\n"
            "NameError: name 'x' is not defined\n"
        )
        result = parse_traceback(stderr)
        assert result is not None
        assert result["exception_type"] == "NameError"
        assert result["line"] == 1

    def test_value_error_parsed(self):
        stderr = (
            "Traceback (most recent call last):\n"
            '  File "t.py", line 3, in <module>\n'
            "    raise ValueError('boom')\n"
            "ValueError: boom\n"
        )
        result = parse_traceback(stderr)
        assert result["exception_type"] == "ValueError"
        assert "boom" in result["exception_message"]

    def test_innermost_frame_extracted(self):
        stderr = (
            "Traceback (most recent call last):\n"
            '  File "a.py", line 5, in outer\n'
            "    inner()\n"
            '  File "a.py", line 2, in inner\n'
            "    raise RuntimeError('boom')\n"
            "RuntimeError: boom\n"
        )
        result = parse_traceback(stderr)
        assert result["line"] == 2
        assert result["innermost_frame"] is not None


# ─────────────────────────────────────────────────────────────────────────────
# TestSelfHealingLoop — uses main's self_heal_code with mocked PipelineLLM
# ─────────────────────────────────────────────────────────────────────────────


class TestSelfHealingLoop:
    def test_success_on_first_run(self, mock_llm):
        result = self_heal_code("x = 1\nprint(x)", mock_llm)
        assert result.validated is True
        assert result.result.ok is True

    def test_healing_with_mock_llm(self, mock_llm):
        fixed = "x = 1\nprint(x)"
        mock_llm.complete.return_value = fixed
        result = self_heal_code("print(undefined_var)", mock_llm, max_retries=2)
        assert result.validated is True
        assert result.code == fixed

    def test_exhausted_loop(self, mock_llm):
        mock_llm.complete.return_value = "print(still_broken)"
        result = self_heal_code("print(broken)", mock_llm, max_retries=2)
        assert result.validated is False

    def test_log_populated(self, mock_llm):
        result = self_heal_code("print('hi')", mock_llm)
        assert len(result.log) >= 1
        assert "exit_code" in result.log[0]
        assert "iteration" in result.log[0]

    def test_clean_code_zero_corrections(self, mock_llm):
        result = self_heal_code("print(1)", mock_llm)
        assert result.validated is True
        assert mock_llm.complete.call_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# TestStubTTS — #17 zh→ja bug fix included
# ─────────────────────────────────────────────────────────────────────────────


class TestStubTTS:
    def test_returns_mp3(self):
        p = synthesize_stub("Hello world", lang_code="en")
        assert p.exists()
        assert p.suffix == ".mp3"

    def test_cache_hit(self):
        p1 = synthesize_stub("Cache test", lang_code="en")
        t1 = p1.stat().st_mtime
        time.sleep(0.05)
        p2 = synthesize_stub("Cache test", lang_code="en")
        assert p1 == p2
        assert p2.stat().st_mtime == t1

    def test_different_texts_different_files(self):
        p1 = synthesize_stub("Text alpha", lang_code="en")
        p2 = synthesize_stub("Text beta", lang_code="en")
        assert p1 != p2

    def test_arabic_voice_resolves(self):
        p = synthesize_stub("مرحبا", lang_code="ar")
        assert p.exists()

    def test_zh_does_not_use_japanese_voice(self):
        """#17 bug fix: zh must not silently use the Japanese voice ID."""
        ja_voice = "jBpfuIE2acCO8z3wKNLl"
        assert voice_id_for_lang("zh") != ja_voice, (
            "zh is mapped to the Japanese voice — this was the bug fixed in #17"
        )

    def test_unknown_lang_falls_back_to_english(self):
        en_voice = voice_id_for_lang("en")
        assert voice_id_for_lang("xx") == en_voice


# ─────────────────────────────────────────────────────────────────────────────
# TestAudioUtils
# ─────────────────────────────────────────────────────────────────────────────


class TestAudioUtils:
    def test_mp3_frame_duration(self, tmp_path):
        f = _silent_mp3(tmp_path / "test.mp3", num_frames=50)
        dur = _duration_mp3_frames(f)
        assert dur is not None
        assert 1.0 < dur < 2.0

    def test_compute_stretch_factor_basic(self):
        assert compute_stretch_factor(2.0, 4.0) == pytest.approx(2.0)

    def test_compute_stretch_factor_clamped_high(self):
        assert compute_stretch_factor(0.5, 5.0) == 2.0

    def test_compute_stretch_factor_clamped_low(self):
        assert compute_stretch_factor(10.0, 1.0) == 0.5

    def test_compute_stretch_factor_zero_guard(self):
        assert compute_stretch_factor(0, 1) == 1.0
        assert compute_stretch_factor(1, 0) == 1.0

    def test_adjust_timestamps(self):
        assert adjust_timestamps([0.0, 1.0, 2.0], stretch_factor=2.0, offset=1.0) == [
            1.0,
            3.0,
            5.0,
        ]

    def test_adjust_timestamps_empty(self):
        assert adjust_timestamps([], 1.5, 0.0) == []


# ─────────────────────────────────────────────────────────────────────────────
# TestTimelineSync
# ─────────────────────────────────────────────────────────────────────────────


class TestTimelineSync:
    def test_is_rtl_arabic(self):
        assert is_rtl("ar") is True

    def test_is_rtl_english(self):
        assert is_rtl("en") is False

    def test_is_rtl_hebrew(self):
        assert is_rtl("he") is True

    def test_segment_rtl_flag(self, syncer, tmp_path):
        mp3 = _silent_mp3(tmp_path / "ar.mp3")
        seg = NarrationSegment(
            segment_id="ar_test",
            lang_code="ar",
            text="مرحبا",
            audio_path=str(mp3),
            original_duration_estimate=1.0,
            events=[
                TimelineEvent(event_type="type_char", timestamp=0.0, payload={"char": "م"}),
            ],
        )
        synced = syncer.sync_segment(seg)
        assert synced.rtl is True

    def test_event_timestamps_scaled(self, syncer, tmp_path):
        seg = NarrationSegment(
            segment_id="s2",
            lang_code="en",
            text="test",
            audio_path=None,
            original_duration_estimate=2.0,
            events=[
                TimelineEvent(event_type="type_char", timestamp=0.5, payload={}),
                TimelineEvent(
                    event_type="highlight_line",
                    timestamp=1.0,
                    duration=1.0,
                    payload={},
                ),
            ],
        )
        with patch(
            "app.services.pipeline.tts.timeline_sync.duration_seconds",
            return_value=4.0,
        ):
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
                tf.write(b"\x00" * 10)
                seg.audio_path = tf.name
            try:
                synced = syncer.sync_segment(seg)
                assert synced.stretch_factor == pytest.approx(2.0, abs=0.1)
                assert synced.events[0].timestamp == pytest.approx(1.0, abs=0.05)
            finally:
                os.unlink(tf.name)

    def test_build_master_timeline(self, syncer, demo_segments, tmp_path):
        with_audio = _attach_audio(tmp_path, demo_segments)
        try:
            master = syncer.build_master_timeline(with_audio)
        except RuntimeError:
            pytest.skip("Audio backend unavailable")
        assert isinstance(master, MasterTimeline)
        assert master.validated is True
        assert len(master.segments) == len(demo_segments)

    def test_save_output_files(self, syncer, demo_segments, tmp_path):
        with_audio = _attach_audio(tmp_path, demo_segments)
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
# TestTimelineAdapter — #15 round-trip
# ─────────────────────────────────────────────────────────────────────────────


class TestTimelineAdapter:
    def test_adapts_to_shared_schema(self, syncer, tmp_path):
        segments = _attach_audio(tmp_path, make_demo_segments())
        try:
            master = syncer.build_master_timeline(segments)
        except RuntimeError:
            pytest.skip("Audio backend unavailable")
        shared = master_timeline_to_shared(master)
        assert isinstance(shared, Timeline)
        assert len(shared.events) > 0

    def test_json_round_trip(self, syncer, tmp_path):
        segments = _attach_audio(tmp_path, make_demo_segments())
        try:
            master = syncer.build_master_timeline(segments)
        except RuntimeError:
            pytest.skip("Audio backend unavailable")
        shared = master_timeline_to_shared(master)
        reparsed = validate_timeline_json(shared.model_dump_json())
        assert reparsed == shared

    def test_type_chars_merged(self, syncer, tmp_path):
        segments = _attach_audio(tmp_path, make_demo_segments())
        try:
            master = syncer.build_master_timeline(segments)
        except RuntimeError:
            pytest.skip("Audio backend unavailable")
        shared = master_timeline_to_shared(master)
        type_events = [e for e in shared.events if e.event_type == "type"]
        assert len(type_events) >= 1
        assert "d" in type_events[0].code
        assert "e" in type_events[0].code

    def test_pause_dropped_not_errored(self, syncer, tmp_path):
        segments = _attach_audio(tmp_path, make_demo_segments())
        try:
            master = syncer.build_master_timeline(segments)
        except RuntimeError:
            pytest.skip("Audio backend unavailable")
        shared = master_timeline_to_shared(master)
        assert isinstance(shared, Timeline)

    def test_events_sorted_chronologically(self, syncer, tmp_path):
        segments = _attach_audio(tmp_path, make_demo_segments())
        try:
            master = syncer.build_master_timeline(segments)
        except RuntimeError:
            pytest.skip("Audio backend unavailable")
        shared = master_timeline_to_shared(master)
        starts = [e.start_ms for e in shared.events]
        assert starts == sorted(starts)

    def test_unmapped_event_type_raises(self):
        seg = NarrationSegment(
            segment_id="bad",
            lang_code="en",
            text="x",
            events=[TimelineEvent(event_type="zoom", timestamp=0.0, payload={})],
        )
        with pytest.raises(ValueError, match="unmapped event_type"):
            _segment_to_shared_events(seg)