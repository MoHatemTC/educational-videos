"""Tests and experiments for the self-healing loop."""

import json
from unittest.mock import patch

from sandbox import (
    SandboxConfig,
    SelfHealingLoop,
)

# Config with short retry limit for testing

config = SandboxConfig(use_docker=False, max_correction_attempts=3, log_path="logs/execution_log.jsonl")

loop = SelfHealingLoop(config)

# ── Test 1: Clean code — no healing needed ─────────────────────────

print("=" * 50)
print("TEST 1: Clean code")
result = loop.run("x = 1 + 1\nprint(x)")
print(f"Healed  : {result.healed}")
print(f"Attempts: {result.attempts}")
print(f"Output  : {result.final_result.stdout.strip()}")

# ── Test 2: Broken code — mock LLM fixes it ────────────────────────

print("=" * 50)
print("TEST 2: Broken code healed by mock LLM")
fixed_code = "x = 1\nprint(x)"
with patch.object(loop, "_request_correction", return_value=fixed_code) as mock_llm:
    result = loop.run("print(undefined_variable)")
    print(f"Healed       : {result.healed}")
    print(f"Attempts     : {result.attempts}")
    print(f"LLM called   : {mock_llm.call_count} time(s)")
    print(f"Final code   : {result.final_code.strip()}")

# ── Test 3: LLM keeps returning broken code — exhausted ───────────

print("=" * 50)
print("TEST 3: Loop exhausted — LLM never fixes it")
with patch.object(loop, "_request_correction", return_value="print(still_broken)") as mock_llm:
    result = loop.run("print(broken)")
    print(f"Healed       : {result.healed}")
    print(f"Attempts     : {result.attempts}")
    print(f"LLM called   : {mock_llm.call_count} time(s)")

# ── Test 4: LLM returns None — stops gracefully ────────────────────

print("=" * 50)
print("TEST 4: LLM returns None — loop stops cleanly")
with patch.object(loop, "_request_correction", return_value=None) as mock_llm:
    result = loop.run("print(broken_var)")
    print(f"Healed       : {result.healed}")
    print(f"Attempts     : {result.attempts}")
    print(f"LLM called   : {mock_llm.call_count} time(s)")

# ── Test 5: Check the log file ─────────────────────────────────────

print("=" * 50)
print("TEST 5: Execution log")


with open("logs/execution_log.jsonl") as f:
    lines = f.readlines()
last = json.loads(lines[-1])
print("Last log entry:")
print(f"  attempt : {last['attempt']}")
print(f"  success : {last['success']}")
print(f"  errors  : {last['errors']}")
print(f"  duration: {last['duration_seconds']:.3f}s")
