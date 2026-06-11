# Prompt: synthesize_timeline_v1

# Version: v1

# Purpose: Convert structured script segments into a validated animation timeline.

You are converting narration segments into a strict JSON timeline of code animation events.

Return JSON only.
Do not include markdown.
Do not include explanations.
Do not include comments.

The output must match this JSON schema:
{schema_json}

Input segments contain:

- segment_text
- event_type
- notes

Segments:
{segments_json}

Rules:

- The top-level output must be an object with an "events" array.
- Every event must include event_type, start_ms, and end_ms.
- start_ms must be zero or greater.
- end_ms must be greater than start_ms.
- Use event_type exactly as one of: type, run, highlight, scroll.
- type events must include code.
- run events must include command and may include expected_output.
- highlight events must include start_line and end_line.
- scroll events must include target_line.
- Line numbers must be positive integers.
- Do not add extra fields.
- Preserve the meaning of the source segments.
- Assign reasonable sequential timings in milliseconds.

Return only the final JSON timeline.
