# Evaluation Report

## Design Narrative

The project uses strict Pydantic v2 schemas, schema-constrained generation, and a validation-and-repair loop to convert narration scripts into code-animation timelines. The schema rejects extra fields, unsupported event types, invalid line ranges, and impossible timing such as `end_ms <= start_ms`.

## Why a Two-Stage Prompt Chain

The two-stage chain separates script understanding from timeline synthesis. The segmentation prompt first extracts ordered action segments with `segment_text`, `event_type`, and `notes`. The synthesis prompt then converts those structured segments into the final timeline schema. This reduces the chance that one large prompt mixes planning, schema formatting, and timing decisions incorrectly.

## Trade-offs

The two-stage chain adds one extra LLM call, so it is slower than a single prompt. In return, the intermediate segment array is easier to inspect, test, and repair. It also makes semantic drift easier to detect because the original segment text remains visible before final timeline generation.

## Failure Modes Observed

Common failure modes were invalid JSON syntax, missing temporal fields, incorrect event type names, extra fields, invalid highlight ranges, and timing mistakes where `end_ms` was not greater than `start_ms`. Another risk was semantic drift during repair, where a repair step could produce schema-valid JSON but change the meaning of the original script.

## Repair Strategy

The repair prompt includes the schema, invalid output, validation error, and original script context. Including the source context helps the LLM repair structure without inventing unrelated actions or changing the intended animation sequence.

## Metrics

The evaluation harness tracks:

- `schema_conformance_rate`: percentage of outputs that validate against the Pydantic timeline schema.
- `mean_repair_rounds`: average number of repair attempts used per item.
- `sequence_level_accuracy`: percentage of comparable items where the predicted event-type sequence matches the expected sequence.

## Limitations

Schema validation proves structural correctness, not perfect semantic correctness. Sequence-level accuracy checks event ordering, but it does not fully judge whether generated code is pedagogically ideal. Human review is still useful for quality, pacing, and animation clarity.
