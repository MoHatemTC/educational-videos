# Prompt: segment_script_v1

# Version: v1

# Purpose: Split narration scripts into structured event-planning segments.

You are segmenting a narration script into ordered code animation actions.

Return JSON only.
Do not include markdown.
Do not include explanations.

Return a JSON array. Each array item must contain:

- segment_text: the original or lightly cleaned narration segment
- event_type: one of "type", "run", "highlight", "scroll"
- notes: short implementation notes for the timeline generator

Output format:
[
{
"segment_text": "Define a function called add.",
"event_type": "type",
"notes": "Generate code for a simple add function."
},
{
"segment_text": "Highlight the return line.",
"event_type": "highlight",
"notes": "Highlight the line containing the return statement."
}
]

Rules:

- Preserve the meaning of the original narration.
- Do not invent actions not implied by the narration.
- Use only event_type values: type, run, highlight, scroll.
- Return only the JSON array.

Narration script:
{script}
