# Prompts


Include any LLM prompts used verbatim along with a brief rationale.


## Extraction prompt rationale
- Ask the model to output strict JSON with keys matching required fields.
- Provide examples and enforce that missing fields are `null` or empty arrays.


## Audit prompt rationale
- Enumerate risky clause categories and ask for findings with `severity` and `evidence` spans.
- Emphasize not to hallucinate outside provided context.


See `prompts/system_prompt.txt` for the base system prompt used in LLM calls.




