"""Describing a requested schema to a provider that cannot enforce one.

An adapter declaring ``structured_outputs=False`` has no API field that binds
the answer to a shape: the most it can send is "reply with JSON". That leaves
the schema — which the caller did supply — with nowhere to go, and dropping it
is not a neutral choice:

* the model answers valid JSON under keys of its own invention;
* the gateway validates afterwards, rejects it, and still bills the attempt;
* the fallback answers instead, so every structured call is served by the
  second model in the plan at the second model's price.

Nothing in that sequence looks like a bug. The caller gets a correct result and
a fallback notice, and the difference shows up on the invoice rather than in
any test. So the schema is put where a provider without structured outputs can
still read it: in the conversation.

The description is deliberately minimal — the JSON Schema the caller already
declared, and a sentence saying it is binding. It is not a prompt: no tone, no
task framing, no examples. Those are the application's, and this package ships
no prompts.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

# Also the reason this text names JSON explicitly: Groq rejects `json_object`
# outright unless the word appears in the messages, and OpenAI documents the
# same rule. An adapter that asks for JSON mode owns that requirement instead
# of hoping the caller's own prompt happens to mention it.
_INSTRUCTION = (
    "Reply with a single JSON object that validates against this JSON Schema, "
    "using exactly its field names, with no other text and no code fences:"
)


def schema_instruction(schema: type[BaseModel]) -> str:
    """The schema, stated so a model can follow it without provider support."""
    declaration = json.dumps(schema.model_json_schema(), ensure_ascii=False, sort_keys=True)
    return f"{_INSTRUCTION}\n{declaration}"
