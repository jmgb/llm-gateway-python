"""Saying in the conversation what a provider cannot be told any other way.

An adapter declaring ``structured_outputs=False`` has no API field that binds
the answer to a shape: the most it can send is ``{"type": "json_object"}``.
Requesting that mode leaves two things unsaid, and both fail expensively.

**The schema has nowhere to go.** The caller supplied one; dropping it is not a
neutral choice:

* the model answers valid JSON under keys of its own invention;
* the gateway validates afterwards, rejects it, and still bills the attempt;
* the fallback answers instead, so every structured call is served by the
  second model in the plan at the second model's price.

Nothing in that sequence looks like a bug. The caller gets a correct result and
a fallback notice, and the difference shows up on the invoice rather than in
any test.

**The mode itself has a precondition.** Groq answers HTTP 400 when
``json_object`` is requested and the word "json" appears nowhere in the
messages, and OpenAI documents the same rule. Whoever sets ``response_format``
is the one who owes that word — leaving it to the caller's prompt turns a plain
JSON request into a 400 whenever they did not think to mention it.

Both are facts about the provider, so both are settled here rather than in
every application. What gets added is deliberately minimal: the JSON Schema the
caller already declared, or one sentence asking for JSON. It is not a prompt —
no tone, no task framing, no examples. Those are the application's, and this
package ships no prompts.
"""

from __future__ import annotations

import json

from llm_gateway.contracts import LLMRequest, ResponseFormat

# Both texts name JSON explicitly, and that is load-bearing twice over: it is
# what tells the model the shape, and it is what satisfies the providers that
# reject `json_object` unless the word appears in the messages.
_SCHEMA_INSTRUCTION = (
    "Reply with a single JSON object that validates against this JSON Schema, "
    "using exactly its field names, with no other text and no code fences:"
)

_JSON_MODE_INSTRUCTION = "Reply with a single JSON object and no other text."


def system_prompt_for(request: LLMRequest, *, structured_outputs: bool = False) -> str | None:
    """The caller's system prompt, plus whatever the requested format needs.

    Three cases, and only the format decides which applies:

    * ``TEXT`` — nothing is added; the prompt is the caller's alone.
    * ``JSON_OBJECT`` — one sentence asking for JSON, because a provider that
      requires the word would otherwise answer HTTP 400. No shape is implied,
      since the caller asked for none.
    * ``JSON_SCHEMA`` — the schema itself, so a provider that cannot enforce
      one still tells the model which field names to use.

    ``structured_outputs`` says the adapter binds the shape through an API
    field of its own. Such a provider needs no schema in the conversation —
    sending it again would pay for the same declaration twice — but the
    precondition of ``json_object`` is not about enforcement and still
    applies: the word has to be somewhere in the messages either way.

    A caller who already said "json" is left untouched: the requirement is met,
    and appending to a prompt that does not need it only adds noise the model
    has to reconcile with instructions of its own.
    """
    prompt = request.system_prompt
    instruction = _instruction_for(request, structured_outputs=structured_outputs)
    if instruction is None:
        return prompt
    if prompt is None:
        return instruction
    if request.response_format is ResponseFormat.JSON_OBJECT and _mentions_json(prompt):
        return prompt
    return f"{prompt}\n\n{instruction}"


def _instruction_for(request: LLMRequest, *, structured_outputs: bool) -> str | None:
    if request.response_format is ResponseFormat.JSON_SCHEMA:
        if structured_outputs:
            return None
        schema = request.response_schema
        assert schema is not None  # guaranteed by LLMRequest validation
        declaration = json.dumps(schema.model_json_schema(), ensure_ascii=False, sort_keys=True)
        return f"{_SCHEMA_INSTRUCTION}\n{declaration}"
    if request.response_format is ResponseFormat.JSON_OBJECT:
        return _JSON_MODE_INSTRUCTION
    return None


def _mentions_json(text: str) -> bool:
    return "json" in text.casefold()
