"""The strict subset the Responses API accepts is not what Pydantic emits.

`strict: true` is what makes a schema a guarantee rather than a suggestion, and
it is rejected outright unless every object closes itself and lists every
property as required. Pydantic omits a field with a default from `required` and
never writes `additionalProperties`, so sending its schema unchanged turns the
whole call into a 400 — at the worst moment, when a fallback is already running.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from llm_gateway import ConfigurationError
from llm_gateway.providers.strict_schema import strict_json_schema


class Line(BaseModel):
    description: str
    quantity: int = 1


class Invoice(BaseModel):
    number: str
    lines: list[Line]
    note: str | None = None
    reference: Line | None = None


def _defs(schema: dict[str, Any]) -> dict[str, Any]:
    definitions: dict[str, Any] = schema["$defs"]
    return definitions


def test_a_property_with_a_default_is_still_required() -> None:
    """Strict mode has no notion of optional: the model must emit every key."""
    schema = strict_json_schema(Line)

    assert schema["required"] == ["description", "quantity"]


def test_a_nullable_property_is_required_too() -> None:
    schema = strict_json_schema(Invoice)

    assert set(schema["required"]) == {"number", "lines", "note", "reference"}


def test_every_object_closes_itself() -> None:
    schema = strict_json_schema(Invoice)

    assert schema["additionalProperties"] is False
    assert _defs(schema)["Line"]["additionalProperties"] is False


def test_a_nested_definition_is_normalised_not_just_the_root() -> None:
    schema = strict_json_schema(Invoice)

    assert _defs(schema)["Line"]["required"] == ["description", "quantity"]


def test_references_are_left_intact() -> None:
    """`$ref` is how strict mode expresses reuse; rewriting it loses the schema."""
    schema = strict_json_schema(Invoice)

    assert schema["properties"]["lines"]["items"] == {"$ref": "#/$defs/Line"}
    assert any("$ref" in branch for branch in schema["properties"]["reference"]["anyOf"])


def test_an_object_inside_a_union_branch_is_normalised() -> None:
    class Wrapper(BaseModel):
        payload: Line | str

    schema = strict_json_schema(Wrapper)

    objects = [b for b in _defs(schema).values() if b.get("type") == "object"]
    assert objects and all(b["additionalProperties"] is False for b in objects)


def test_the_original_pydantic_schema_is_not_mutated() -> None:
    before = Invoice.model_json_schema()

    strict_json_schema(Invoice)

    assert Invoice.model_json_schema() == before


def test_a_free_form_object_is_refused_before_the_call_is_billed() -> None:
    """`dict[str, str]` cannot be expressed in strict mode.

    Closing it silently would change what the model is allowed to answer;
    sending it unchanged buys a provider 400. Refusing early costs nothing and
    names the field.
    """

    class Loose(BaseModel):
        metadata: dict[str, str] = Field(default_factory=dict)

    with pytest.raises(ConfigurationError, match="metadata"):
        strict_json_schema(Loose)
