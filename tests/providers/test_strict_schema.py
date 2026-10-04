"""The strict subset the Responses API accepts is not what Pydantic emits.

`strict: true` is what makes a schema a guarantee rather than a suggestion, and
it is rejected outright unless every object closes itself and lists every
property as required. Pydantic omits a field with a default from `required` and
never writes `additionalProperties`, so sending its schema unchanged turns the
whole call into a 400 — at the worst moment, when a fallback is already running.
"""

from __future__ import annotations

from typing import Any, Literal

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


def test_a_reference_with_sibling_metadata_is_expanded_and_keeps_the_metadata() -> None:
    class DescribedInvoice(BaseModel):
        line: Line = Field(description="The billed line")

    schema = strict_json_schema(DescribedInvoice)
    line = schema["properties"]["line"]

    assert "$ref" not in line
    assert line["description"] == "The billed line"
    assert line["type"] == "object"
    assert line["additionalProperties"] is False


def test_a_nullable_default_is_removed_from_the_provider_schema() -> None:
    schema = strict_json_schema(Invoice)

    assert "default" not in schema["properties"]["note"]


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


class Cat(BaseModel):
    kind: Literal["cat"]
    lives: int


class Dog(BaseModel):
    kind: Literal["dog"]
    good: bool


class Shelter(BaseModel):
    pet: Cat | Dog = Field(discriminator="kind")


def test_a_discriminated_union_is_sent_as_the_any_of_strict_mode_accepts() -> None:
    """Pydantic writes a tagged union as `oneOf` plus `discriminator`; strict
    mode knows neither, so the call would be a 400 on every attempt. The tags
    make the branches mutually exclusive, so `anyOf` admits the same answers."""
    pet = strict_json_schema(Shelter)["properties"]["pet"]

    assert "oneOf" not in pet
    assert "discriminator" not in pet
    assert {branch["$ref"] for branch in pet["anyOf"]} == {"#/$defs/Cat", "#/$defs/Dog"}


class TreeNode(BaseModel):
    label: str
    parent: TreeNode = Field(description="The node above this one")


def test_a_recursive_reference_with_sibling_metadata_terminates() -> None:
    """Expanding a `$ref` that points back at its own definition never ends,
    so the cycle is closed with the bare reference strict mode supports."""
    schema = strict_json_schema(TreeNode)

    parent = _defs(schema)["TreeNode"]["properties"]["parent"]
    assert parent == {"$ref": "#/$defs/TreeNode"}
