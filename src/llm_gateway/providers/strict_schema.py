"""Pydantic's JSON Schema, rewritten into the subset ``strict`` mode accepts.

``strict: true`` is what turns a schema from a suggestion into a guarantee, and
it is also what makes the Responses API refuse anything outside a narrow
subset. Two of Pydantic's perfectly ordinary outputs are outside it:

* a field with a default is left out of ``required`` — strict mode has no
  notion of optional and rejects the schema;
* no object declares ``additionalProperties``, which strict mode requires to
  be ``false``.

Sending the raw schema therefore buys a 400 on every structured call. Worse, it
buys it *per attempt*, so a fallback chain spends its whole plan discovering the
same thing. The rewrite is recursive because ``$defs`` is where a nested model
ends up, and a schema is only as strict as its least strict definition.

What this module does not do is loosen the schema. A construct strict mode
cannot express is refused by name, not quietly reshaped into something the
model is allowed to answer differently.
"""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel

from llm_gateway.errors import ConfigurationError

# Keywords whose value is itself a schema, or a list of schemas.
_NESTED_SCHEMA_KEYS = ("items", "not", "contains", "propertyNames")
_NESTED_SCHEMA_LIST_KEYS = ("anyOf", "oneOf", "allOf", "prefixItems")
_NAMED_SCHEMA_MAP_KEYS = ("$defs", "definitions", "properties", "patternProperties")


def strict_json_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """The model's schema, normalised for the Responses API's strict mode."""
    return cast("dict[str, Any]", _strict(schema.model_json_schema(), path=schema.__name__))


def _strict(node: Any, *, path: str) -> Any:
    """Rewrite one subschema, returning a copy: the caller's dict is not ours."""
    if isinstance(node, list):
        return [_strict(item, path=f"{path}[{index}]") for index, item in enumerate(node)]
    if not isinstance(node, dict):
        return node

    rewritten: dict[str, Any] = dict(node)

    for key in _NAMED_SCHEMA_MAP_KEYS:
        nested = rewritten.get(key)
        if isinstance(nested, dict):
            rewritten[key] = {
                name: _strict(value, path=f"{path}.{name}") for name, value in nested.items()
            }
    for key in _NESTED_SCHEMA_KEYS:
        if key in rewritten:
            rewritten[key] = _strict(rewritten[key], path=f"{path}.{key}")
    for key in _NESTED_SCHEMA_LIST_KEYS:
        if key in rewritten:
            rewritten[key] = _strict(rewritten[key], path=f"{path}.{key}")

    if _is_object(rewritten):
        _close(rewritten, path=path)
    return rewritten


def _is_object(node: dict[str, Any]) -> bool:
    """An object is anything that declares properties, however it says so."""
    declared = node.get("type")
    if declared == "object" or (isinstance(declared, list) and "object" in declared):
        return True
    return "properties" in node


def _close(node: dict[str, Any], *, path: str) -> None:
    properties = node.get("properties")
    extra = node.get("additionalProperties")

    if extra is not False and (extra is not None or not isinstance(properties, dict)):
        # A free-form object (``dict[str, str]``, ``additionalProperties: true``)
        # has no strict equivalent. Closing it would change what the model is
        # allowed to answer, so the schema is refused before anything is spent.
        raise ConfigurationError(
            f"{path} is a free-form object, which the Responses API cannot enforce in "
            f"strict mode; declare its properties, or ask for "
            f"ResponseFormat.JSON_OBJECT and validate the payload yourself"
        )

    node["additionalProperties"] = False
    # Strict mode has no optional properties: every key must be listed, and a
    # field with a default is emitted by the model rather than assumed.
    node["required"] = list(properties or {})
