"""The one way a live test gets a real client, or skips saying why it cannot.

Two gaps describe the canonical development environment rather than a fault:
it installs no provider extra, and a contributor need not hold an account with
every provider. Either is a test that cannot run, not a test that failed, so
both skip — and they skip *before* anything is spent.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import pytest

from llm_gateway.errors import ProviderNotInstalled


def live_client(build: Callable[..., Any], *variables: str) -> Any:
    """Build the client from the first variable that holds a key.

    Several variables because some providers are known by more than one name
    for the same key; the first is the one the skip message asks for.
    """
    key = next((os.environ[name] for name in variables if os.environ.get(name)), None)
    if key is None:
        pytest.skip(f"{variables[0]} is not set")
    try:
        return build(api_key=key)
    except ProviderNotInstalled as absent:
        pytest.skip(str(absent))
