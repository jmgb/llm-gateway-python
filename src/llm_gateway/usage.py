"""Token accounting.

The central invariant: absence of usage is never reported as zero usage. A
provider that returns nothing produces ``TokenUsage.unknown()``, whose
``complete`` flag is ``False``; a provider that genuinely reports zero produces
a complete measurement. Cost estimation depends on that distinction.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


def _add(left: int | None, right: int | None) -> int | None:
    if left is None and right is None:
        return None
    return (left or 0) + (right or 0)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Tokens reported by a provider. ``None`` means "not reported"."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    retrieved_document_tokens: int | None = None
    cached_input_tokens: int | None = None
    partial_aggregate: bool = False
    """Set when this total absorbed an attempt that reported nothing."""

    @classmethod
    def unknown(cls) -> TokenUsage:
        """Usage the provider did not report at all."""
        return cls()

    @property
    def complete(self) -> bool:
        """True only when every billable dimension of every attempt was reported."""
        if self.partial_aggregate:
            return False
        return self.input_tokens is not None and self.output_tokens is not None

    @property
    def billable_input_tokens(self) -> int | None:
        """Input plus retrieved documents, which are billed as input context."""
        if self.input_tokens is None:
            return None
        return self.input_tokens + (self.retrieved_document_tokens or 0)

    @property
    def billable_output_tokens(self) -> int | None:
        """Output plus reasoning, which is billed at the output rate."""
        if self.output_tokens is None:
            return None
        return self.output_tokens + (self.reasoning_tokens or 0)

    def merge(self, other: TokenUsage) -> TokenUsage:
        """Aggregate two attempts. An unknown operand taints the total."""
        return replace(
            self,
            input_tokens=_add(self.input_tokens, other.input_tokens),
            output_tokens=_add(self.output_tokens, other.output_tokens),
            reasoning_tokens=_add(self.reasoning_tokens, other.reasoning_tokens),
            retrieved_document_tokens=_add(
                self.retrieved_document_tokens, other.retrieved_document_tokens
            ),
            cached_input_tokens=_add(self.cached_input_tokens, other.cached_input_tokens),
            partial_aggregate=not (self.complete and other.complete),
        )
