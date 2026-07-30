"""Token usage must distinguish "no usage reported" from "zero tokens used"."""

from llm_gateway import TokenUsage


def test_unknown_usage_is_not_zero_usage() -> None:
    unknown = TokenUsage.unknown()

    assert unknown.complete is False
    assert unknown.input_tokens is None
    assert unknown.output_tokens is None


def test_reported_zero_is_a_real_measurement() -> None:
    reported = TokenUsage(input_tokens=0, output_tokens=0)

    assert reported.complete is True
    assert reported.input_tokens == 0


def test_partial_usage_is_incomplete() -> None:
    partial = TokenUsage(input_tokens=100, output_tokens=None)

    assert partial.complete is False
    assert partial.input_tokens == 100


def test_billable_output_includes_reasoning_tokens() -> None:
    usage = TokenUsage(input_tokens=10, output_tokens=20, reasoning_tokens=5)

    assert usage.billable_output_tokens == 25


def test_billable_input_includes_retrieved_documents() -> None:
    usage = TokenUsage(input_tokens=10, output_tokens=20, retrieved_document_tokens=40)

    assert usage.billable_input_tokens == 50


def test_usage_aggregates_across_attempts() -> None:
    first = TokenUsage(input_tokens=10, output_tokens=5)
    second = TokenUsage(input_tokens=7, output_tokens=3)

    total = first.merge(second)

    assert total.input_tokens == 17
    assert total.output_tokens == 8
    assert total.complete is True


def test_aggregating_unknown_usage_taints_the_total() -> None:
    known = TokenUsage(input_tokens=10, output_tokens=5)

    total = known.merge(TokenUsage.unknown())

    assert total.complete is False
    assert total.input_tokens == 10
