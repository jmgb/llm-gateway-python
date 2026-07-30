# Security policy

## Reporting a vulnerability

Please report security issues **privately**, through GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
on this repository — not in a public issue.

Include what you did, what happened, and what you expected. A proof of concept
helps. Please do not include real credentials or real user data in the report.

## Supported versions

While the package is `0.x`, only the latest released version receives fixes.

## What this package promises

These are the properties a security report can reasonably be measured against:

- **It never reads credentials.** No module reads environment variables or
  constructs a client from a key it found by itself. Applications build their
  own SDK clients and inject them.
- **It never logs prompts or responses by default.** Sinks receive metadata
  only: identifiers, model names, token counts, cost, latency, finish reason.
- **Provider error messages are not propagated.** Typed errors are raised with
  a message this package wrote; the original exception is preserved as
  `__cause__` for local debugging, because provider messages routinely echo
  request payloads and occasionally credentials.
- **No provider SDK is imported at import time**, so installing the package
  does not pull in code you did not ask for.

If you find a case where one of those is not true, that is a valid report even
without a further exploit.

## What is out of scope

- Vulnerabilities in provider SDKs themselves — report those upstream.
- Anything requiring an attacker who already controls the calling application.
- Cost overruns from a misconfigured retry or fallback policy. Those are
  visible in `result.execution.attempts` by design; if they are *not* visible,
  that is a bug worth reporting.
