# Test Authoring Guidance

- Prefer structured test surfaces such as response fields, metrics, or direct helper APIs for functional and semantic checks.
- If a router-internal fact is only exposed as a structured tracing event, parse it through a shared test helper.
- Logs may be included in assertion messages as diagnostics.
