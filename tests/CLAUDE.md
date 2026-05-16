# Test Authoring Guidance

- Do not use debug log output as the oracle for functional or semantic checks.
- Prefer existing structured router test surfaces such as router helpers, `nvext` response fields, metrics, or direct helper APIs.
- Logs may be included in assertion messages as diagnostics, but they should not determine pass/fail behavior.
