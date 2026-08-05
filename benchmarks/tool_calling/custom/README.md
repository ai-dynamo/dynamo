# Custom tool-calling qualification tests

This directory contains Dynamo's existing parser-correctness matrix and raw
artifact validator. Generic and model-specific cases cover streaming and
nonstreaming tool calls, thinking controls, tool choice, JSON-schema arguments,
parallel calls, tool-result follow-ups, parser boundaries, truncation, history,
and reserved-marker leaks.

The declarative Kimi K3 profile has 76 scenarios and produces 152 records in
the standard two-mode sweep. Other model profiles are selected from the model
name by `kimi_tool_call_probe.py`.

Inspect a profile without sending requests:

```bash
python3 benchmarks/tool_calling/custom/kimi_tool_call_probe.py \
  --case-profile kimi_k3 \
  --list-cases
```

Use `benchmarks.tool_calling.e2e_verifier` for automated execution. It applies
the model request contract, enforces a bounded wall-clock profile, and converts
the detailed report into the versioned `suite-result.json` contract consumed by
deployment harnesses.

See `TESTS.md` for every assertion and `CASE_PROFILES.md` for adding a
declarative model profile. The smaller guided PR validator remains available in
`../pr_validator` for focused local parser checks.
