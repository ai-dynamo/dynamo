# Custom tool-calling qualification tests

This directory contains Dynamo's existing parser-correctness matrix and raw
artifact validator. Generic and model-specific cases cover streaming and
nonstreaming tool calls, thinking controls, tool choice, JSON-schema arguments,
parallel calls, tool-result follow-ups, parser boundaries, truncation, history,
and reserved-marker leaks.

The automated qualification profile has two explicit subdivisions:

- `generic_cases` pins the same 25 cases for every model and runs each in
  nonstreaming and streaming mode, producing exactly 50 comparable records.
- `model_specific_cases` appends cases only when the resolved model profile
  matches. Kimi K2 currently adds two multi-turn customer regressions, producing
  27 cases and 54 total records for that family.

The `customer_` prefix records provenance, not applicability. Eight customer
cases are generic and two are Kimi K2-specific; nine of the ten pin merged PRs
from the 2026-05-16 through 2026-08-16 audit, while the original calculate-sum
case is retained. The generic set also covers no-argument, scalar, enum, nested,
escaped Unicode string, parallel, tool-choice, and context-isolation behavior.
The verifier records the resolved profile, both case groups, their record counts,
and a selection hash, then rejects a detailed report whose profile, cases,
modes, iterations, or record count drift.

Additional model-specific cases that exercise private control tokens and parser
boundaries remain available for focused diagnostics outside the bounded
qualification selection.

The declarative Kimi K3 profile has 76 scenarios and produces 152 records in
the standard two-mode sweep. Other model profiles are selected from the model
name by `tool_calling_probe.py`.

Inspect a profile without sending requests:

```bash
python3 benchmarks/tool_calling/custom/tool_calling_probe.py \
  --case-profile kimi_k3 \
  --list-cases
```

Use `--cases` with the generic IDs plus the matching model-profile IDs in
`../e2e_verifier/profiles.json` to reproduce the qualification selection
locally. `--exclude-cases` remains available for focused diagnostic runs.

Use `benchmarks.tool_calling.e2e_verifier` for automated execution. It applies
the model request contract, enforces a bounded wall-clock profile, and converts
the detailed report into the versioned `suite-result.json` contract consumed by
deployment harnesses.

See `TESTS.md` for every assertion and the customer case-to-PR provenance map,
and `CASE_PROFILES.md` for adding a declarative model profile. The smaller
guided PR validator remains available in `../pr_validator` for focused local
parser checks.
