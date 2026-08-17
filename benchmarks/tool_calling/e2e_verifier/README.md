# Dynamo E2E Verifier Contract

This package is the Dynamo-owned boundary between tool-calling correctness tests
and external deployment harnesses. Dynamo owns the test profiles, validators,
benchmark versions, scoring, and normalized result contract. An infrastructure
harness owns endpoint deployment, execution, aggregation, notifications, and
dashboard publication.

All normal profiles intentionally use bounded qualification subsets:

| Suite | Coverage | Wall-clock target |
|---|---:|---:|
| Custom | 25 generic cases × two modes = 50 records, plus matching model-specific cases | 5–7 minutes |
| BFCL | 50 fixed BFCL v3 smoke cases across 8 categories | 3–5 minutes |
| tau2 | 9 fixed tasks, 3 per domain, one trial | 10–12 minutes |

`manual`, `pr`, and `nightly` currently extend the same `qualification`
profile. There is no exhaustive profile. Subset results must not be presented as
official full-benchmark scores.

## Inspect without sending requests

```bash
python3 -m benchmarks.tool_calling.e2e_verifier \
  --suite bfcl \
  --base-url http://127.0.0.1:8000/v1 \
  --model moonshotai/Kimi-K2.6 \
  --runtime dynamo-vllm \
  --output-dir /tmp/dynamo-e2e/bfcl \
  --dry-run
```

## Run a suite

```bash
python3 -m benchmarks.tool_calling.e2e_verifier \
  --suite custom \
  --profile qualification \
  --base-url http://127.0.0.1:8000/v1 \
  --model moonshotai/Kimi-K2.6 \
  --runtime dynamo-vllm \
  --request-contract-json '{"enabled":{"thinking":true},"disabled":{"thinking":false}}' \
  --output-dir /tmp/dynamo-e2e/custom
```

BFCL and tau2 use the multi-architecture NeMo Evaluator `26.03` images pinned
by OCI digest in `profiles.json`. The BFCL smoke selection pins exact case IDs
and validates them against the image before model requests begin. Its 50 cases
cover simple, multiple, parallel, parallel-multiple, irrelevance, and three
multi-turn behaviors; live, executable, and long-context categories are left
out of the PR smoke path. Parser and reasoning-marker behavior remains covered
by the Custom suite. Custom pins 25 `generic_cases` for every model and runs
each in streaming and nonstreaming mode. It then appends only the
`model_specific_cases` assigned to the automatically resolved profile. The
`customer_` prefix is provenance metadata and can occur in either group: eight
customer regressions are generic, while two Kimi K2 multi-turn regressions are
model-specific. A non-Kimi model therefore runs 25 cases/50 records; Kimi K2
runs 27 cases/54 records. `suite-result.json` records the resolved profile,
both groups and their counts, and a selection hash. Execution is incomplete
unless the detailed report exactly matches that resolved matrix. Additional
model-specific control-token and parser stress cases remain available as
diagnostic profiles. tau2 additionally requires a fixed user simulator through
`TAU2_SIMULATOR_MODEL`,
`TAU2_SIMULATOR_BASE_URL`, and `TAU2_SIMULATOR_API_KEY`.

Every invocation writes `suite-result.json`. `execution_status` describes
whether the evaluator ran successfully; `verdict` describes model behavior.
A behavioral failure therefore remains a completed execution rather than an
infrastructure error.
