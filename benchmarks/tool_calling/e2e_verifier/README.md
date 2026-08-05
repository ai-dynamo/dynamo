# Dynamo E2E Verifier Contract

This package is the Dynamo-owned boundary between tool-calling correctness tests
and external deployment harnesses. Dynamo owns the test profiles, validators,
benchmark versions, scoring, and normalized result contract. An infrastructure
harness owns endpoint deployment, execution, aggregation, notifications, and
dashboard publication.

All normal profiles intentionally use bounded qualification subsets:

| Suite | Coverage | Wall-clock target |
|---|---:|---:|
| Custom | Existing generic/model-specific parser matrix, two modes | 5–7 minutes |
| BFCL | 200 stratified BFCL v3 cases | 8–10 minutes |
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
by OCI digest in `profiles.json`. tau2 additionally requires a fixed user
simulator through `TAU2_SIMULATOR_MODEL`, `TAU2_SIMULATOR_BASE_URL`, and
`TAU2_SIMULATOR_API_KEY`.

Every invocation writes `suite-result.json`. `execution_status` describes
whether the evaluator ran successfully; `verdict` describes model behavior.
A behavioral failure therefore remains a completed execution rather than an
infrastructure error.
