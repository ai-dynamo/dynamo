<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# BFCL v4: GLM-5.2 native tool calling

This harness evaluates the same served model, `zai-org/GLM-5.2`, through all four
campaign variants:

- `dynamo-vllm`
- `vllm-serve`
- `dynamo-sglang`
- `sglang-serve`

BFCL is pinned to Gorilla commit
`6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`. `scripts/bootstrap.sh` clones that
revision and applies `patches/0001-glm52-openai-chat-completions.patch`.

## Request path

The patch registers `zai-org/GLM-5.2-FC` as an API-inference model backed by
`GLM52OpenAIChatCompletionsHandler`. It deliberately does not use BFCL's
`OSSHandler`:

```text
BFCL test messages + function schemas
                 |
                 v
OpenAI client.chat.completions.create(messages=..., tools=...)
                 |
                 v
POST /v1/chat/completions
                 |
                 v
choices[0].message.tool_calls[]
                 |
                 v
BFCL native FC decoder and evaluator
```

This avoids the OSS prompting path, which renders a prompt locally and sends it to
`/v1/completions`. `scripts/validate_install.py` asserts the handler class, request
shape, and structured `tool_calls` decoding without contacting a server.

## Bootstrap

```bash
cd benchmarks/glm52-nscale/eval/bfcl
./scripts/bootstrap.sh
```

The checkout lives under `.cache/gorilla` and the virtual environment under
`.venv` by default. Override them with `BFCL_CHECKOUT_DIR` and `BFCL_VENV_DIR`.
Importable campaign runs require the exact committed 141-package environment and
the recorded Python version; `BFCL_BOOTSTRAP_PYTHON` may select the interpreter,
but the environment lock rejects any drift.
The bootstrap is idempotent and refuses an unexpected dirty or wrong-revision
checkout.

## Endpoint setup

The deployment lifecycle remains separate from this harness. On the evaluation
runner, point the harness at the active stack's cluster-DNS API root, including
`/v1` (the exact four mappings are in `../README.md`):

```bash
export GLM52_OPENAI_BASE_URL=http://glm52-dynamo-vllm-frontend:8000/v1
export GLM52_OPENAI_API_KEY=EMPTY
```

The preflight requires `/v1/models` to advertise exactly `zai-org/GLM-5.2`. An
authenticated endpoint can also use `GLM52_OPENAI_DEFAULT_HEADERS` as a JSON object.
Secrets are not written to metadata; endpoint URLs have userinfo and query strings
removed.

`GLM52_OPENAI_EXTRA_BODY` is diagnostics-only. Full campaign validation requires
it to be unset, so a run with any extra body cannot be imported as a baseline
result. `GLM52_OPENAI_DEFAULT_HEADERS` is also forbidden for full runs because
secret-sensitive routing or template overrides cannot be compared safely.

Every Chat Completions request is capped at 64,000 output tokens through
`GLM52_OPENAI_MAX_TOKENS`. The campaign wrapper derives it from
`BFCL_MAX_TOKENS` and records the value in immutable run metadata.

## Smoke run

The smoke set has two simple calls plus parallel, irrelevance, and multi-turn cases.
It verifies structured function calling before a full, expensive run.

Run each smoke through `../run-guarded.sh` with phase `validation`, following the
exact command in `../../RUNBOOK.md`; direct script invocation is only a local
harness diagnostic and is not importable campaign evidence.

Generation and evaluation can be resumed separately with an explicit run directory:

```bash
run_dir=$PWD/outputs/dynamo-vllm/smoke-validation-manual
./scripts/smoke-generate.sh dynamo-vllm "$run_dir"
./scripts/smoke-evaluate.sh dynamo-vllm "$run_dir"
```

Set `BFCL_ALLOW_OVERWRITE=1` only when intentionally regenerating existing case IDs.

## Full BFCL v4 run

The default `all_scoring` collection runs non-live, live, multi-turn, web-search,
and memory categories. FC models correctly omit the non-scoring format-sensitivity
suite. At the pinned Gorilla revision this is exactly 5,106 scored case IDs. The
generator also executes 111 memory prerequisite case IDs, so a complete raw result
population contains 5,217 unique IDs.

BFCL's web-search categories execute real SerpAPI queries. A paid/adequately
provisioned key is required:

```bash
export SERPAPI_API_KEY=...
# Deploy one fresh stack, then run the guarded command in ../../RUNBOOK.md.
```

BFCL contributes eight independent full cells. For each framework the required
capture order is `ab` Dynamo, `ab` native, `ba` native, `ba` Dynamo, with teardown
and a fresh deployment between every cell.

`run-full.sh` rejects a `BFCL_CATEGORIES` override so a partial diagnostic cannot
be mislabeled as the full campaign. `full-generate.sh` and `full-evaluate.sh` remain
available for explicitly labeled partial diagnostics.

Upstream uses `SERPAPI_API_KEY` from the environment and retries HTTP 429 responses
with exponential backoff. Confirm quota before starting all four runs. Missing keys
are rejected before generation. `BFCL_CATEGORIES` can select a comma-separated
subset for diagnosis, but anything other than `all_scoring` is not the requested
full BFCL v4 campaign and must be labeled as partial.

Defaults are temperature `0`, a 64,000-token output cap, and 16 request threads, matching the deployment's
concurrency cap. Keep them identical across comparisons:

```bash
export BFCL_TEMPERATURE=0
export BFCL_MAX_TOKENS=64000
export BFCL_NUM_THREADS=16
```

## Evidence and outputs

Each run is isolated under `outputs/<variant>/<mode>-<phase>-<UTC timestamp>/` unless
`BFCL_ARTIFACT_ROOT` or an explicit run directory is supplied:

```text
metadata.json                 endpoint-safe run identity and patch hash
expected-ids.json             exact pinned generated/scored ID manifest and hashes
endpoint-models.json          captured /v1/models response
environment-lock.json         constraints/freeze/Python identity
environment.freeze.txt        canonical exact evaluator environment
campaign_source (metadata)    exact campaign.env/eval tree identity
runtime-binding.json          validated serving deployment identity
runtime-continuity.json       guarded pre/post deployment attestation
logs/generate.log             exact command and generation output
logs/evaluate.log             exact command and evaluator output
logs/validate-*.log           structural completion checks
result/                       raw BFCL per-case responses and inference logs
score/                        upstream per-case failures and leaderboard CSVs
summary.json                  compact category pass/fail/token/error summary
failures.jsonl                all failed cases with BFCL error details
generation-validation.json    generation population and inference-error status
complete-validation.json      generation plus score population status
```

Generation metadata is immutable once a run directory exists. A resumed generation
must match it, and resumed evaluation reads the model and category selection from
that metadata rather than from the current shell environment. Full validation
requires all 5,217 generated IDs exactly once, no recorded inference exceptions,
all 22 score files, and score headers totaling exactly 5,106 cases. Missing,
duplicate, unexpected, malformed, or inference-error results make the scripts exit
nonzero; ordinary benchmark failures remain valid scored outcomes.

`outputs/` is gitignored because full raw trajectories are large. Preserve it on the
campaign artifact volume, then import all eight phase-qualified evidence bundles
through `report/import_result.py`; do not copy or hand-edit aggregate rows.

For a fair paired comparison, use the same BFCL commit, patch hash, categories,
temperature, thread count, and `GLM52_OPENAI_EXTRA_BODY`. Compare category scores
and paired case IDs, not only the BFCL aggregate, because BFCL's overall score uses
category weighting.

## Failure triage

- A 404 for `/chat/completions` indicates the URL is not an OpenAI `/v1` root.
- A missing advertised model indicates a served-name mismatch; do not silently
  change the BFCL model config between stacks.
- `content: null` without `tool_calls` is an inference/deployment failure, not a
  BFCL parser success. Inspect reasoning-parser and tool-call-parser settings.
- Tool-call JSON syntax failures appear in `failures.jsonl`; compare the same case
  across paired deployments before changing generation behavior.
- A full run with web-search errors or SerpAPI 429s is not comparable. Re-run the
  affected IDs under the same endpoint configuration and document the retry.
