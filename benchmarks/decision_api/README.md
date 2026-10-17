<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Decision API performance instrument

This external AIPerf plugin measures structured decision requests without treating their JSON responses as generated text. It does not change Dynamo APIs or inference behavior. The initial target is `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, using SGLang 0.5.19 with one tensor/pipeline rank, overlap scheduling disabled, and at most one running engine request. Results from this profile must not be described as future batched-serving capacity.

Accuracy evaluation, Decision Index campaigns, Kubernetes, replica scaling, and additional models are outside this instrument's initial campaign. CPU tests establish instrument behavior; they do not establish GPU performance or fresh numerical parity.

This is a Dynamo-owned external-plugin prototype for [AIPerf issue #1536](https://github.com/ai-dynamo/aiperf/issues/1536), tracked with the [Decision API umbrella](https://github.com/ai-dynamo/dynamo/issues/15843); it does not modify upstream AIPerf core. Unit tests and synthetic HTTP/SSE integration run independently of Dynamo's serving APIs. Actual Dynamo `/v1/decisions`, `/v1/systemone`, and scoring `/generate` integration requires the unmerged serving implementation stack ending at [qualification PR #15857](https://github.com/ai-dynamo/dynamo/pull/15857). Until that stack lands, use its serving checkout rather than expecting those routes on `main`.

## Contracts

| Dialect | AIPerf endpoint plugin | HTTP route | Transport |
|---|---|---|---|
| `oai` | `decision_oai` | `/v1/decisions` | Non-streaming structured response |
| `systemone` | `decision_systemone` | `/v1/systemone` | Non-streaming structured response |
| `native_score` | `native_score` | `/generate` | SSE terminal scoring frame, zero generated tokens |

Phase 1 exposes only the OpenAI and Jev public contracts, without a format selector. The `/generate` adapter is an internal scoring baseline, not the SGLang-native Decisions wire protocol.

The `inputs_json` dataset loader preserves the supplied request, including typed choices and question ordering. The native adapter adds a fresh request-local cache salt for every actual send, including replayed dataset rows. No other native fields are changed. Decision JSON is never supplied as generated text. TTFT, inter-token latency, and output-token throughput are disabled by endpoint metadata. Both a malformed HTTP-200 response and an incomplete native SSE stream are errors.

Dynamo's server request identifier is not assumed to equal the client's measurement identifier. The instrument validates the server's matching request-ID headers and retains measurement correlation separately. Refusals remain refusals; missing usage remains unknown. API token/cache accounting is not physical GPU work.

The current Dynamo middleware can return conflicting `X-Request-ID` and `X-TypeSafe-Request-ID` values when the caller supplies `X-Request-ID`. The external `decision_http` transport omits that optional request header and uses `X-Decision-Measurement-ID` for benchmark correlation, leaving AIPerf's internal request identity intact. This tests the server-generated response-header path. It does not fix or qualify the supplied-header path, and it must not be presented as complete SDK/header conformance. No benchmark proxy or production API modification is involved.

## Isolated installation

Run these commands from the Dynamo checkout. Keep the serving virtual environment separate from the benchmark environment.

```bash
source dynamo/bin/activate
uv venv --python 3.12 /tmp/decision-aiperf-env
uv pip install --python /tmp/decision-aiperf-env/bin/python -e 'benchmarks/decision_api[test,tokenizer]'
export AIPERF_ENV=/tmp/decision-aiperf-env
export PATH="$AIPERF_ENV/bin:$PATH"
```

The dependency pins AIPerf **0.13.0**, official release commit `794f8bb75f8582f22e412d7e650fc71ca2a3d21a`. Preserve `uv pip freeze` output with campaign evidence. Workload manifests record tokenizer-library versions independently of engine and model revisions. If generating with a different tokenizer environment than the server, compare rendered input IDs, candidate IDs, and tokenizer/template artifact hashes before measurement.

## CPU validation

```bash
cd benchmarks/decision_api
"$AIPERF_ENV/bin/python" -m pytest -q --cov=dynamo_decision_perf --cov-fail-under=80
DECISION_PERF_INTEGRATION=1 "$AIPERF_ENV/bin/python" -m pytest tests/test_instrument_integration.py -q
```

The opt-in integration tests run real AIPerf against an owned loopback HTTP server with known delay, malformed success, explicit overload, and multi-frame native SSE. They check payload identity, per-send native cache isolation, request counts, rejection visibility, and absence of generated-token latency metrics. Unit tests separately cover typed values, predicate/score/mixed contracts, numerical reducers, errors, phase boundaries, unknown accounting, and audit rejection.

`tests/test_mocker_integration.py` additionally exercises a real Dynamo frontend and CPU mock worker using the local Qwen tokenizer. It needs the existing Dynamo development bindings, `etcd`, `nats-server`, and the cached model snapshot. It loads no model weights and uses no GPU. See that test's environment-variable gate for reproduction in the serving environment.

## Freeze deterministic workloads

Set `DECISION_MODEL_SNAPSHOT` to the existing local model snapshot; generation is offline and does not download weights. Output directories must be new.

```bash
"$AIPERF_ENV/bin/python" -m dynamo_decision_perf.workloads \
  --tokenizer "$DECISION_MODEL_SNAPSHOT" --output /tmp/decision-workloads \
  --model Qwen/Qwen3.8-27B --seed 17 --count 8
"$AIPERF_ENV/bin/python" -m dynamo_decision_perf.campaign \
  --workload-manifest /tmp/decision-workloads/manifest.json \
  --output /tmp/decision-screening-plan.json
```

The base is approximately 256 rendered tokens, one question, and eight choices. One-factor variations target 1,024/1,536 tokens, 4/16 questions, and 2/32 choices, plus predicate, score, and mixed requests. Manifests contain actual per-question prompt lengths, expanded token work, label IDs, prompt/payload hashes, and explicit unsupported findings. The tokenizer must verify distinct single-token label suffixes at the actual answer position. There is no truncation or substitution. Native multi-question latency is unsupported because one native call cannot represent the frontend's question fan-out.

For the pinned tokenizer, strict mixed-question and 32-choice/256-token shapes may be unsupported because their prompt overhead cannot meet the requested per-question range. Preserve those findings; a longer-input alternative is a separately named series, not a replacement baseline.

## Local serving and authorization

`python -m dynamo_decision_perf.serving` runs in the **serving** environment with the package source on `PYTHONPATH`. It owns only its launched processes and shuts them down on exit. Modes are `mocker`, `sglang` (Dynamo), and `native` (direct SGLang). Existing development bindings and the Dynamo source checkout must be importable. The launcher writes exact process commands, PIDs, readiness, and the GPU reservation start/deadline under a new log directory.

```bash
source dynamo/bin/activate
export PYTHONPATH="$PWD/benchmarks/decision_api/src:$PWD/components/src:$PWD/lib/bindings/python/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m dynamo_decision_perf.serving \
  --mode mocker --model-path "$DECISION_MODEL_SNAPSHOT" \
  --log-dir /tmp/decision-mocker --duration-seconds 300
```

GPU execution requires explicit operator approval before using `--approve-gpu-hours`. After approval, use `--mode sglang` or `--mode native`, one GPU, and a reservation of at most four GPU-hours. The serving-lifetime budget includes startup and warmup; do not reset it for every measurement. Direct SGLang and Dynamo native baselines must run sequentially on the same GPU, never concurrently. The launcher enforces SGLang 0.5.19, the pinned snapshot, serialized scheduling, 2,048-token context/total-token limits, disabled CUDA graphs, and static memory fraction 0.75.

Pin serving and AIPerf to disjoint available CPU sets with `taskset`. Record competing GPU/CPU activity and the serving/client affinity. The runner requires explicit client affinity for GPU runs; it cannot prove exclusivity by itself. Do not run independent experiments against the same server concurrently.

## Frozen execution profile

A CPU profile is a JSON object with `kind: "cpu_instrument"` and a `series_id`. A GPU profile must record all of the following before the runner will accept it:

| Field | Required evidence |
|---|---|
| `dynamo_commit`, `frontend_commit` | Exact source identities, including local modifications |
| `model_revision` | Pinned snapshot revision |
| `tokenizer_sha256`, `template_sha256` | Artifacts used by the serving renderer |
| `engine_version`, `precision` | Actual SGLang version and loaded dtype |
| `gpu`, `driver` | Device identity, GPU count, driver/runtime information |
| `context_limit` | `2048` |
| `scheduler` | `{"tp":1,"pp":1,"max_running_requests":1,"disable_overlap_schedule":true}` |
| `qualification_evidence` | Fresh smoke and numerical-parity artifacts |
| `series_id`, `cache_phase`, `cost_boundary` | Comparison identity, runtime warmup state, measured path |

Additional fields should include launch arguments, total-token and memory limits, graph configuration, backend metrics URLs, relevant process IDs, competing activity, and software dependency exports. Profile fields are operator-supplied evidence, not an automatic discovery proof. Any model, precision, template, scheduler, or dependency change starts a new series and requires another smoke/parity check. The tolerance is the existing qualification tolerance: absolute `5e-4`, relative `1e-3`; do not change it after observing results.

## One bounded run

The runner is dry-run by default: it freezes a new artifact directory and prints the command. `--execute` launches AIPerf. Use the actual loopback URL from the launcher's `ready.json`.

```bash
"$AIPERF_ENV/bin/python" -m dynamo_decision_perf.runner \
  --profile /tmp/decision-profile.json \
  --dataset /tmp/decision-workloads/base.oai.json \
  --dialect oai --url http://127.0.0.1:8000 \
  --output /tmp/decision-runs/serial-oai \
  --concurrency 1 --requests 8 --duration 300 --execute
```

For a GPU run add `--gpu --approved-gpu-hours HOURS --budget-started START --budget-deadline DEADLINE --client-cpus CPU_LIST`; reservation start/deadline come from the serving launcher's readiness record. Pass each owned frontend/worker PID using `--monitor-pid PID`. The runner checks the cumulative deadline against the approved hours and captures process CPU time, memory, affinity, AIPerf GPU telemetry, raw exports, and logs. Process lifetime is recorded separately from the profiling window.

For native scoring use `base.native_score.json` and `--dialect native_score`. The exact same input IDs, candidate IDs, precision, zero-decode parameters, and per-request cache-salt policy must be used for direct SGLang and Dynamo `/generate`.

## Campaign order and stop gates

Run configurations sequentially. Audit every point before advancing. The campaign module emits a schedule but deliberately does not launch an unattended GPU sweep or infer live safety observations.

1. Verify CPU instrumentation and fresh GPU schema/numerical parity.
2. Run the single-question serial baseline against direct SGLang `/generate`, Dynamo `/generate`, and all three public contracts. Store cold-start observations separately from warmed-runtime samples.
3. Run supported shape variations one factor at a time.
4. Screen concurrency 1, 2, 4, and 8 with at most four requests per concurrency slot. These are screening observations, not tail-latency evidence.
5. Select screened capacity only from audited, error-free points using `screen_capacity`. Supply the resulting screening report to `campaign --screening-report REPORT.json` to produce 25%, 50%, 75%, 100%, and 125% open-loop rate points. Each has 30 seconds of excluded warmup and 300 seconds of profiling.
6. Run a short overload interval followed by the 25% recovery rate. Confirm the server returns to its low-load latency regime and outstanding work drains before releasing the GPU reservation.

Stop escalation on OOM, schema/numerical failure, unbounded backlog, client saturation, missing evidence, or failure to recover. `can_continue` requires an audited run and explicit OOM/backlog/drain/client-saturation observations; unknown observations fail closed. An operator must monitor server logs and queue/in-flight metrics during each run and interrupt the owned benchmark/serving processes when a stop condition occurs. Capture existing preflight, first-branch, and fan-out Prometheus metrics separately; the generic runner does not infer backend metric addresses or automatically diagnose queue stability. Cancellation is not proof of physical GPU settlement.

Use one run per configuration initially. Before making small-delta claims, collect a three-run noise pilot for the same configuration; the pilot does not justify repeating every point. No individual measured run exceeds 30 minutes. All campaign stages, startup, warmup, and failed attempts count against the authorized reservation.

## Audit and report

```bash
"$AIPERF_ENV/bin/python" -m dynamo_decision_perf.report \
  --artifacts /tmp/decision-runs/serial-oai/aiperf \
  --metadata /tmp/decision-runs/serial-oai/audit_metadata.json \
  --workload /tmp/decision-workloads/base.metadata.json \
  --output /tmp/decision-reports/serial-oai
```

Repeat the artifacts/metadata/workload argument triple to build a multi-run report. Keep reports outside raw artifact directories. The analyzer requires an execution record bound to the manifest hash and parses raw exports using the pinned AIPerf models. It resolves controller phase boundaries, not a success-only latency span; where only aggregate microsecond timestamps are available, it requires the recorded UTC offset and labels the precision limitation. Offered requests remain unknown unless independently measured. Configured rates and request targets are not fabricated observations.

Reports retain HTTP errors, malformed success responses, refusals, rejections, timeouts, incomplete requests, unknown usage, and invalid evidence. They include valid requests/s and completed questions/s separately. Percentiles use nearest-rank estimates: p95 is suppressed below 100 samples and p99 below 1,000. No SLO-based goodput is invented. Paired cost comparisons require matching logical requests and frozen profile/series/cache identity; duplicated replay IDs are excluded from pairing unless separately correlated. Never subtract unrelated p95 values.

Full API minus pretokenized scoring represents additional frontend/serving cost, not pure routing overhead. Invalid runs remain visible but cannot establish performance gains. SVG latency/saturation charts are descriptive; screening sample sizes and unknown offered counts remain explicit.

## Artifact inventory

Each run stores `manifest.json` and its hash, an immutable workload copy/hash, source identities, `audit_metadata.json`, `command.json`, `benchmark_execution.json`, `aiperf.log`, CPU telemetry, and untouched AIPerf exports. Keep serving logs, readiness/command evidence, exact dependency exports, numerical smoke evidence, metric captures, and operator safety observations with the series. Reports contain machine-readable audits/summaries, a Markdown table, and SVG plots.

## References

- [AIPerf 0.13.0 release](https://github.com/ai-dynamo/aiperf/releases/tag/v0.13.0)
- [Pinned plugin interfaces](https://github.com/ai-dynamo/aiperf/blob/v0.13.0/docs/plugins/creating-your-first-plugin.md)
- [Pinned HTTP transport](https://github.com/ai-dynamo/aiperf/blob/v0.13.0/src/aiperf/transports/aiohttp_transport.py)
