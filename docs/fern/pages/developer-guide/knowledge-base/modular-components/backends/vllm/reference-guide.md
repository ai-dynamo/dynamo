---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Reference Guide
subtitle: Configuration, arguments, and operational details for the vLLM backend
---

## Overview

The vLLM backend in Dynamo integrates [vLLM](https://github.com/vllm-project/vllm) engines into Dynamo's distributed runtime, enabling disaggregated serving, KV-aware routing, and request cancellation. Dynamo leverages vLLM's native KV cache events, NIXL-based transfer mechanisms, and metric reporting.

Dynamo vLLM uses vLLM's native argument parser — all vLLM engine arguments are passed through directly. Dynamo adds its own arguments for disaggregation mode, KV transfer, and prompt embeddings.

## Argument Reference

The vLLM backend accepts all upstream vLLM engine arguments plus Dynamo-specific arguments. The authoritative source is always the CLI:

```bash
python -m dynamo.vllm --help
```

The `--help` output is organized into the following groups:

- **Dynamo Runtime Options** — Namespace, discovery backend, request/event plane, endpoint types, tool/reasoning parsers, and custom chat templates. These are common across all Dynamo backends and use `DYN_*` env vars. See [Runtime Configuration](../../../../../reference/components/runtime-configuration.mdx) for the full field reference.
- **Dynamo vLLM Options** — Disaggregation mode, tokenizer selection, sleep mode, multimodal flags, vLLM-Omni pipeline configuration, headless mode, and ModelExpress. These use `DYN_VLLM_*` env vars. See [vLLM Configuration](../../../../../reference/backends/vllm-configuration.mdx) for the full field reference.
- **vLLM Engine Options** — All native vLLM arguments (`--model`, `--tensor-parallel-size`, `--kv-transfer-config`, `--kv-events-config`, `--enable-prefix-caching`, etc.). See the [vLLM serve args documentation](https://docs.vllm.ai/en/stable/configuration/serve_args.html).

### Tool and Reasoning Parsers

Use `--dyn-tool-call-parser` and `--dyn-reasoning-parser` to match the model's output format when the model emits tool calls and/or reasoning content. The current supported values are documented in [Tool Call Parsing (Dynamo)](../../../../../use-cases/tool-calling-and-reasoning/tool-call-parsing.mdx#supported-tool-call-parsers) and [Reasoning Parsing (Dynamo)](../../../../../use-cases/tool-calling-and-reasoning/reasoning-parsing.mdx#supported-reasoning-parsers).

To set the thinking mode used when a request omits an explicit control, pass
`--dyn-default-thinking-mode enabled|disabled` or set
`DYN_DEFAULT_THINKING_MODE`. Request-level thinking controls, including
adaptive thinking, take precedence. See
[Deployment-Level Thinking Default](../../../../../use-cases/tool-calling-and-reasoning/reasoning-parsing.mdx#deployment-level-thinking-default).

Set chat template defaults with `--default-chat-template-kwargs` on the
frontend, not on the worker. The frontend renders the chat template for workers
and sidecars, so the worker refuses the vLLM flag at startup. A `vllm-rs serve`
engine behind the vLLM sidecar applies its own flag only to its own HTTP server,
and the sidecar cannot detect it.

For reasoning models with structured output (`response_format`, JSON schema,
or required/named tool choice), configure both reasoning parsers on the worker:

```bash
python -m dynamo.vllm --model <model> \
  --reasoning-parser <vllm-parser> \
  --dyn-reasoning-parser <dynamo-parser>
```

The vLLM parser delays grammar enforcement until reasoning ends; the Dynamo
parser populates `reasoning_content`. Parser names can differ between registries.

### Priority Scheduling

vLLM engine-level request priority is controlled by the upstream vLLM
`--scheduling-policy priority` argument.

```bash
python -m dynamo.vllm \
    --model <model> \
    --scheduling-policy priority
```

Clients still send the Dynamo API value directly:
`nvext.agent_hints.priority`. Higher values mean higher priority at the Dynamo
API layer. Dynamo converts that value before passing it to vLLM, which uses a
different native priority polarity internally.

Do not negate `nvext.agent_hints.priority` in the client for vLLM. If you are
also using the router queue, configure the frontend-side
`--router-queue-threshold` separately; vLLM engine scheduling only applies
after a request reaches the worker.

For the cross-layer behavior, see
[Priority Scheduling](../../../../../use-cases/agents/priority-scheduling.mdx). For the upstream
flag definition, see the
[vLLM serve args documentation](https://docs.vllm.ai/en/stable/configuration/serve_args.html).

### Prompt Embeddings

Dynamo supports [vLLM prompt embeddings](https://docs.vllm.ai/en/stable/features/prompt_embeds.html) — pre-computed embeddings bypass tokenization in the Rust frontend and are decoded to tensors in the worker.

- Enable with `--enable-prompt-embeds` (disabled by default)
- Embeddings are sent as base64-encoded PyTorch tensors via the `prompt_embeds` field in the Completions API
- NATS must be configured with a 15MB max payload for large embeddings (already set in default deployments)

## Hashing Consistency for KV Events

When using KV-aware routing, ensure deterministic hashing across processes to avoid radix tree mismatches. Choose one of the following:

- Set `PYTHONHASHSEED=0` for all vLLM processes when relying on Python's built-in hashing for prefix caching.
- If your vLLM version supports it, configure a deterministic prefix caching algorithm:

```bash
vllm serve ... --enable-prefix-caching --prefix-caching-algo sha256
```

See the high-level notes in [Router Design](../../router/router-design.md#deterministic-event-ids) on deterministic event IDs.

## Graceful Shutdown

vLLM workers use Dynamo's graceful shutdown mechanism. When a `SIGTERM` or `SIGINT` is received:

1. **Discovery unregister**: The worker is removed from service discovery so no new requests are routed to it
2. **Grace period**: In-flight requests are allowed to complete (configurable via `DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS`, default 5s)
3. **Resource cleanup**: Engine resources and temporary files (Prometheus dirs, LoRA adapters) are released

All vLLM endpoints use `graceful_shutdown=True`, meaning they wait for in-flight requests to finish before exiting. An internal `VllmEngineMonitor` also checks engine health every 2 seconds and initiates shutdown if the engine becomes unresponsive.

For more details, see [Graceful Shutdown](../../../../../kubernetes/fault-tolerance/graceful-shutdown.md).

## Health Checks

Each worker type registers a health check with the Dynamo runtime. Set
`DYN_HEALTH_CHECK_ENABLED=true` to enable active checks.

| Worker Type | Health Check Strategy |
|------------|----------------------|
| Decode / Aggregated | Short generation request (`max_tokens=1`) using the model's BOS token |
| Prefill | Same payload structure as decode, adapted for prefill request format |
| [vLLM-Omni](../../../../../reference/backends/vllm-omni-configuration.mdx) (aggregated) | Native engine health check; see [vLLM-Omni Health Checks](../../../../../reference/observability/health-checks.mdx#vllm-omni-health-checks) for defaults and custom inference checks |
| vLLM-Omni ([stage worker](vllm-omni-disaggregated-serving.md)) | Short generation request using the model's BOS token |

An aggregated Omni worker serves the complete model pipeline through one Dynamo endpoint. A stage
worker runs a single stage in a disaggregated pipeline.

The runtime's HTTP health endpoints expose the resulting status to Kubernetes probes. Override the
payload with `DYN_HEALTH_CHECK_PAYLOAD`. See the
[Health Check Reference](../../../../../reference/observability/health-checks.mdx#active-canary-configuration)
for configuration and [Observability Architecture](../../../concepts/observability-architecture.md#active-worker-health-checks)
for the active health-check design.

## Request Cancellation

When a user cancels a request (e.g., by disconnecting from the frontend), the request is automatically cancelled across all workers, freeing compute resources.

| | Prefill | Decode |
|-|---------|--------|
| **Aggregated** | ✅ | ✅ |
| **Disaggregated** | ✅ | ✅ |

For more details, see the [Request Cancellation Architecture](../../../concepts/fault-tolerance/request-cancellation-architecture.md) documentation.

## Request Migration

Dynamo supports [request migration](../../../../../kubernetes/fault-tolerance/request-migration.md) to handle worker failures gracefully. When enabled, requests can be automatically migrated to healthy workers if a worker fails mid-generation. See the [Request Migration Architecture](../../../../../kubernetes/fault-tolerance/request-migration.md) documentation for configuration details.

## Experimental Decision Execution

### Runtime Qualification

**Experimental.** The causal runtime contract supports exact vLLM `0.32.0` and `0.33.0` releases, including local wheel build metadata. Capability publication verifies the installed cache-statistics source owners against the selected release's Git blobs; missing or modified sources disable optional causal decisions without disabling ordinary generation. The initial causal profile requires `max_num_seqs=1`, which serializes ordinary generation on that worker as well as decision execution; these results do not establish batched serving capacity. A generated, source-verified CPU contract establishes equivalent first-prefill capture, arithmetic and output projection for both releases. Recorded live-model evidence covers earlier source revisions on `0.33.0`; each changed integration revision requires fresh qualification. `0.32.0` remains pending fresh GPU/model qualification. The optional source-pinned native profiles require `0.33.0` and do not change the causal baseline's runtime allow-list.

### Routing and Lifecycle

Private causal scoring reuses the ordinary raw token-routing host for its text-only aggregated execution. It does not enter generation migration, replay, or generated-text decoding. Final dispatch validates the selected qualification epoch and current worker admission, then records a request-local receipt; the caller rejects an unchecked score before trusting it. Compatible membership changes preserve in-flight requests while their pinned worker remains admitted; qualification changes and withdrawal fence the earlier epoch. If an internal wrapper discards the whole registry, the receipt detects the missing validation but cannot prove that physical execution never began. This does not add disaggregated decision execution or change ordinary generation's retry policy.

Native head and native read profiles support round-robin, random, power-of-two choices, least-loaded and device-aware weighted routing. They do not support KV-aware or direct routing. This restriction is independent of the causal profile's reuse of the ordinary raw token-routing host.

### Availability and Recovery

Decision state distinguishes structural resource presence, supported qualification and live worker readiness. A qualified, preparable route with live workers is ready before its first request initializes a lazy resource. Deterministic tokenizer parse or template errors, including invalid UTF-8 and invalid I/O arguments, disable token scoring for that WorkerSet and return unsupported 400; correct the configuration and rebuild the WorkerSet to clear that cached failure. Other typed frontend-local I/O failures return retryable 503 and permit a later admitted initialization attempt after a one-second backoff. Fixing the underlying file or I/O condition can therefore recover on the same WorkerSet, including after same-checksum worker re-registration, without resetting its qualification epoch. The pinned TikToken loader discards typed causes, so errors at that opaque load boundary conservatively follow the retryable policy even when their underlying cause might be deterministic. No background retry is scheduled. Other usable model surfaces remain available. While qualification remains known, empty or closed worker-membership views make serving unavailable without removing the qualified HTTP route or reclassifying its resources as a prefill-only set. Materialization is owned by the resource cell, not cached in a second readiness flag; readiness inspection does not wait behind initialization.

Causal and native execution share the pinned default-scheduler qualification policy. Custom schedulers, mismatched synchronous/asynchronous scheduler classes, and unresolved scheduler modes fail closed. FPM and benchmark modes can inject Dynamo's `InstrumentedScheduler`, which is outside this qualification; their startup logs identify that injection. Use the pinned default scheduler for the initial decision profile. Successful workers must supply integer cache-read counters within the prompt-token count; missing, boolean, negative, or excessive counters produce a typed backend-protocol error rather than a successful response. The Rust collector owns causal frame validation; native adapters validate each native child. These checks cannot distinguish a measured zero from the upstream output processor's default zero. Measurement provenance therefore depends on the pinned standard scheduler capturing cache statistics before first execution and attaching them to the first successful output, not on integer validation alone. CPU source-equivalence and scheduler/output-processor tests qualify this dependency; a different scheduler or runtime needs fresh provenance checks.

### Validation and Reference Boundaries

The Rust and Python wire tests consume the same [native bounds fixture](https://github.com/ai-dynamo/dynamo/blob/main/tests/fixtures/decisions/native-bounds-v1.json) contract for question, candidate, input-token, tokenization-work, and encoding-byte limits. Both encoding checks measure a compact wrapped request, independently; serde and Python floating-point spellings need not be byte-identical at the boundary. This wire-size check does not change JSON formatting used in model prompts. The sibling `native-question-v1.json` is an exact valid request, not a container for extra fields. CPU fixtures exercise these contracts without claiming GPU execution or model quality.

Native reference fixtures execute hash-verified, AST-selected strategy and reducer definitions with stand-in request models, renderer, tokenizer and sampling parameters. They establish CPU logic parity for Dynamo's canonical System One question representation only, not unmodified upstream request-validation, production rendering, endpoint or GPU parity. Both public API dialects use that shared execution representation. Upstream vLLM's OpenAI Decisions adapter renders different questions; its unmodified endpoint responses are not the numerical reference for these native profiles. These adapters call private upstream strategy and reducer methods. Every vLLM upgrade requires a behavioral re-audit of those adapters and their canonical input/rendering paths, not only refreshed source hashes.

Diffusion read profiles use per-question canvas reads, matching the pinned upstream HTTP request default `joint=false`. They do not expose upstream's optional `joint=true` execution mode. Comparing probabilities requires the same execution mode, normalized questions, prompt and model configuration.

### Execution Bounds

`DYN_SYSTEMONE_MAX_INPUT_TOKENS` bounds cumulative rendered input across question executions; it is not a tokenization-work setting. Causal preflight derives its separate work budget as the input-token limit multiplied by the model's maximum candidate count plus two. Native preflight uses that derived budget capped at 1,048,576 units. These are profile-specific CPU-work bounds, not equivalent GPU-token measurements. Native success frames carry the existing `finish_reason: "stop"` completion marker so a configured backend admission gate releases waiting capacity on the result even when stream EOF is delayed; request capacity remains held until stream termination. The gate counts network requests, not native child sequences, and its limits are independent of the engine's own admission checks.

Native adapters serialize their private tokenizer work with one guard and offload that work from the event loop. Head profiles render and tokenize within that guarded CPU preflight. Read profiles tokenize candidate labels under the guard, but render prompts through the shared asynchronous `OnlineRenderer` outside it. Read rendered-token work is charged after rendering; the earlier byte bound limits input size but is not a pre-tokenization token-work bound. This guard does not serialize admitted engine execution. Cancellation drains any started CPU preflight before releasing the guard. When engine completion and context cancellation are both ready, the completed operation takes precedence.

### Startup Qualification

Native head workers verify an engine-only pooling read before executing the strict decision canary and advertising the optional capability. Native read workers prepare a fixed text request before advertisement; this CPU-only check does not execute the model or qualify its outputs. Adapter construction, including the engine capability query, preparation, and, for heads, the engine-only read and at most two strict canary attempts share one engine health-check timeout, with the existing 30-second default when that timeout is disabled; cancellation still drains owned cleanup and can exceed that deadline. Only explicitly transient strict-canary errors receive one retry. Exhausted transient attempts or deadline expiry leave decisions unadvertised without a failure marker; a later qualification attempt requires worker restart or reinitialization.

Stage determines the failure contract. A `NativeRuntimeError` during adapter construction denotes unsupported version, source or effective-runtime configuration and leaves decisions unadvertised. Failure to obtain an isolated tokenizer, whether copying raises or aliases the original tokenizer, is construction-time incompatibility and does not publish a failure marker. After adapter construction, deterministic preparation or canary-contract errors, including a `NativeRuntimeError`, publish a persistent qualification-failure marker; no engine execution is required for a preparation-contract failure. Unexpected initialization failures also publish this marker. Engine shutdown and caller cancellation propagate. A blocked cohort returns HTTP 400 with `model_qualification_failed` from both public contracts, distinct from transient rollout HTTP 503. Operator logs identify the blocking workers; public errors do not expose internal worker identifiers. Restart and requalify the failed worker after correcting its configuration or runtime failure. An unknown or ineligible profile does not publish this marker.

Retain the model-specific pooling health probe only after its engine-only read has returned a valid terminal. Preparation alone does not establish a usable probe. A later decision-decoding failure can leave this verified probe available for ordinary engine liveness; an unverified probe does not replace the worker's default health payload. Startup verification and periodic probes share the same native read and cancellation-cleanup path without decision decoding. Liveness bypasses the decision-child semaphore, so a full decision admission budget alone cannot block a probe from reaching the engine. The engine can still queue or reject that read; this bypass does not reserve GPU capacity or guarantee health-check latency under engine saturation. Every real decision still validates its exact executor result; a successful liveness probe is not decision-output or GPU-settlement qualification.

D3 decoder profiles accept both enabled and disabled chunked prefill under the pinned default-scheduler policy. This follows the pinned runtime's decoder LAST-pooling support; it does not qualify custom schedulers or other pooling modes.

### Native Template Policy

The optional native profiles do not forward Dynamo's `--custom-jinja-template` override into their rendering context. Their `OnlineRenderer` and `ReadContext` use `chat_template=None` and empty default template arguments. Next-token read profiles derive their assistant suffix from the effective engine tokenizer's chat template with thinking disabled; D3 also uses that tokenizer's template when preparing its prompt. Engine tokenizer overrides can change this effective tokenizer. Clef joint-schema profiles construct their trained framing directly. An ordinary-serving template override therefore does not customize these native prompts or establish numerical parity with an upstream server configured with that override.

Use the checkpoint template and trained framing qualified for the selected native profile. A missing or incompatible template follows the startup stage where it is detected: construction-time incompatibility leaves decisions unadvertised, while a deterministic preparation-contract failure publishes the qualification-failure marker described above. The override is not a workaround for native-template incompatibility. Supporting a different native template requires a separate rendering, label/tail and model qualification; it is not part of this profile's compatibility guarantee.

### Error Handling

Native request validation retains client-error classification, malformed executor contracts return 502, and unexpected internal exceptions return a sanitized 500. The pinned upstream `EngineGenerateError` and `RetryableRequestError` return recoverable request-scoped 503 responses without inhibiting the shared worker; ordinary serving remains available. A base `GenerationError` remains a 500. `EngineDeadError` becomes an engine-shutdown error (503); after owned children have been cancelled and drained, factory-created adapters immediately request runtime shutdown. Engine shutdown during the startup canary propagates instead of publishing a failed profile or registering a worker on the shutting-down runtime. This withdraws the runtime but does not duplicate the ordinary handler's hard process exit, guarantee delivery of an error response during shutdown, or establish physical GPU abort settlement. The existing engine monitor remains responsible for process-level failure handling. Hash-verified CPU tests exercise the actual upstream exception definitions without installing or running an inference engine.

### Model-Specific Input Semantics

Native execution flattens OpenAI message-list input into text: join ordered text parts with one newline and ordered messages with two newlines. OpenAI string input remains unchanged. This input-only rule also appears in upstream's OpenAI Decisions adapter; it does not make the question prompts equivalent. Normalization is limited to the native transport; Jev structured state retains its original shape, and the causal frontend prompt contract preserves message and part boundaries.

Both public dialects intentionally render native questions through the same canonical System One representation, with typed values retained separately for public response projection. OpenAI wire-schema compatibility does not promise native prompt or probability parity with an unmodified upstream vLLM `POST /v1/decisions` request. The [pinned upstream OpenAI adapter](https://github.com/vllm-project/vllm/blob/4ba787ec9959e566d3b587a796dff07673324da5/vllm/entrypoints/openai/decisions/adapters.py) differs as follows:

| Question | Dynamo Native Execution for Both Dialects | Upstream vLLM OpenAI Decisions |
| --- | --- | --- |
| Predicate | System One `noul` question; canonical candidates are `true`, then `false`, with model-specific ordering and projection back to that order | Choice question with `false`, then `true`, labeled A/B; returns the second probability |
| Choice | Raw string option names when all values are strings; JSON spelling when boolean candidates are present, preserving distinctions such as `true` and `"true"` | JSON spelling for every option value, including quotes around strings |
| Score | Combine an available level label and description into the option name; execute a System One score question | Keep label and description separate; execute a choice question |

Hash-verified CPU fixtures assert these deliberate differences. Numerical comparison requires the same canonical questions, option order, ordinal IDs, normalized input, execution mode and model configuration in the System One strategy reference. Comparing the same public OpenAI request against upstream's OpenAI endpoint does not meet that condition. This shared representation avoids introducing a second native question-execution contract; it does not establish hosted OpenAI or Jev model behavior or calibration parity.

Clef joint-schema profiles use the pinned upstream strategy's state-prefix truncation after reserving its fixed prompt and question schema within the trained/runtime context limit. State tokenization is charged against Dynamo's preflight work budget before truncation, so Dynamo can reject an input that an upstream server would truncate and accept. Dynamo intentionally replaces caller question IDs with ordinal IDs (`question-0`, `question-1`, and so on) before this strategy runs. These canonical IDs appear in the Clef prompt's `ID` fields and, when instructions are blank, in its instruction text through the upstream `instructions or id` fallback. Caller IDs remain response-projection metadata, not prompt content. This differs from upstream native requests that render caller-supplied IDs. Hash-verified CPU reference fixtures explicitly exercise the canonical IDs and blank-instruction fallback. Live probability parity requires identical normalized question IDs in the reference inputs; these fixtures do not establish GPU/model qualification or parity with upstream endpoint requests that retain caller IDs. Causal and D3 profiles instead reject their over-length inputs. API compatibility does not imply that every profile preserves the entire original state; workloads relying on a long state need separate truncation-aware quality qualification.

For SGLang serial-cohort policy, see [Decision Rollout Compatibility](../sglang/reference-guide.md#decision-rollout-compatibility).

## See Also

- **[Examples](../../../../../recipes/cli-templates/vllm.mdx)**: Local deployment launch scripts
- **[vLLM README](overview.md)**: Quick start and feature overview
- **[Observability](observability.md)**: Metrics and monitoring setup
- **[Configuration and Tuning](../../router/configuration-and-tuning.md)**: KV-aware routing configuration
- **[Fault Tolerance](../../../../../kubernetes/fault-tolerance/introduction.md)**: Request migration, cancellation, and graceful shutdown
