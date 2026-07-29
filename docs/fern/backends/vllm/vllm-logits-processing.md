---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Logits Processing
subtitle: Custom logits processors plug into vLLM through Dynamo's engine-loaded adapter and per-request SamplingParams.extra_args.
---

For general vLLM features and configuration, see the [Reference Guide](vllm-reference-guide.md).

---

Logits processors let you modify the next-token logits at every decoding step (e.g., to apply custom constraints or sampling transforms). Dynamo provides a backend-agnostic interface and an adapter for vLLM so you can plug in custom processors.

### How it works

- **Interface**: Implement `dynamo.logits_processing.BaseLogitsProcessor`, which defines `__call__(input_ids, logits)` and modifies `logits` in-place.
- **Shared spec layer**: The engine declares a backend-neutral `LogitsProcessorSpec` (see `dynamo.common.backend.engine`). The shared `logits_processors_for_request` helper owns the generation-stage gating (activate only on `AGGREGATED` / `DECODE`) and the per-request freshness policy.
- **vLLM mechanism**: Unlike TRT-LLM, vLLM cannot accept a live per-request callable. Its custom logits processors are *engine-loaded*: a class is registered at engine init and instantiated once, then called for every batch. Per-request activation rides on `SamplingParams.extra_args` (vLLM's `vllm_xargs`). Dynamo's adapter lives at `dynamo.vllm.logits_processing.adapter`.

### Quick test: HelloWorld processor

`DYN_ENABLE_TEST_LOGITS_PROCESSOR=1` is a built-in test hook (not a production processor loader) that forces the model to respond with "Hello world!". It verifies the callback path without modifying your model or engine code:

```bash
cd $DYNAMO_HOME/examples/backends/vllm
export DYN_ENABLE_TEST_LOGITS_PROCESSOR=1

# unified aggregated
./launch/agg.sh
```

Send a normal chat/completions request; the response should contain "Hello world!".

```{note}
- vLLM initializes the tokenizer by default in Dynamo, so (unlike TRT-LLM/SGLang) there is no `skip_tokenizer_init` flag to flip for this hook.
- Expected chat response contains "Hello world".
```

#### Disaggregated caveat

The quick test targets aggregated deployments. In disaggregated mode the prefill worker emits one token before decode resumes, and the test processor has per-request state. The unified backend skips the test hook on the prefill role (the shared generation-stage gating returns no entries there), but the decode-side output can still be affected by the prefill-produced leading token. Use aggregated mode to verify the wiring.

### How the unified backend wires this up

The unified vLLM engine threads logits processors through the shared spec layer in `dynamo.common.backend.engine` and the per-backend realizer at `dynamo.vllm.logits_processing.adapter`:

- `start()` registers the engine-loaded adapter (`DynamoVllmLogitsProcessor`) onto `engine_args.logits_processors` **before** building the engine config — but only when the env hook is on and the worker is a generation role (`AGGREGATED` / `DECODE`). Production paths leave `logits_processors` untouched. After the engine (and tokenizer) is up, it resolves a `LogitsProcessorSpec` once via `resolve_test_logits_processor_spec`, tokenizing `"Hello world!"` into a `ForcedTokenSequenceSpec` with the token IDs already resolved. `None` when the env var is off or on a non-generation role.
- `generate()` calls `logits_processors_for_request(spec, disaggregation_mode=...)` to get the per-request entry list (empty on PREFILL or when spec is `None`), then `activate_logits_processors(sampling_params, entries)` serializes the entries into `sampling_params.extra_args["dynamo_logits"]`.
- `DynamoVllmLogitsProcessor.new_req_logits_processor(params)` (called once per request by vLLM) reads `extra_args["dynamo_logits"]`, realizes a **fresh** per-request `ForcedSequenceLogitsProcessor`, and returns a request callable that applies it. Requests with no activation return `None`, so vLLM skips them.

The same shared layer hosts the TRT-LLM and SGLang slices; each backend translates the same `LogitsProcessorSpec` into its native shape. The public config-driven loader (when it lands) plugs in by resolving a `LogitsProcessorSpec` from CLI/config instead of from this env var; no engine code changes.

### Current limitations

- The env hook ships only `ForcedTokenSequenceSpec` (pre-resolved token IDs). Arbitrary Dynamo `BaseLogitsProcessor` instances and a public import-string/plugin loader are deferred follow-ups (see the design doc).
- `PythonProcessorSpec` (the TRT-LLM in-process escape hatch wrapping a live callable) is **not** serializable, so the vLLM adapter rejects it.
- Processors must modify the per-request 1-D logits vector in-place.
