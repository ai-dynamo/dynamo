---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Logits Processing
---

For general SGLang features and configuration, see the [Reference Guide](sglang-reference-guide.md).

---

Logits processors let you modify the next-token logits at every decoding step (e.g., to apply custom constraints or sampling transforms). Dynamo provides a backend-agnostic interface and an adapter for SGLang so you can plug in custom processors.

### How it works

- **Interface**: Implement `dynamo.logits_processing.BaseLogitsProcessor`, which defines `__call__(input_ids, logits)` and modifies `logits` in-place.
- **Shared spec layer**: The engine declares a backend-neutral `LogitsProcessorSpec` (see `dynamo.common.backend.engine`). The shared `logits_processors_for_request` helper owns the generation-stage gating (activate only on `AGGREGATED` / `DECODE`) and the per-request freshness policy.
- **SGLang mechanism**: SGLang gates the feature behind the server flag `--enable-custom-logit-processor`, takes a *serialized* processor class as a top-level `async_generate` argument (`custom_logit_processor`), and hands that processor a **batch** logits tensor plus a `custom_param_list` (one `sampling_params["custom_params"]` dict per request in the batch). Dynamo's adapter lives at `dynamo.sglang.logits_processing.adapter`.

### Quick test: HelloWorld processor

`DYN_ENABLE_TEST_LOGITS_PROCESSOR=1` is a built-in test hook (not a production processor loader) that forces the model to respond with "Hello world!". It verifies the callback path without modifying your model or engine code:

```bash
cd $DYNAMO_HOME/examples/backends/sglang
export DYN_ENABLE_TEST_LOGITS_PROCESSOR=1

# unified aggregated
./launch/agg.sh
```

Send a normal chat/completions request; the response should contain "Hello world!".

```{note}
- When enabled on a generation worker, Dynamo flips `--enable-custom-logit-processor` on and forces `skip_tokenizer_init=False` so the hook can resolve "Hello world!" to token IDs at startup.
- Expected chat response contains "Hello world".
```

#### Disaggregated caveat

The quick test targets aggregated deployments. In disaggregated mode the prefill worker emits one token before decode resumes. The unified backend skips the test hook on the prefill role (the shared generation-stage gating returns no entries there), but the decode-side output can still be affected by the prefill-produced leading token. Use aggregated mode to verify the wiring.

### How the unified backend wires this up

The unified SGLang engine threads logits processors through the shared spec layer in `dynamo.common.backend.engine` and the per-backend realizer at `dynamo.sglang.logits_processing.adapter`:

- `from_args()` sets `server_args.enable_custom_logit_processor = True` and `server_args.skip_tokenizer_init = False` when the env hook is on **and** the worker is a generation role — after user overrides, so an explicit `skip_tokenizer_init=True` can't starve the hook. PREFILL keeps its configured flags.
- `start()` resolves a `LogitsProcessorSpec` once via `resolve_test_logits_processor_spec`, tokenizing `"Hello world!"` into a `ForcedTokenSequenceSpec`. `None` when the env var is off or on a non-generation role.
- `generate()` calls `logits_processors_for_request(spec, disaggregation_mode=...)`, then `activate_logits_processors(sampling_params, entries, request_uid=...)`, which stashes the serialized entries into `sampling_params["custom_params"]` and returns the `custom_logit_processor` kwarg for `async_generate`.
- `DynamoSglangLogitProcessor.__call__(logits, custom_param_list)` maps each batch row to its request's serialized entries and applies a per-request processor to that row.

#### Per-request state

SGLang's callback does not pass the per-request generated-token position, and `custom_param_list` is static across decode steps. So the env hook supports only `ForcedTokenSequenceSpec`, whose realized `ForcedSequenceLogitsProcessor` advances a purely internal counter (it ignores `input_ids`). To make that counter advance, the adapter keeps per-request processor state keyed by a request UID (`context.id()`) injected into `custom_params`, relying on SGLang caching a single `DynamoSglangLogitProcessor` instance (by its serialized string) and reusing it across decode steps. Because SGLang expands `n > 1` into multiple batch rows that share one request's `custom_params`, the hook forces `n = 1` while it is active (a forced sequence has no meaningful `n > 1`). Arbitrary `BaseLogitsProcessor` instances that need the real token history are deliberately out of scope until SGLang exposes that state at this callback.

The same shared layer hosts the TRT-LLM and vLLM slices; each backend translates the same `LogitsProcessorSpec` into its native shape. The public config-driven loader (when it lands) plugs in by resolving a `LogitsProcessorSpec` from CLI/config instead of from this env var; no engine code changes.

### Current limitations

- The env hook ships only `ForcedTokenSequenceSpec` (pre-resolved token IDs). Arbitrary Dynamo `BaseLogitsProcessor` instances and a public import-string/plugin loader are deferred follow-ups (see the design doc).
- `PythonProcessorSpec` (the TRT-LLM in-process escape hatch wrapping a live callable) is **not** serializable, so the SGLang adapter rejects it.
- **Known issue — unbounded state map:** the per-request state map is not pruned (no completion signal reaches the callback), so it grows for the life of the worker process while the hook is active. This is acceptable for the env-gated smoke hook (a test surface), but is **not** a production processor surface.
- **`n` is forced to 1** while the hook is active: SGLang expands `n > 1` into batch rows that share one request's `custom_params`, which would collide the per-request processor state. A forced "Hello world!" has no meaningful `n > 1`.
