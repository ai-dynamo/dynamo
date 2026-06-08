---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Logits Processing
---

For general TensorRT-LLM features and configuration, see the [Reference Guide](trtllm-reference-guide.md).

---

Logits processors let you modify the next-token logits at every decoding step (e.g., to apply custom constraints or sampling transforms). Dynamo provides a backend-agnostic interface and an adapter for TensorRT-LLM so you can plug in custom processors.

### How it works

- **Interface**: Implement `dynamo.logits_processing.BaseLogitsProcessor` which defines `__call__(input_ids, logits)` and modifies `logits` in-place.
- **TRT-LLM adapter**: Use `dynamo.trtllm.logits_processing.adapter.create_trtllm_adapters(...)` to convert Dynamo processors into TRT-LLM-compatible processors and assign them to `SamplingParams.logits_processor`.
- **Examples**: See example processors in `lib/bindings/python/src/dynamo/logits_processing/examples/` ([temperature](https://github.com/ai-dynamo/dynamo/tree/main/lib/bindings/python/src/dynamo/logits_processing/examples/temperature.py), [hello_world](https://github.com/ai-dynamo/dynamo/tree/main/lib/bindings/python/src/dynamo/logits_processing/examples/hello_world.py)).

### Quick test: HelloWorld processor

`DYNAMO_ENABLE_TEST_LOGITS_PROCESSOR=1` is a built-in test hook (not a production processor loader) that forces the model to respond with "Hello world!". It is useful to verify the callback path without modifying your model or engine code. It works on both the legacy and the unified TRT-LLM aggregated launchers:

```bash
cd $DYNAMO_HOME/examples/backends/trtllm
export DYNAMO_ENABLE_TEST_LOGITS_PROCESSOR=1

# legacy aggregated
./launch/agg.sh

# unified aggregated
./launch/agg.sh --unified
```

<Note>
- When enabled, Dynamo initializes the tokenizer so the HelloWorld processor can map text to token IDs.
- Expected chat response contains "Hello world".
</Note>

#### Disaggregated caveat

The quick test targets aggregated deployments. In disaggregated mode the prefill worker emits one token before decode resumes, and the test processor has per-request state that resets across the prefill/decode boundary. As a result the leading characters of the response can be duplicated or otherwise corrupted. The unified backend skips the test hook in the prefill role for this reason, but the decode-side output is still affected by the prefill-produced token. Use aggregated mode to verify the wiring.

For a public, user-defined processor loader (CLI/import-string), see the deferred follow-up in the design doc; this env hook intentionally stays test-focused.

### How the unified backend wires this up

The unified TRT-LLM engine threads logits processors through a shared, backend-agnostic spec entry layer in `dynamo.common.backend.engine` (next to the `LLMEngine` ABC) and a per-backend realizer at `dynamo.trtllm.logits_processing.adapter`:

- `from_args` forces `engine_args["skip_tokenizer_init"] = False` when the env hook is on **and** the worker is a generation role (AGGREGATED/DECODE), so an attached processor can't be starved of the tokenizer by an explicit user `skip_tokenizer_init=True`. PREFILL/ENCODE workers never attach the hook (see `generate()` below), so they are left alone and keep whatever `skip_tokenizer_init` they were configured with. The flag is backend-shaped (TRT-LLM dict; SGLang/vLLM set their own).
- `start()` resolves a `LogitsProcessorSpec` once via `resolve_test_logits_processor_spec(get_tokenizer)`, but only on generation roles — `logits_processor_spec()` returns `None` for PREFILL/ENCODE before any tokenizer access, so a prefill worker never resolves a spec or touches a tokenizer. The lambda is invoked lazily after the env check, so engines started with `skip_tokenizer_init=True` and the hook off don't crash. The spec carries a `ForcedTokenSequenceSpec` with token IDs already resolved at startup — no per-request tokenizer access. `None` when the env var is off.
- `generate()` calls `logits_processors_for_request(spec, disaggregation_mode=...)` to get the spec entry list (empty on non-generation workers such as PREFILL/ENCODE, or when spec is `None`), then `attach_logits_processors(sampling_params, entries)` to plug them into TRT-LLM. The TRT-LLM realizer materializes each spec entry into a fresh `BaseLogitsProcessor` (e.g. `ForcedSequenceLogitsProcessor`) and wraps in `TrtllmDynamoLogitsAdapter`.

The same shared layer will host the vLLM and SGLang slices when those land. vLLM loads a batch-level adapter class at engine init and activates it per request via `SamplingParams.extra_args`; SGLang flips `--enable-custom-logit-processor` at startup and passes a serialized class spec + `custom_params` per request. Each backend translates the same `LogitsProcessorSpec` differently. The public config-driven loader (when it lands) plugs in by resolving a `LogitsProcessorSpec` from CLI/config instead of from this env var; no engine code changes.

### Bring your own processor

Implement a processor by conforming to `BaseLogitsProcessor` and modify logits in-place. For example, temperature scaling:

```python
from typing import Sequence
import torch
from dynamo.logits_processing import BaseLogitsProcessor

class TemperatureProcessor(BaseLogitsProcessor):
    def __init__(self, temperature: float = 1.0):
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.temperature = temperature

    def __call__(self, input_ids: Sequence[int], logits: torch.Tensor):
        if self.temperature == 1.0:
            return
        logits.div_(self.temperature)
```

Wire it into TRT-LLM by adapting and attaching to `SamplingParams`:

```python
from dynamo.trtllm.logits_processing.adapter import create_trtllm_adapters
from dynamo.logits_processing.examples import TemperatureProcessor

processors = [TemperatureProcessor(temperature=0.7)]
sampling_params.logits_processor = create_trtllm_adapters(processors)
```

### Current limitations

- Per-request processing only (batch size must be 1); beam width > 1 is not supported.
- Processors must modify logits in-place and not return a new tensor.
- If your processor needs tokenization, ensure the tokenizer is initialized (do not skip tokenizer init).
