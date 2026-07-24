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

`DYN_ENABLE_TEST_LOGITS_PROCESSOR=1` is a built-in test hook (not a production processor loader) that forces the model to respond with "Hello world!". It is useful to verify the callback path without modifying your model or engine code. It works with the TRT-LLM aggregated launcher:

```bash
cd $DYNAMO_HOME/examples/backends/trtllm
export DYN_ENABLE_TEST_LOGITS_PROCESSOR=1

./launch/agg.sh
```

<Note>
- When enabled, Dynamo initializes the tokenizer so the HelloWorld processor can map text to token IDs.
- Expected chat response contains "Hello world".
</Note>

#### Disaggregated caveat

The quick test targets aggregated deployments. In disaggregated mode the prefill worker emits one token before decode resumes, and the test processor has per-request state that does not carry across the prefill/decode boundary. As a result the leading characters of the response can be duplicated or otherwise corrupted. Use aggregated mode to verify the wiring.

For a public, user-defined processor loader (CLI/import-string), see the deferred follow-up in the design doc; this env hook intentionally stays test-focused.

### How TRT-LLM wires this up

The test hook lives in the TRT-LLM request handler path and adapts processors for the engine through `dynamo.trtllm.logits_processing.adapter`:

- **At worker startup** (`dynamo.trtllm.workers.llm_worker`), when the env hook is on, `engine_args["skip_tokenizer_init"]` is forced to `False`, overriding an explicit `skip_tokenizer_init=True`, so the processor is never starved of the tokenizer it needs to map text to token IDs.
- **Per request** (`dynamo.trtllm.request_handlers.handler_base`), when the hook is on, the handler builds a `HelloWorldLogitsProcessor(self.engine.llm.tokenizer)`, adapts it for TRT-LLM via `create_trtllm_adapters` (which wraps each `BaseLogitsProcessor` in `TrtllmDynamoLogitsAdapter`), and assigns the result to `sampling_params.logits_processor`.

vLLM and SGLang expose the same env hook through their own handler paths: vLLM loads a batch-level adapter class at engine init and activates it per request via `SamplingParams.extra_args` (see [vLLM Logits Processing](../vllm/vllm-logits-processing.md)); SGLang flips `--enable-custom-logit-processor` at startup and passes a serialized class spec + `custom_params` per request (see [SGLang Logits Processing](../sglang/sglang-logits-processing.md)). The public config-driven loader (when it lands) plugs in by resolving processors from CLI/config instead of from this env var; no engine code changes.

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
