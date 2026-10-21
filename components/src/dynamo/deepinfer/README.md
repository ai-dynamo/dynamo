<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepInfer Dynamo backend

Experimental aggregated text adapter for the DeepInfer checkout used during
development. It implements Dynamo's unified `LLMEngine` contract, not an HTTP
proxy or a wrapper around the blocking offline `LLM.generate()` method. Dynamo
owns model registration, the OpenAI frontend, cancellation monitoring and drain.

## Scope and validation

CPU fake-engine tests exercise the adapter; neither actual DeepInfer inference
nor the native Dynamo/OpenAI path has been verified in this Windows environment.
This is not a production-ready replacement for the vLLM/SGLang integrations.

**Only DP=EP=CP=1 is accepted.** The adapter does not coordinate DeepInfer's
cross-rank collective scheduling. A model must fit this topology; the large
default DeepSeek checkpoint is not promised to fit or run on one device. If your
checkpoint requires multiple devices, this version is insufficient: an actual
multi-rank serving coordinator and GPU validation are still required.

| Capability | Implementation / boundary |
| --- | --- |
| Aggregated text token streaming | Low-level add/step/flush API |
| Concurrent requests | Shared scheduler; native calls serialized on one thread |
| Cancellation and generator close | Abort queued between native steps |
| temperature, top_p, top_k | Validated and mapped to DeepInfer |
| seed | Mapped to seq_seed; deterministic results additionally require DeepInfer's documented environment settings |
| max_tokens / ignore_eos | Mapped from stop_conditions; zero max_tokens rejected |
| Omitted max_tokens | Uses remaining registered context length |
| Stop strings and token visibility | Owned by Dynamo's token decoder, not implemented by this adapter for direct token clients |
| Completion usage | Native prompt_len / num_output_tokens |
| Health canary | One token through the regular generate path |
| Slow consumers | Bounded queue; explicit stream error and abort, never silently drop tokens |
| Logprobs (including zero), nondefault penalties, guided decoding, extra_args | Rejected |
| Multimodal / PD disaggregation / KV-aware routing / KVBM | Not implemented; KV event publication/indexing disabled |
| Tool and reasoning protocols, LoRA, operator integration | Not provided or validated |

The frontend and DeepInfer must use the same model/tokenizer directory. There is
no separate tokenizer override because the shared registration path uses the
model's tokenizer. Plain chat requires a compatible template in that directory;
model-specific tool/reasoning parsing is not configured here. Ordinary frontend
round-robin routing can distribute requests across independent workers, but is
not DeepInfer internal data parallelism.

## Build and install (Linux)

The local DeepInfer README requires Linux, Python 3.12, GCC >= 13 and, for its
NVIDIA path, SM100 hardware and CUDA 13. Ascend support in DeepInfer does not
establish support in this adapter; it has not been tested here. The commands
below are intended for a suitable Linux machine, not native Windows.

Use paths to copies of the two repositories on that machine. Start from a clean
Python 3.12 environment and follow both projects' native build prerequisites:

```bash
export DEEPINFER_ROOT=/workspace/DeepInfer-main
export DYNAMO_ROOT=/workspace/dynamo-main
python3.12 -m venv /workspace/deepinfer-env
source /workspace/deepinfer-env/bin/activate
export DEEPINFER_ACCELERATOR=nvidia
cd "$DEEPINFER_ROOT"
./build.sh
# DeepInfer's build links native libraries into its source package.
export PYTHONPATH="$DEEPINFER_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Build matching Dynamo runtime bindings using Dynamo's contribution guide,
# or install a compatible runtime wheel matching this checkout (1.6.0).
cd "$DYNAMO_ROOT"
python -m pip install -e . uvloop
python -m pip check
python -m dynamo.deepinfer --help
```

No published PyPI package named `deepinfer` is assumed. The adapter imports its
native modules only during `start()`, so parsing backend options does not load
Torch or DeepInfer. Installation/build commands above have not been executed
in this Windows session.

## Start

Start the usual Dynamo discovery services (for example the repository's
`deploy/docker-compose.yml` for etcd/NATS). Ensure frontend and backend share
the same discovery configuration. In the prepared environment:

```bash
# Terminal 1: use a local checkpoint compatible with single-device DeepInfer.
export CUDA_VISIBLE_DEVICES=0
python -m dynamo.deepinfer \
  --model /models/compatible-checkpoint \
  --served-model-name deepinfer-model \
  --namespace dynamo --component deepinfer --endpoint generate \
  --max-num-seqs 4 --gpu-memory-utilization 0.85

# Terminal 2: same environment/discovery configuration.
python -m dynamo.frontend --http-port 8000

# Terminal 3: once registration and readiness succeed.
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepinfer-model","prompt":"Hello","max_tokens":16,"stream":true}'
```

Do not use KV routing or prefill/decode mode with this adapter. A canary exercises
the engine; DeepInfer's `/profile` port is not a serving health check. Native
steps and model loading cannot be forcibly interrupted by Python: a wedged
CUDA/collective call requires an external process supervisor timeout. Shutdown
waits for the current native call before closing resources. If native construction
raises before returning an engine handle, cleanup of those allocations remains
DeepInfer's responsibility.

## CPU regression tests

From the Dynamo repository root, with Python and NumPy installed:

```bash
python components/src/dynamo/deepinfer/tests/test_llm_engine.py -v
python -m compileall -q components/src/dynamo/deepinfer
```

Tests load the actual adapter with temporarily isolated runtime stubs and a fake
engine. They need neither pytest nor CUDA, restore `sys.modules` after loading,
and cover concurrent admission, pooled-output ownership, flush-only terminals,
request validation, rejection, cancellation, queue overflow, startup failure
and cleanup. These are adapter unit tests, not native integration tests.
