<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepInfer Aggregated Serving

**Experimental.** Serve [DeepSeek's DeepInfer](https://github.com/deepseek-ai/DeepInfer)
through Dynamo's backend SDK. DeepInfer ships offline inference only, so the Dynamo
worker drives its engine loop on one dedicated thread and Dynamo's frontend supplies
the OpenAI-compatible API, discovery and request lifecycle.

## Scope

- Aggregated serving, NORMAL mode, and a single attention data-parallel rank.
- Multiple GPUs through DeepInfer's native context- and expert-parallel workers:
  `--context-parallel-size` GPUs are used, and it must be a multiple of
  `--expert-parallel-size`.
- No KV-aware routing, KV events, load metrics, prefill/decode disaggregation, or
  guided decoding yet.
- Requests using features DeepInfer cannot honour are rejected with an
  `InvalidArgument` error rather than served differently: penalties, `min_p`,
  `n > 1`, `logit_bias`, `min_tokens`, guided decoding, and `prompt_logprobs`.
  Stop strings and stop token ids are enforced by Dynamo's decoder, not the engine.

## Prerequisites

- NVIDIA SM100 GPUs. DeepInfer also lists Huawei Ascend 950, which this backend has
  not been tried on.
- Linux, GCC 13 or newer, Python 3.12, and CUDA 13.
- etcd and NATS reachable from the frontend and worker. See
  [runtime services](../../README.md#getting-started).
- A Dynamo build, as described in the repository
  [contribution guide](../../../docs/fern/pages/community/contributing/overview.md).

## Build DeepInfer from a pinned revision

DeepInfer is not on PyPI. Build the revision this backend was written against into
the same Python 3.12 environment as Dynamo:

```bash
git clone https://github.com/deepseek-ai/DeepInfer.git
git -C DeepInfer checkout 774ce244b87cc0eb8d4b94ab9f5de2ab22049f01
export DEEPINFER_ACCELERATOR=nvidia
(cd DeepInfer && ./build.sh)
```

Keep `DEEPINFER_ACCELERATOR=nvidia` exported wherever the worker runs. The default
checkpoint,
[`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017),
is downloaded into the Hugging Face cache on first start. Set `--model` to a local
checkpoint directory to skip the download.

## Launch

```bash
./launch/agg.sh --expert-parallel-size 8 --context-parallel-size 8
```

The script starts `dynamo.frontend` and `dynamo.deepinfer`. Pass any other
`dynamo.deepinfer` option through, for example `--max-model-len` or
`--gpu-memory-utilization`. Run `python3 -m dynamo.deepinfer --help` for the full list.
The smallest `--context-parallel-size` that fits the model has not been established
yet; see the validation record below.

## Smoke test

```bash
curl -s localhost:8000/v1/models

curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    "messages": [{"role": "user", "content": "Name three prime numbers."}],
    "max_tokens": 64
  }'

curl -sN localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    "messages": [{"role": "user", "content": "Count to ten."}],
    "max_tokens": 64,
    "stream": true
  }'
```

Stop a streaming request partway (Ctrl+C) and confirm the next request still succeeds
and the worker log shows the request being aborted. Then send SIGTERM to the launch
script and confirm the worker exits cleanly.

## Validation record

Record each run here. Report deployment failures separately from successful
inference, and include the real OpenAI responses.

| Date | Dynamo revision | DeepInfer revision | GPUs (count / type) | `--ep` / `--cp` | Result |
|------|-----------------|--------------------|---------------------|-----------------|--------|
| _not yet run on SM100 hardware_ | | | | | |

The CPU tests run without a GPU:

```bash
pytest -m unit components/src/dynamo/deepinfer/tests
```
