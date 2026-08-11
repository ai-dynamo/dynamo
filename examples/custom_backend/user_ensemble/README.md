<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# User ensemble worker with remote vLLM

This example implements an aggregated model chain with public Dynamo worker and
client APIs. It runs the decoder as a stock `dynamo.vllm` worker instead of
embedding native vLLM inside `UserEnsembleEngine`.

```text
dynamo.frontend
       |
       v
UserEnsembleEngine
       |
       +---- AsyncVisionEncoder ----> classifier ----+
       |                                             |
       +---- client.generate(request, context) ------|----+
                                                     |    v
                                                     |  remote
                                                     |  dynamo.vllm
                                                     |    |
                                                     +----+
                                                       join
```

The frontend sends the ordinary OpenAI request to `UserEnsembleEngine`. The
ensemble encodes the image for its local classifier and concurrently forwards
the original preprocessed request to the remote vLLM endpoint. It propagates the
same Dynamo request context so cancellation reaches the decoder. The decoder's
delta chunks are accumulated into one terminal response, then the classifier
result is attached to `nvext.engine_data`.

The remote vLLM worker has its own custom encoder because Python encoder
artifacts are not JSON-serializable Dynamo request data. Consequently, this
variant encodes each image twice: once in the ensemble process for the
classifier and once in the vLLM process for decoding. The in-process variant
from pull request #12713 encodes once and shares the artifact objects with both
branches. This remote variant trades that duplicate work and an inter-process
request hop for a smaller API surface and an independently scalable stock vLLM
worker.

The supplied classifier deliberately returns `dummy-classification`. Replace
`DummyClassifier` with application logic that consumes the encoder artifact.
This example accepts exactly one image and one output choice and fails the whole
request if either branch fails.

## Run

From the repository root:

```bash
./examples/custom_backend/user_ensemble/launch.sh
```

The launcher starts three processes: the frontend, a `dynamo.vllm` worker on
`dynamo.remote-vllm.generate`, and `UserEnsembleEngine` on
`dynamo.backend.generate`. The decoder registers a private served-model alias so
frontend traffic for the public model is routed only through the ensemble.

Common overrides are:

```bash
DYN_MODEL=<model> \
DYN_WORKER_GPU=0 \
DYN_VLLM_GPU_MEMORY_UTILIZATION=0.8 \
DYN_DECODER_COMPONENT=remote-vllm \
./examples/custom_backend/user_ensemble/launch.sh
```

Then issue a non-streaming request:

```bash
curl -sS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "According to The Hitchhiker’s Guide to the Galaxy, The Answer to"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        {"type": "text", "text": " is? Reply with only the number."}
      ]
    }],
    "max_tokens": 8,
    "stream": false,
    "nvext": {"extra_fields": ["engine_data"]}
  }'
```

The generated text should contain `42`, and the response should contain:

```json
{
  "nvext": {
    "engine_data": {
      "ensemble": {
        "classifier": "dummy-classification"
      }
    }
  }
}
```
