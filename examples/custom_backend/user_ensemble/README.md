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
       +---- AsyncVisionEncoder ----> artifact ----> classifier ----+
                                      |                              |
                                      +---- NIXL descriptor ---------+----+
                                                                         v
                                                               remote dynamo.vllm
                                                                         |
                                                                  import artifact
                                                                         |
                                                                      decode
```

The frontend sends the ordinary OpenAI request to `UserEnsembleEngine`. The
ensemble encodes the image once. It passes the artifact directly to the local
classifier and publishes a NIXL transfer descriptor in `encoder_result`. The
request sent to the remote vLLM endpoint omits the raw media payload. The remote
worker imports the artifact and uses its decoder-specific adapter to construct
the vLLM prompt without loading or running another encoder.

The ensemble propagates the same Dynamo request context so cancellation reaches
the decoder. It requests one final-only decoder response to avoid forwarding a
nested token stream that the ensemble would immediately reassemble. The
classifier result is attached to `nvext.engine_data` on that terminal response.

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
DYN_EMBEDDING_TRANSFER_MODE=nixl-read \
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
