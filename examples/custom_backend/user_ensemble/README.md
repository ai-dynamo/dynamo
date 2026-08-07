# User Ensemble Worker

This prototype shows that an application can implement an aggregated model
chain without changing `dynamo.vllm`:

```text
dynamo.frontend
       |
       v
UserEnsembleEngine
       |
       v
AsyncVisionEncoder -- shared artifacts --> classifier
       |                                  |
       +---- adapter --> EmbeddedVllmDecoder
                          |               |
                          +-- AsyncLLM ---+
                                          |
                                          v
                                  terminal response join
```

The frontend sends the ordinary OpenAI request. It does not hold or forward
encoder tensors. The user worker runs the encoder once, passes the same
in-process artifact objects to the classifier and decoder adapter, waits for
both branches, and returns one terminal response. Classifier data is attached
to `nvext.engine_data`; clients must request that optional field.

`EmbeddedVllmDecoder` is a library component, not another serving endpoint. It
owns native vLLM initialization, request translation, final-output
normalization, abort, shutdown, and registration metadata. The frontend still
listens on the only HTTP inference port, and `UserEnsembleEngine` remains the
only Dynamo backend endpoint for this chain.

The supplied classifier deliberately returns `dummy-classification`. Replace
`DummyClassifier` with application logic that consumes the encoder artifact.
This first version accepts exactly one image and one output choice and fails the
whole request if either branch fails.

## Run

From the repository root:

```bash
./examples/custom_backend/user_ensemble/launch.sh
```

When the decoder shares a GPU with a substantial encoder, reserve room for the
encoder with `DYN_VLLM_GPU_MEMORY_UTILIZATION` (for example, `0.4`). The default
is `0.8`, suitable only when the colocated encoder has enough remaining memory.

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

The current custom-encoder contract returns CPU-safe artifacts. A future
in-process contract can add explicitly owned CUDA artifacts; disaggregated
consumers will require handles and a transfer mechanism such as NIXL rather
than frontend-mediated tensors.
