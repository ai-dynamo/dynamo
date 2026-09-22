# Remote Custom Encoder

This deployment keeps an application-owned custom encoder inside a bespoke orchestrator worker and calls a stock aggregated `dynamo.vllm` worker for generation.

```text
OpenAI client
     │
     ▼
generic Dynamo frontend
     │
     ▼
bespoke orchestrator worker
     ├── inline VisionEncoderBackend
     ├── ExternalEncoderHandoff
     └── LLMUnaryClient
                 │
                 ▼
       stock aggregated dynamo.vllm
```

The encoder remains transport-agnostic: users implement only `VisionEncoderBackend`. Dynamo's `ExternalEncoderHandoff` drives that backend, extracts image inputs, packages its ordered CPU tensors into the versioned `GenerateRequest.encoder_result`, and removes source-side multimodal fields before invoking vLLM. The application orchestrator only selects the generator model and forwards the prepared request. The remote vLLM endpoint streams token chunks internally; `LLMUnaryClient.complete()` folds them into the single terminal result returned by the orchestrator.

The initial handoff supports contiguous two-dimensional CPU `bfloat16`, `float16`, and `float32` linear embeddings. It requires a text-only aggregated generator configured with `--enable-prompt-embeds`. The payload travels through the ordinary MsgPack request plane; it does not use NIXL.

## Run

From the repository root:

```bash
./examples/custom_encoder/remote/launch.sh
```

The default `HitchhikersVisionEncoder` ignores the image contents and substitutes the embeddings for a known phrase, making the prompt-splicing path easy to inspect. Replace it with another `VisionEncoderBackend` using:

```bash
DYN_ENCODER_CLASS=your_package.YourVisionEncoder \
./examples/custom_encoder/remote/launch.sh
```

Send an OpenAI-compatible request to the public orchestrator model:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "remote-custom-encoder",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Based on The Hitchhiker’s Guide to the Galaxy, The Answer to"},
        {"type": "image_url", "image_url": {"url": "https://example.com/ignored.png"}},
        {"type": "text", "text": " is?"}
      ]
    }],
    "max_tokens": 24,
    "temperature": 0
  }'
```

The default backend is expected to steer the model toward `42`; it is a semantic smoke test rather than a real image encoder.
