# Remote encoder, classifier, and stock vLLM workflow

> Experimental: This workflow API is under active development and may evolve
> based on feedback. It is an optional orchestration layer over Dynamo's
> discoverable workers and endpoints; applications may compose those primitives
> directly when they prefer to own the orchestration logic.

This example qualifies one static disaggregated workflow:

```text
                                  NIXL read 1
                              ┌────────────────> classifier ──┐
request -> remote encoder ----┤                               ├─> inline response
                              └────────────────> stock vLLM ──┘
                                  NIXL read 2
```

The encoder packs dynamically sized per-image feature rows into one tensor and
publishes `row_splits` plus `image_token_id` as JSON metadata. The workflow
runtime schedules the two consumers concurrently. The generator binding calls a
normal aggregated `dynamo.vllm` Generate endpoint; that worker imports the
tensor and creates its mixed `EmbedsPrompt` without a workflow-specific decoder.
The application-owned response stage runs inline in the orchestrator process,
so it adds no worker deployment or network hop. The frontend remains generic
and discovers the orchestrator as an ordinary model worker.

Dynamo supplies `EncoderStage`, `DynamoVllmStage`, and the remote encoder
launcher. This application owns the custom encoder selection, classifier,
workflow, endpoint bindings, response shaping, and process deployment. Set
`DYN_ENCODER_CLASS` to any compatible zero-argument `VisionEncoderBackend`
subclass; it must return one nonempty 2D CPU tensor per image and declare
`image_token_id`. `DYN_ENCODER_MODEL` may select encoder weights independently
and defaults to `DYN_MODEL`.

From the repository root, run:

```bash
examples/experimental/workflow/user_ensemble/remote/launch.sh
```

Then send an image-bearing chat request:

```bash
curl localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Based on The Hitchhiker’s Guide to the Galaxy, The Answer to"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="}},
        {"type": "text", "text": " is?"}
      ]
    }],
    "max_tokens": 32,
    "nvext": {"extra_fields": ["engine_data"]}
  }'
```

The default deterministic encoder converts the image slot into embeddings for
the Hitchhiker phrase, so a successful semantic run answers `42`. It keeps its
features on CPU while the stock vLLM worker uses one GPU. With the `engine_data`
opt-in above, classifier output is returned at
`nvext.engine_data.ensemble.classifier_scores`; without the opt-in, the existing
frontend response policy omits `engine_data`.
