# Integrated-encoder stock vLLM workflow

This example qualifies a static workflow that keeps multimodal encoding inside
the stock aggregated `dynamo.vllm` worker:

```text
                         original GenerateRequest
                    ┌──────────────────────────────> stock vLLM ──┐
request -> orchestrator                                             ├─> response stage
                    └──────────────────────────> dummy classifier ──┘
```

The orchestrator fans the frontend-preprocessed request out unchanged. The
stock vLLM worker runs the configured custom encoder in process and folds its
token stream into one workflow completion. A replaceable remote CPU classifier
also consumes the request. The application-owned response stage attaches
classifier scores to `engine_data`. It runs inline in the frontend process by
default, or as a separately discovered CPU worker when
`DYN_USER_ENSEMBLE_RESPONSE_PLACEMENT=remote`.

There is no external encoder stage, NIXL tensor, or decoder-specific workflow
worker. Dynamo supplies `DynamoVllmStage`, `GenerateEndpointBinding`, the
orchestrator, and the stock vLLM custom-encoder path. The application owns only
its classifier, response shape, graph, deployment bindings, and encoder class
selection.

From the repository root, run:

```bash
examples/custom_backend/user_ensemble/remote/launch.sh
```

To place the response stage in its own process, run:

```bash
DYN_USER_ENSEMBLE_RESPONSE_PLACEMENT=remote \
examples/custom_backend/user_ensemble/remote/launch.sh
```

Then send an image-bearing request:

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

The default deterministic encoder turns the image slot into the Hitchhiker
phrase, so a successful semantic run answers `42`. Classifier output appears at
`nvext.engine_data.ensemble.classifier_scores` when `engine_data` is requested.
