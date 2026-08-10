<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Author an aggregated inference workflow

This working example runs one encoder, shares its artifacts with a classifier
and an embedded LLM, then joins both results behind one Dynamo endpoint. The
pipeline itself is the small part:

Start with `workflow.py`. It contains only the authored graph; `stages.py`
contains the application-specific runners, while `worker.py` loads resources,
binds them to the graph, and exposes the endpoint.

```python
workflow = Workflow("encoder-classifier-llm")
image_url = workflow.input("image_url", type="text")
request = workflow.input(
    "request", type="object", class_id="dynamo.common.backend.GenerateRequest"
)

encoder = workflow.stage(
    "encoder", EncoderStage, image_url=image_url, request=request
)
classifier = workflow.stage(
    "classifier", DummyClassifier, artifacts=encoder.artifacts
)
generator = workflow.stage(
    "generator", VllmDecoderStage, request=request, prompt=encoder.prompt
)

workflow.output("scores", classifier.scores)
workflow.output("chunk", generator.chunk)
```

Each worker declares its input and output contract once and implements the
two-argument async `run` method. A different classifier with the same contract
can replace `DummyClassifier`; Dynamo validates the binding, runs the encoder
once, schedules the independent branches concurrently, joins them, and handles
failure cancellation.

The image URL is intentionally just ingress data for this example. Workflow
ports can also describe decoded images, tensors, text, bytes, JSON, or internal
Dynamo objects. Nothing in the workflow core is specific to vision or vLLM.

## Run

From the repository root on a GPU machine:

```bash
./examples/custom_backend/user_ensemble/launch.sh
```

Then send a non-streaming request:

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

The ordinary OpenAI response contains generated text. The joined classifier
result is returned under `nvext.engine_data.ensemble.classifier_scores`:

```json
{"dummy-classification": 1.0}
```

The frontend owns the only HTTP inference port. `UserEnsembleEngine` owns the
endpoint registration. `VllmDecoderRuntime` owns the native vLLM engine, and
`VllmDecoderStage` borrows it as a local workflow adapter.
