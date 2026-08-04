---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Custom Vision Encoders
subtitle: Run an author-defined vision encoder with a text-only model in an aggregated Dynamo vLLM worker
---

NVIDIA Dynamo can run an author-defined vision encoder before a text-only vLLM
decoder. The encoder and decoder run in the same aggregated `dynamo.vllm`
worker process. The encoder returns one `EncoderResult` per image, and Dynamo
adapts each result's artifact into the decoder prompt.

Use this path when a text-only language model can consume external prompt
embeddings but its vision tower or projector is private, experimental, or not
available in vLLM. This path is not encoder disaggregation: it has no separate
encode worker or embedding transfer.

## Current Scope

| Capability | Support |
| --- | --- |
| Backend | Python `dynamo.vllm` worker |
| Topology | Aggregated only |
| Decoder | Text-only vLLM model; multimodal decoders are rejected |
| Input | URL or data URL from each `image_url` content part |
| Video and audio | Not supported |
| Decoder input | Mixed token IDs and prompt embeddings |
| Cross-request batching | Eager batching on one encoder actor thread |
| Response metadata | Optional JSON through `nvext.custom_encoder` with request opt-in |
| CUDA graph buckets | Reserved for future support; `target_bucket` is currently `None` |

Configure the worker with `--custom-encoder-class`, `--enable-multimodal`, and
`--enable-prompt-embeds`. `--custom-encoder-class` also accepts the
`DYN_CUSTOM_ENCODER_CLASS` environment variable. The custom encoder is
incompatible with:

- `--use-vllm-tokenizer`
- `--frontend-decoding`
- Any `--disaggregation-mode` other than `agg`

Dynamo validates these combinations during worker startup.

## Request Flow

For an image request, Dynamo performs the following steps:

1. The frontend renders the chat template and tokenizes the prompt. The template
   must emit exactly one image-placeholder token for each image content part.
2. The frontend sends the prompt token IDs and the original image URL strings to
   the aggregated worker.
3. `AsyncVisionEncoder` optionally preprocesses the image URLs on a bounded CPU
   thread pool.
4. A dedicated actor thread coalesces items from concurrent requests and calls
   `VisionEncoderBackend.forward_batch()` synchronously.
5. The backend returns one ordered `EncoderResult` per input. Each result contains
   a decoder artifact and optional response metadata.
6. `LinearEmbedsAdapter` validates the artifacts and creates a mixed vLLM
   `EmbedsPrompt`. vLLM embeds the text positions and uses the custom artifacts
   for the image positions.
7. If the request selects `custom_encoder` in `nvext.extra_fields`, Dynamo carries
   the optional response metadata to `nvext.custom_encoder`.

The adapter expands each single image-placeholder token to the number of rows in
that image's artifact. A template must not emit a pre-expanded placeholder span.

> [!IMPORTANT]
> Before rendering a mixed-content chat template, the frontend changes an
> `image_url` content part to `{"type":"image"}`. Match
> `content.type == "image"` in a custom template. The bundled template shows the
> expected form.

## Run the Included Backend

From the repository root, launch the aggregated worker:

```bash
bash examples/custom_encoder/launch/agg_custom.sh --gpu 0
```

The launcher uses `Qwen/Qwen2.5-1.5B-Instruct`, the bundled Qwen chat template,
and `HitchhikersVisionEncoder`. The backend ignores the image URL and substitutes
the language model embeddings of a fixed phrase, which verifies the complete
mixed-embedding path without loading a vision model. It is a test backend, not a
production vision encoder.

Send a request from another terminal:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Based on The Hitchhiker's Guide to the Galaxy, The Answer to"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "text", "text": " is?"}
      ]
    }],
    "max_tokens": 32,
    "temperature": 0,
    "nvext": {"extra_fields": ["custom_encoder"]}
}
JSON
```

The response should contain `42` and a `phrase_token_count` value under
`nvext.custom_encoder.items[0]`.

To use another backend with the launcher, set its wrapper-specific
`DYN_ENCODER_CLASS` variable or pass `--encoder-class`:

```bash
DYN_MODEL=my-org/my-language-model \
DYN_ENCODER_CLASS=my_package.encoders.MyVisionEncoder \
bash examples/custom_encoder/launch/agg_custom.sh --gpu 0
```

The launcher supplies the required multimodal and prompt-embedding flags. Set
`DYN_CUSTOM_JINJA_TEMPLATE` or pass `--custom-jinja-template` when the model's
chat template does not emit the required placeholder token.

The reusable Qwen-family base, bundled template, and semantic test backend are
under
[`examples/custom_encoder`](https://github.com/ai-dynamo/dynamo/tree/main/examples/custom_encoder).

## Implement `VisionEncoderBackend`

Import and subclass `VisionEncoderBackend` from
`dynamo.vllm.multimodal_utils.custom_encoder`. The backend is synchronous and
contains model policy and compute. Dynamo owns its threads, batching, and async
request integration.

| Member | Execution Context | Responsibility |
| --- | --- | --- |
| `image_token_id` | Available when the backend is instantiated | Integer placeholder token ID used by the model and chat template |
| `build(model_id)` | Encoder actor thread, once during startup | Load the encoder, choose its device, and initialize thread-affine resources |
| `preprocess(raw)` | Optional CPU thread pool | Fetch, decode, resize, or patchify one image and return `Preprocessed(item, cost)` |
| `forward_batch(items, target_bucket=None)` | Encoder actor thread | Run one synchronous batched forward and return one `EncoderResult` per item, in order |
| `close()` | Encoder actor thread, once during shutdown | Release thread-affine resources |

`build()` and `forward_batch()` are required. `preprocess()` and `close()` have
defaults.

```python
import torch

from dynamo.vllm.multimodal_utils.custom_encoder import (
    EncoderResult,
    VisionEncoderBackend,
)


class MyVisionEncoder(VisionEncoderBackend[str, str, torch.Tensor]):
    image_token_id = 151655
    max_batch_cost = 8

    def build(self, model_id: str) -> None:
        self.model = load_encoder(model_id)

    def forward_batch(
        self,
        items: list[str],
        target_bucket: int | None = None,
    ) -> list[EncoderResult[torch.Tensor]]:
        artifacts = self.model(items)
        return [
            EncoderResult(
                artifact=artifact.detach().cpu(),
                response_data={"visual_tokens": int(artifact.shape[0])},
            )
            for artifact in artifacts
        ]
```

The current `LinearEmbedsAdapter` accepts only `torch.Tensor` artifacts. Every
artifact must meet all of these requirements:

- Shape `(number_of_visual_tokens, decoder_hidden_size)`
- At least one visual-token row
- The decoder's configured dtype
- CPU device after all encoder device work is synchronized

Return one `EncoderResult` for every input item, in the same order. Returning a
raw tensor, a different result count, or an invalid artifact fails the request.
The adapter also requires one placeholder token per result and rejects a
placeholder-to-result count mismatch.

`image_token_id` must be available before `build()` runs because Dynamo creates
the decoder adapter first. Set it as a class attribute or during backend
construction rather than discovering it in `build()`.

## Configure Preprocessing

Preprocessing is disabled by default. With `preprocess_concurrency = 0`, Dynamo
does not call `preprocess()` and passes each raw image URL directly to
`forward_batch()` with a default cost of `1`.

To enable preprocessing, override `preprocess()` and set
`preprocess_concurrency` to a positive integer. Dynamo rejects a backend that
overrides `preprocess()` but leaves the concurrency at `0`. The method must be:

- Synchronous and thread-safe because pool threads can call it concurrently
- Deterministic for the same raw input
- CUDA-free because it does not run on the encoder actor thread
- Responsible for returning `Preprocessed(item, cost)` with a positive integer
  cost; pass-through batching ignores the value when `max_batch_cost` is `None`

Dynamo waits for every image in one request to finish preprocessing before it
submits any of them to the actor. If one image fails preprocessing, Dynamo
submits no encoder work for that request.

> [!WARNING]
> The backend owns any media retrieval performed by `preprocess()`. Apply
> Dynamo's
> [media URL policy](../../use-cases/multimodal-serving/overview.md#security-url-validation),
> finite network timeouts, response-size limits, and image decode limits rather
> than fetching arbitrary request URLs directly.

## Configure Cross-Request Batching

The encoder actor thread collects items from all concurrent requests, calls
`forward_batch()` for each physical batch, and returns each result to its
original request and image position.

The batcher does not add a timer. A lone image runs when the actor becomes free.
Items that accumulate while the actor is busy are drained together on its next
iteration. Batching therefore helps overlapping requests but does not turn a
serial workload into a larger batch.

### Choose a Batch Cost

`Preprocessed.cost` is the positive integer amount that one image contributes to
`max_batch_cost`. The batcher packs only by this scalar and does not inspect item
shapes.

| Processed Image Regime | Recommended Cost |
| --- | --- |
| Every item has the same bounded shape | `1`; the limit acts as a maximum image count |
| Native or variable resolution | Number of visual patches or tokens after preprocessing |
| Backend-specific memory relationship | A documented unit proportional to the limiting resource |

For variable-resolution inputs, a count-only limit can combine several
maximum-resolution images and exhaust GPU memory. Compute the processed grid in
`preprocess()`, use its patch or visual-token count as `cost`, and set a finite
`max_batch_cost` that the encoder can serve alongside the decoder.

With `max_batch_cost = None`, the batcher ignores costs and sends every item
already queued when the actor becomes free to one `forward_batch()` call. Use
this pass-through mode only when the backend performs its own safe sizing. A
finite limit rejects an individual item whose cost exceeds the limit.

The queue currently has no admission-capacity limit. Control request concurrency
upstream and use a finite batch cost when unbounded accumulation would exceed
memory or latency targets.

The encoder and vLLM share GPU memory. Leave enough memory outside vLLM's
`gpu_memory_utilization` for encoder weights and peak activation memory at
`max_batch_cost`. Exercise the maximum legal batch during deployment validation.

## Return Response Metadata

Each `EncoderResult.response_data` can be `None` or a JSON object. This data is
separate from the artifact consumed by the decoder.

To request the data, select `custom_encoder` through
[`nvext.extra_fields`](../additional-resources/nvidia-request-extensions-nvext.md#response-extensions):

```json
{
    "nvext": {
        "extra_fields": ["custom_encoder"]
    }
}
```

Dynamo returns one entry per input image in order and preserves missing values
as `null`:

```json
{
    "nvext": {
        "custom_encoder": {
            "items": [
                {"visual_tokens": 256},
                null
            ]
        }
    }
}
```

The selector controls client exposure, not backend computation. Dynamo still
calls `forward_batch()` and validates every non-`None` `response_data` value when
the client does not request the field. The Python worker and Rust response
builder both enforce the selector before data reaches the client.

Metadata must satisfy these constraints:

- Each non-`None` value is a JSON object, not a scalar or array
- Nested values are JSON-serializable and contain no `NaN` or infinity values
- The compact UTF-8 encoding of the combined `{"items":[...]}` payload is at
  most 64 KiB

For simplicity and performance, the Python layer does not prevalidate
arbitrary-size integers against Rust `serde_json`'s `i64`/`u64` range. Keep
metadata integers within those ranges; larger values can fail at the
Python-to-Rust response boundary.

Dynamo omits `nvext.custom_encoder` when the request does not select it or every
item has `response_data=None`. A streamed Chat Completions response emits the
metadata once on the first non-error response chunk; a unary response retains it
while aggregating the internal stream.

Requests that select `custom_encoder` and include a nonempty `tools` list are
rejected. A tool-bearing request can still use the custom encoder artifact for
generation when it does not request the response metadata.

Invalid or oversized metadata fails the request before decoder generation. The
validation round trip also detaches the response payload from mutable backend
objects.

## Failure and Cancellation Behavior

A `build()` failure prevents the worker from starting. During a request, Dynamo
converts preprocessing, encoder, metadata, placeholder, and adapter failures to
a `CustomEncoder failed` request error rather than invoking the decoder with a
partial prompt.

Once submitted, items from different requests can share one `forward_batch()`.
An exception from that call fails every live request represented in the physical
batch. Validate malformed, oversized, or unsupported images during
`preprocess()` so input-dependent errors do not reach a shared GPU forward.

If an awaiting request is canceled before its work passes the final dispatch
check, Dynamo tombstones its items and excludes them from later batches. A
synchronous `forward_batch()` already committed for execution cannot be
preempted. It finishes, and Dynamo discards the canceled request's result.

## Operational Checklist

- Use a text-only decoder and enable multimodal input plus prompt embeddings.
- Make `image_token_id` available when the backend is instantiated.
- Match `content.type == "image"` and emit exactly one placeholder token per
  image in the chat template.
- Return one nonempty CPU tensor per image with the decoder hidden size and
  dtype.
- Use distinct images in correctness tests so reordered results cannot pass.
- Keep optional response metadata JSON-safe and below the 64 KiB payload limit.
- Do not combine `nvext.custom_encoder` response metadata with request tools.
- Test the largest image and maximum batch cost with the decoder resident on the
  same GPU.
- Exercise concurrent requests because a serial smoke test does not prove
  coalescing.
- Keep blocking media operations off the actor thread by enabling the
  preprocessing pool.
- Bound request concurrency upstream because the encoder queue has no admission
  capacity limit.
