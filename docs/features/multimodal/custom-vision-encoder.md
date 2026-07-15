---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Custom Vision Encoders
subtitle: Run a bespoke vision tower in an aggregated Dynamo vLLM worker
---

A custom vision encoder lets an aggregated `dynamo.vllm` worker use an
author-provided vision tower or projector instead of vLLM's built-in multimodal
encoder. The encoder runs in the worker process, produces one embedding tensor
per image, and Dynamo inserts those embeddings at the image-placeholder positions
of a mixed prompt.

Use this path when the language model can consume external prompt embeddings but
the vision encoder is private, experimental, or otherwise unavailable in vLLM.
It is not encoder disaggregation: the encoder and language model share one worker
process and GPU, and no embedding transfer occurs.

## Current Scope

| Capability | Support |
| --- | --- |
| Backend | Legacy Python `dynamo.vllm` worker |
| Topology | Aggregated only |
| Input | `image_url` content |
| Video and audio | Not supported |
| Language-model input | Mixed token IDs and prompt embeddings |
| Cross-request batching | Yes |
| CUDA graph bucket selection | Reserved for future support; current dispatch is eager |

The custom-encoder path requires `--enable-multimodal` and
`--enable-prompt-embeds`. It is incompatible with frontend decoding, the vLLM
tokenizer mode, legacy multimodal worker roles, and non-aggregated disaggregation
modes. Dynamo validates these combinations during startup.

## Run the Included Path

From the repository root, launch the aggregated worker:

```bash
bash examples/custom_encoder/launch/agg_custom.sh --gpu 0
```

The launcher defaults to `Qwen/Qwen2.5-1.5B-Instruct` and the
`HitchhikersVisionEncoder`. That encoder intentionally ignores the image and
substitutes embeddings for a fixed phrase so the complete prompt-embedding path
can be checked semantically. It is a test backend, not a production vision
encoder.

Select your own backend with a dotted Python class path:

```bash
DYN_MODEL=my-org/my-language-model \
DYN_ENCODER_CLASS=my_package.encoders.MyVisionEncoder \
bash examples/custom_encoder/launch/agg_custom.sh --gpu 0
```

The launcher supplies the required multimodal and prompt-embedding flags. If the
language model's chat template does not render an image-placeholder token, also
provide `DYN_CUSTOM_JINJA_TEMPLATE` or `--custom-jinja-template`.

<Warning>
The backend owns any media retrieval performed by `preprocess()`. Apply Dynamo's
[media URL policy](README.md#security-url-validation), finite network timeouts,
response-size limits, and image decode limits rather than fetching arbitrary
request URLs directly.
</Warning>

## Implement `VisionEncoderBackend`

Subclass
`dynamo.vllm.multimodal_utils.vision_encoder_backend.VisionEncoderBackend` and
implement the following contract:

| Member | Execution context | Responsibility |
| --- | --- | --- |
| `image_token_id` | Configuration | Hardcoded placeholder token ID used by the model or custom template |
| `build(model_id)` | Encoder actor thread, once | Load the encoder, choose its device, and initialize thread-affine resources |
| `preprocess(raw)` | Optional CPU thread pool | Fetch, decode, resize, or patchify one image and return `Preprocessed(item, cost)` |
| `forward_batch(items, target_bucket=None)` | Encoder actor thread | Run one synchronous batched forward and return one CPU tensor per item, in order |
| `close()` | Encoder actor thread, once | Release thread-affine resources |

`forward_batch()` returns one tensor shaped
`(number_of_visual_tokens, language_model_hidden_size)` for every input item. It
must synchronize any device work and copy the results to CPU before returning;
the request coroutine consumes them from a different thread.

Preprocessing is disabled by default. To enable it, override `preprocess()` and
set `preprocess_concurrency` to a positive value. The method must be synchronous,
thread-safe, deterministic, and CUDA-free because multiple pool threads may call
it concurrently. With `preprocess_concurrency = 0`, Dynamo skips `preprocess()`
and passes the raw image URL directly to `forward_batch()`.

The reusable Qwen-family base and the semantic test backend are under
[`examples/custom_encoder`](https://github.com/ai-dynamo/dynamo/tree/main/examples/custom_encoder).

## Cross-Request Batching

Each request calls the async encoder with its images. A dedicated actor thread
collects items from all concurrent requests, invokes `forward_batch()` once for
the physical batch, and returns each result to the request and position that
submitted it.

The batcher does not add a timer. A lone image runs as soon as the actor is free;
images that accumulate while the actor is busy are coalesced on its next pass.
Batching therefore helps when requests overlap. It does not change a serial,
single-request workload into a larger batch.

### Choose a Batch Cost

`Preprocessed.cost` is the amount one image contributes to
`max_batch_cost`. The batcher does not inspect image tensors or shapes.

| Processed image regime | Recommended cost |
| --- | --- |
| Every item has the same bounded shape | `1`; the limit acts as a maximum image count |
| Native or variable resolution | Number of visual patches or tokens after preprocessing |
| Backend-specific memory relationship | A documented positive unit that remains proportional to the limiting resource |

For variable-resolution inputs, a count-only limit can combine several maximum
resolution images and exhaust GPU memory. Compute the processed grid in
`preprocess()`, use its patch or visual-token count as `cost`, and set a finite
`max_batch_cost` that the encoder can serve alongside the language model.

With `max_batch_cost = None`, the batcher passes every item already queued when
the actor becomes free to one `forward_batch()` call and ignores per-item costs.
Use this pass-through mode only when the backend performs its own safe sizing.
A finite limit rejects an individual item whose cost cannot fit any batch.

The encoder and vLLM share GPU memory. Leave enough memory outside vLLM's
`gpu_memory_utilization` for the encoder weights and the peak activation memory
at `max_batch_cost`, and exercise that maximum legal batch during startup or
deployment validation.

### Failure and Cancellation Behavior

Validate malformed, oversized, or unsupported images in `preprocess()`. Dynamo
waits for every image in one request to finish preprocessing and submits no GPU
work if any of them fails.

Once submitted, items from different requests may share one `forward_batch()`.
An exception from that call fails every live request represented in that physical
batch, so input-dependent failures should not be deferred to the GPU forward.

If an awaiting request is canceled before its work passes the final dispatch
check, Dynamo tombstones its items and excludes them from later shared batches. A
synchronous `forward_batch()` already committed for execution cannot be
preempted; it finishes, and the canceled request's result is discarded.

## Operational Checklist

- Confirm the chat template emits exactly one placeholder span for every image.
- Confirm each returned tensor has the language model's hidden size and the row
  count expected at its placeholder span.
- Use distinct images in correctness tests so reordered results cannot pass.
- Test the largest permitted image and maximum batch cost with the language model
  resident on the same GPU.
- Exercise concurrent requests; a serial smoke test does not prove coalescing.
- Keep blocking media operations out of the actor thread by enabling the
  preprocessing pool.
