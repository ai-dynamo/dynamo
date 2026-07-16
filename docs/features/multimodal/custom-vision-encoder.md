---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Custom Vision Encoders
subtitle: Run a bespoke vision tower in an aggregated Dynamo vLLM worker
---

A custom vision encoder lets an aggregated `dynamo.vllm` worker use an
author-provided vision tower or projector instead of vLLM's built-in multimodal
encoder. The encoder runs in the worker process and produces encoded media for
each image. Dynamo chooses a mixed prompt-embedding route or a model-native
external multimodal route from the backend's setup-time contract.

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
| Language-model input | Mixed prompt embeddings, or native Qwen2/2.5 multimodal embeddings |
| Cross-request batching | Yes |
| CUDA graph bucket selection | Reserved for future support; current dispatch is eager |

The custom-encoder path requires `--enable-multimodal`. Linear-row backends also
require `--enable-prompt-embeds`; Qwen2/2.5 native-multimodal backends require
`--enable-mm-embeds`. It is incompatible with frontend decoding, the vLLM
tokenizer mode, legacy multimodal worker roles, and non-aggregated disaggregation
modes. Dynamo validates these combinations during startup.

The native Qwen adapter is currently validated against vLLM 0.24.0 and 0.25.1
(current Dynamo main pins 0.25.1). It requires TP=PP=DP=1 and rejects
`--language-model-only` and vLLM multimodal-encoder CUDA graphs. The latter graph
path expects pixel inputs and is not compatible with external image embeddings.
Prefix caching and chunked prefill are also rejected in the correctness MVP until
their image-span boundary and cache-identity parity matrices are complete; the
included launcher disables both in native mode.

| Backend result | Engine request | Position semantics |
| --- | --- | --- |
| Legacy tensor or `LinearRowsV1` | `EmbedsPrompt` | Ordinary one-dimensional positions |
| `Qwen2VLImageEncodingV1(projected, grid_thw)` | `TokensPrompt` with `image_embeds` and `image_grid_thw` | Qwen grid-driven M-RoPE |

Qwen backends must use the native route. A mixed `EmbedsPrompt` has no place for
`image_grid_thw`, so vLLM would assign ordinary text positions to the image rows.

## How `BoundCustomEncoderAdapter` Fits

`BoundCustomEncoderAdapter` is the compatibility boundary between an
author-provided backend and the loaded vLLM model. "Bound" means it validates one
backend's setup-time contract against one resolved model configuration. It then
translates that backend's canonical request results into a closed engine prompt
plan. It does not fetch images, schedule batches, execute the vision model, or
calculate M-RoPE.

At startup:

```
 +---------------------------+       +---------------------------+
 | VisionEncoderBackend      |       | resolved vLLM config      |
 | - encoding_spec           |       | - architecture            |
 | - producer fingerprint    |       | - hidden size / dtype     |
 | - output geometry         |       | - merge size / token IDs  |
 +-------------+-------------+       | - flags and TP/PP/DP      |
               |                     +-------------+-------------+
               |                                   |
               +------------------+----------------+
                                  |
                                  v
                   +-------------------------------+
                   | BoundCustomEncoderAdapter     |
                   | - validate ABI/model match    |
                   | - validate vLLM version       |
                   | - validate flags/topology     |
                   | - select prompt route         |
                   | - remember Qwen token IDs     |
                   +---------------+---------------+
                                   |
                          mismatch | match
                            +------+------+
                            |             |
                            v             v
                      fail startup   load encoder
```

For each request:

```
 token IDs + image inputs
           |
           v
 +---------------------------+
 | AsyncVisionEncoder        |
 | preprocess all-or-nothing |
 +-------------+-------------+
               |
               v
 +---------------------------+
 | ThreadedMicroBatcher      |
 | cross-request actor       |
 +-------------+-------------+
               |
               v
 +---------------------------+
 | VisionEncoderBackend      |
 | forward_batch             |
 +-------------+-------------+
               |
               | tagged encoded-media results
               v
 +---------------------------+
 | reconcile_and_canonicalize|
 | validate, reorder, clone  |
 +-------------+-------------+
               |
               v
 +---------------------------+
 | BoundCustomEncoderAdapter |
 | prepare_prompt_plan       |
 +-------------+-------------+
               |
       +-------+-------+
       |               |
       v               v
 MixedEmbedsPlan   NativeMMPlan
       |               |
       v               v
 EmbedsPrompt      TokensPrompt
                       |
                       v
              vLLM computes Qwen M-RoPE
```

The adapter selects the route from `encoding_spec.adapter_abi`:

```
 linear-rows-v1
     |
     +--> splice visual rows into the text embedding sequence
     +--> MixedEmbedsPlan --> EmbedsPrompt

 vllm-qwen2-vl-external-v1
     |
     +--> concatenate projected image rows in request order
     +--> stack image_grid_thw in the same order
     +--> NativeMMPlan --> TokensPrompt
     +--> vLLM expands placeholders and constructs grid-driven M-RoPE
```

Actor-side correlation and tensor-ownership checks are implemented by the
separate `reconcile_and_canonicalize()` helper in the same adapter module. This
keeps `ThreadedMicroBatcher` generic and torch-free while ensuring a backend
cannot accidentally associate one image's rows with another image's grid.

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

The launcher supplies the required multimodal and selected engine-input flag. If the
language model's chat template does not render an image-placeholder token, also
provide `DYN_CUSTOM_JINJA_TEMPLATE` or `--custom-jinja-template`.

To run the eager Qwen2.5-VL correctness backend with the registered full model
wrapper, select the native input mode:

```bash
DYN_MODEL=Qwen/Qwen2.5-VL-3B-Instruct \
DYN_ENCODER_CLASS=examples.custom_encoder.qwen2_5_vl_vision_encoder.Qwen2_5VLVisionEncoder \
DYN_ENCODER_INPUT_MODE=native \
bash examples/custom_encoder/launch/agg_custom.sh --gpu 0 \
  --revision 66285546d2b821cf421d4f5eb2576359d3770cd3
```

The included backend accepts base64 image data URLs and is intentionally eager.
It loads the stock vision tower outside vLLM for parity testing. The full Qwen
wrapper remains resident inside vLLM, but supplying external image embeddings
skips that wrapper's vision forward for the request. vLLM may still execute its
resident tower during startup profiling; this MVP does not claim zero tower
forwards across the entire process lifecycle.

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
| `encoding_spec` | Configuration | Optional typed output/engine-adapter handshake; omitted for legacy tensor backends |
| `image_token_id` | Configuration | Placeholder token ID for the linear route; unused by native Qwen adapters |
| `build(model_id)` | Encoder actor thread, once | Load the encoder, choose its device, and initialize thread-affine resources |
| `preprocess(raw)` | Optional CPU thread pool | Fetch, decode, resize, or patchify one image and return `Preprocessed(item, cost)` |
| `forward_batch(items, target_bucket=None)` | Encoder actor thread | Return legacy CPU tensors, or correlation-tagged typed media declared by `encoding_spec` |
| `close()` | Encoder actor thread, once | Release thread-affine resources |

`forward_batch()` returns one tensor shaped
`(number_of_visual_tokens, language_model_hidden_size)` for every input item. It
must synchronize any device work and copy the results to CPU before returning;
the request coroutine consumes them from a different thread.

A typed backend declares `BackendEncodingSpecV1`. Dynamo then passes
`ForwardItemV1(correlation_id, item)` values and expects
`EncodedMediaResultV1(correlation_id, media)` values. Results may be returned in
any order: Dynamo verifies a complete correlation-ID bijection, restores input
order, validates every tensor, and clones it into owned CPU storage before the
batcher resolves request futures.

For Qwen2/2.5, set `adapter_abi="vllm-qwen2-vl-external-v1"` and return one
`Qwen2VLImageEncodingV1` per image. Its `projected` tensor has shape
`(T * H * W / spatial_merge_size**2, decoder_hidden_size)` and `grid_thw` is the
corresponding positive `(T, H, W)` tuple. The initial image-only contract requires
`T == 1` and CPU, dense, contiguous, finite tensors in the declared dtype.
The tensor is after the vision projector/patch merger and is in canonical
`(t, merged_h, merged_w)` raster order. In particular, Qwen2.5 producers must
apply the model's inverse window permutation before returning it. A tensor in the
temporary window order has the right shape but receives spatially incorrect
M-RoPE positions.

Native Qwen specs must provide `expected_decoder_config_fingerprint`; matching
only hidden width and dtype is not sufficient to establish projector/decoder
compatibility. The producer fingerprint should bind the encoder/projector
weights, processor and tokenizer revisions, resize/normalization policy, and row
layout version used to produce the artifact.

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

The included Qwen2.5 correctness backend runs a near-maximum square-image canary
during `build()`, after the vLLM engine has allocated its model and KV cache. A
co-residency failure therefore stops startup instead of becoming the first legal
request's OOM.

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
