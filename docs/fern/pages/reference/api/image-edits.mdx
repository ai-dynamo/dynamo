---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Image Edits API
subtitle: Uploads, JSON references, input limits, errors, and backend capability registration.
---

**Experimental.** `/v1/images/edits` accepts a prompt and reference images for a
model that supports image conditioning. Check the
[build requirements and request examples](../../use-cases/diffusion/workflows/text-to-image.md#create-a-variation-with-image-edits)
before using this API. For vLLM-Omni model limits, see
[Image Edit Capabilities](../backends/vllm-omni-configuration.mdx#image-edit-capabilities).

## Request Contract

Send either `multipart/form-data` uploads or `application/json` references.
`prompt` and at least one input image are required. `n` controls the number of
output images, independently of the input count.

### Multipart Uploads

- Use `image` for one file, or repeat `image` or `image[]` for multiple files.
  Choose one field name per request; input order is multipart part order.
- Upload PNG, JPEG, or WebP. File contents determine the format, regardless of
  the filename or declared MIME type.
- Do not mix files with `input_reference`, `url`, or `url[]` fields. Mask fields
  (`mask` and `mask_image`), duplicate scalar fields, and unknown fields return
  HTTP 400.

Supported scalar fields are `prompt`, `model`, `n`, `quality`, `background`,
`output_format`, `response_format`, `size`, `style`, `user`, and `moderation`.
Backend validation determines which controls the loaded model supports.
Encode `nvext` as one form field containing a JSON object.

### JSON References

Set `input_reference` to an HTTP(S) image URL, a base64 data URL such as
`data:image/png;base64,...`, or an ordered array of those strings. Empty arrays,
blank references, and filesystem paths are invalid. Do not combine
`input_reference` with image, URL, or mask fields.

`/v1/images/generations` also accepts this JSON extension for reference-capable
models. Both routes apply the same input-count and serialized-request limits.
Workers validate URL destinations and redirects before fetching references.

## Image Edit Limits

Set these environment variables before starting the listed processes. Values
must be positive decimal integers without unit suffixes; invalid values fail
startup. Pixel counts are width multiplied by height before RGB expansion.
**Both** means the frontend and each aggregated Omni worker. When frontend,
worker, and model limits differ, the lower applicable limit wins.

| Variable | Default | Set on | Scope |
|---|---:|---|---|
| `DYN_IMAGE_EDIT_MAX_IMAGES` | 4 | Both | Input count, also bounded by model capability. |
| `DYN_IMAGE_EDIT_MAX_FILE_BYTES` | 10485760 (10 MiB) | Frontend | Each uploaded file. |
| `DYN_IMAGE_EDIT_MAX_TOTAL_BYTES` | 20971520 (20 MiB) | Frontend | Uploaded image bytes per request. |
| `DYN_IMAGE_EDIT_MAX_TEXT_BYTES` | 65536 (64 KiB) | Frontend | Multipart scalar bytes per request. |
| `DYN_IMAGE_EDIT_MAX_PARTS` | 32 | Frontend | All multipart parts. |
| `DYN_IMAGE_EDIT_MAX_PIXELS` | 16000000 | Both | Each image: uploads at the frontend, every reference at the worker. |
| `DYN_IMAGE_EDIT_MAX_TOTAL_PIXELS` | 32000000 | Both | Images per request: uploads at the frontend, every reference at the worker. |
| `DYN_IMAGE_EDIT_MAX_REQUEST_BYTES` | 29360128 (28 MiB) | Frontend | Normalized worker request, including base64 expansion and other fields. |

Additional limits apply:

- [`DYN_HTTP_BODY_LIMIT_MB`](../components/frontend-configuration.mdx#http-and-networking)
  caps the complete HTTP body, including multipart framing and chunked uploads.
- [`DYN_MM_MAX_DATA_URL_MB` and `DYN_MM_MAX_FILE_SIZE_MB`](../components/runtime-configuration.mdx#media-input-limits)
  bound inline references and remote downloads on the worker. Uploads become
  data URLs, so account for base64 expansion when configuring both sides.
- [`DYN_TCP_MAX_MESSAGE_SIZE`](../components/runtime-configuration.mdx#communication-planes)
  caps each TCP frame. The default 28 MiB request budget leaves room below the
  default 32 MiB frame cap. If increasing the request budget, allow space for
  the transport envelope and configure the frame limit on both sides.

These are per-request limits. They do not bound total process memory under
arbitrary concurrency or guarantee that an accepted request fits GPU memory.

## Errors and Cancellation

These statuses cover request validation, reference loading, and admission:

| HTTP status | Cause | Client action |
|---|---|---|
| 400 | Malformed fields, missing inputs, mixed sources, unsupported input count, or an unavailable URL connection. | Correct the request, check the URL, or use a model and deployment that support the input count. |
| 408 | A reference-image download timed out. | Check that the worker can fetch the URL within its timeout, or upload the image. |
| 413 | An image, body, text, part-count, pixel, or serialized-request budget was exceeded. | Reduce the input size or count; ask the deployment operator about the relevant limit. |
| 415 | Unsupported request content type or image format. | Use JSON references or multipart PNG, JPEG, or WebP uploads. |
| 503 | No ready serving capacity, or worker membership changed before dispatch. | Retry after capacity stabilizes. |

Remote image servers can return other HTTP errors, which retain their status.
A failing reference stops the edit before generation; Dynamo does not return a
successful edit using only the earlier references. Client disconnects cancel
pending upload and reference-fetch work.

## Capabilities and Rolling Upgrades

The frontend admits an edit only when every worker it might route that request
to supports the input count. A legacy worker or one with unknown capability
limits those requests to one input. An explicitly advertised zero rejects
edits. The Omni worker also checks its loaded model and configured limit before
loading references; changing the model name in a request cannot raise the limit.

Upgrade all workers eligible to serve the model before sending multiple inputs.
A single reference retains the legacy string wire format; multiple references
use an array under the same `input_reference` field. Older workers reject the
array instead of silently using only its first image.

## Custom Backend Capability Schema

Backend authors, including authors of custom in-process engines, register this
capability in the image engine's
[`ModelDeploymentCard.user_data`](https://github.com/ai-dynamo/dynamo/blob/main/lib/llm/src/model_card.rs).
It is deployment metadata, not a field in an HTTP image request. The following
example is the value of `user_data`:

```json
{
  "image_edits": {
    "schema_version": 1,
    "max_input_images": 4
  }
}
```

| Field within `image_edits` | Type | Meaning |
|---|---|---|
| `schema_version` | Integer | Must be `1` for this schema. |
| `max_input_images` | Nonnegative integer | `0` rejects edits; `1` admits one reference; larger values admit ordered arrays up to that count. |

Missing or invalid records, and unknown schema versions, fall back to one
reference. This compatibility fallback does not establish that a backend
supports editing. Publish the smaller of the loaded model's supported count
and the deployment's configured count. The engine must still validate and load
all references in order; advertising a capability does not implement it.
