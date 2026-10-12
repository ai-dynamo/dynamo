---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Generation Artifact Delivery
subtitle: Experimental metadata_upload contract and single-worker delivery authority
---

NVIDIA Dynamo carries delivery controls and opaque artifact bytes. The backend owns tensor capture and encoding. This experimental vLLM producer emits `generation_artifact_v1`, not native vLLM serialization.

> [!WARNING]
> Available in development builds containing this implementation. Local transport and capture tests do not establish production readiness. Public tenant-authorized status APIs, distributed takeover, relay reader leases, retention sweeping, and live inference-service validation remain separate work.

## Request Contract

Use the existing `nvext.metadata_upload` field. Do not add a separate `nvext.generation_artifact` field.

| Shape | Meaning |
|-------|---------|
| `{ "url": "s3://bucket/prefix" }` | Existing SGLang cumulative metadata upload; unchanged serializer and legacy fsspec transport. |
| `{ "kind": "generation_artifact", ... }` | Explicit artifact capture and delivery. |

The legacy contract also accepts optional `fallback_url` for alignment with the SGLang sidecar contract. The Python uploader attempts the primary prefix, then uploads the same serialized bytes once to the fallback prefix on an ordinary failure. Cancellation does not trigger fallback. Artifact requests are not automatically substituted for legacy metadata uploads.

| Artifact Field | Type | Constraint |
|----------------|------|------------|
| `kind` | string | `generation_artifact` |
| `contents` | string array | Unique names. vLLM supports `moe_routes`, `selected_logprobs`, and `topk_logprobs`; an empty array still records exact sequence IDs. Rust preserves bounded opaque names for future adapters. |
| `mode` | string | `inline_base64`, `object_store`, or reserved `relay_store` |
| `destination` | object | Absent for inline; required and mode-compatible otherwise |

```json
{
  "nvext": {
    "metadata_upload": {
      "kind": "generation_artifact",
      "contents": ["moe_routes", "selected_logprobs"],
      "mode": "object_store",
      "destination": {
        "kind": "managed_store",
        "profile": "rollout",
        "object_key": "artifacts/request-unique.dynexp"
      }
    }
  }
}
```

### Destinations

| Kind | Required Fields | Behavior |
|------|-----------------|----------|
| `managed_store` | `profile`, `object_key` | Native Python OpenDAL writes through a trusted profile. Keys must be normalized, relative, and within configured prefixes. Conditional create is required; existing objects are not replaced. |
| `presigned_http_put` | `url`, `required_headers`, `expires_at`, `max_bytes`, `object_id`, `verification_ref` | Sends one exact HTTP PUT. The protocol makes `verification_ref` optional, but this authority requires a registered verifier. |
| `mooncake_store` | `profile`, `consumer_group`, `retention_seconds` | Protocol reserves this shape for `relay_store`; this worker rejects public relay requests pending a configured lease/consumer authority. |

Presigned `url` is a capability, not a prefix. The uploader preserves its escaped path and query, applies required signed headers, and does not append a filename, follow redirects, use proxy environment variables, or reinterpret it through OpenDAL. Expiration is an RFC3339 timestamp. `max_bytes` must be positive and at most 64 MiB. Include a signed `if-none-match: *` header; when content length is signed, it must match the encoded artifact exactly. The authority binds the exact PUT URL hash to an independently authorized read profile and object key.

Public HTTPS targets require an exact host/port allowlist and connect-time address checks. Private-address access requires explicit allowed networks. Plain HTTP additionally requires an exact authority in `allow_http_hosts`. Credentials and grants are never returned in receipts.

## Results and Operations

The final generation response projects the backend receipt to `nvext.metadata_upload` independently of `extra_fields`. No tensor schema is decoded in Rust.

Inline results contain `kind`, `mode`, `format`, `contents`, `byte_count`, and `data_base64`, where the Base64 represents the complete binary artifact. Object delivery returns `state: "pending"`, `operation_id`, and `revision: 1` after durable handoff. The transfer owns source bytes independently of the response iterator.

| State | Meaning | Additional Fields |
|-------|---------|-------------------|
| `pending` | Source and encrypted destination grant are retained by this worker | `operation_id`, `revision` |
| `ready` | Bounded readback confirms exact length and SHA-256; returned provider version is read explicitly when available | `actual_bytes`, `sha256`, `object_id`, optional `provider_identity` |
| `failed` | Delivery or capture failed; the operation cannot be advertised as ready | `error_code` |

The trusted engine administrative route `artifact_operations` accepts `{ "operation_id": "ga_..." }`. It reads only the configured authority's operation records; it is not a public HTTP endpoint or a tenant authorization boundary. Keep it on an authenticated administrative network.

One worker and event loop exclusively own each SQLite operation database. A filesystem lock prevents concurrent authorities from performing recovery against the same canonical database path. Grants remain encrypted while pending and are erased at terminal completion. Lost PUT acknowledgements trigger exact-destination readback without a second PUT. Restart recovery marks lost volatile sources `failed/source_lost`; it does not claim upload resumption or cross-worker takeover. Retention and old-record cleanup must be managed operationally until a sweeper is implemented.

## Operator Configuration

Install the optional `artifact-delivery` extra for OpenDAL and encrypted operation records. Preserve the vLLM backend's ordinary dependencies. Inline-only delivery does not require this extra.

| Environment Variable | Default | Meaning |
|----------------------|---------|---------|
| `DYNAMO_ARTIFACT_CONFIG` | Unset | Trusted JSON configuration file; object delivery is disabled when unset. |
| `DYNAMO_ARTIFACT_CAPABILITY_KEY` | Required with config | Persistent Fernet key from a secret manager; protects pending grants. |
| `DYNAMO_ARTIFACT_CAPTURE_BYTES` | `268435456` | Process-wide capture admission bytes, including conservative source/scratch estimates. |
| `DYNAMO_ARTIFACT_ROUTER_PROFILE` | `[]` | JSON array of `[router_id, expert_count, top_k]` triples from trusted model configuration. Required for route capture. |

| Config Field | Default | Constraint |
|--------------|---------|------------|
| `owner` | Required | Fixed security-domain identity, 1–128 characters; not client supplied. |
| `operation_db` | Required | Dedicated persistent SQLite path for this worker; protect directory access. |
| `max_source_bytes` | `268435456` | Positive queued source-byte capacity. |
| `max_pending_operations` | `128` | Positive maximum queued operations, independently bounding capability/task overhead for small payloads. |
| `profiles` | `{}` | Named trusted OpenDAL profiles. Each requires `service`, `options`, `allowed_prefixes`, and `immutable: true`. |
| `verifiers` | `{}` | Named entries with `put_url_sha256`, `profile`, and `object_key`; required for delegated PUT. |
| `storage_policy.allowed_hosts` | `[]` | Exact authorities allowed for delegated PUT. |
| `storage_policy.allowed_networks` | `[]` | Explicit private-address CIDRs; public addresses otherwise allowed. |
| `storage_policy.allow_http_hosts` | `[]` | Exact authorities permitted to use plain HTTP. |
| `storage_policy.max_bytes` | `67108864` | Maximum single PUT, at most 64 MiB. |
| `storage_policy.timeout_seconds` | `60` | Positive finite upload timeout. |
| `storage_policy.max_presigned_ttl_seconds` | `3600` | Maximum remaining delegated capability lifetime. |

`immutable: true` is an operator certification, not an object-store configuration change. Enforce it with provider policy and unique keys. Managed profiles use OpenDAL's native options and credentials from protected server configuration, never from the request. This implementation uses Python OpenDAL; it does not extract or duplicate the SGLang sidecar's Rust storage service. Legacy Python SGLang uploads still use fsspec.

Delivery concurrency is two and total operation timeout is 60 seconds. Capture and delivery quotas are separate. Both source bytes and operation count are bounded. Source bytes remain reserved through cancellation cleanup and terminal persistence. Authority shutdown stops new admissions and drains owned transfers before releasing the database lock.

## vLLM Producer and Binary Format

Capture currently requires aggregated, single-choice, token-mode vLLM generation with exact prompt IDs. It rejects multimodal prompts, native-generate inputs, disaggregated capture, frontend-local string stops, full-vocabulary logprobs, and custom logprob token selection before inference. Enable vLLM routed-expert capture and provide the model's router profile rather than guessing router IDs. With vLLM 0.31, auxiliary capture also requires Model Runner V2 (`VLLM_USE_V2_MODEL_RUNNER=1`) and prefix caching. The native capture helper requires uniform top-k width, while the binary format supports heterogeneous per-router widths. Logprob values follow the worker's configured vLLM logprob mode; RL-enabled Dynamo workers require `processed_logprobs`.

The v1 format has a 26-byte prelude and a Zstandard-compressed record body. It records exact effective prompt and completion IDs. Each route record is a per-router `[T, K_r]` tensor with explicit router ID, expert count, and token start. Counts through 256 use unsigned 8-bit IDs; counts through 65536 use unsigned 16-bit IDs. Suffix capture with nonzero `token_start` is valid. Generated route extent excludes the last unconsumed generated token; prefill-only extent follows its separate bound. Selected logprobs are float32; top-k records pair candidate token IDs with float32 scores.

The codec validates geometry, range, dtype, component identity, framing, and bounded decompression. Transport sees immutable bytes, lengths, and digests only. A future SGLang or TensorRT-LLM adapter can supply its own serializer without teaching the Rust core its experience schema.

## Mooncake Foundation

The native `MooncakeByteRelay` adapter publishes scoped unique keys, verifies bounded readback, checks SHA-256, and cleans up only its owned keys after readers finish. It is independent of OpenDAL and can carry either artifact bytes or weight payloads. The optional native test suite exercises the actual Store client rather than an in-process dictionary.

This foundation does not implement public relay reader groups, lease renewal, consumer acknowledgement, or a live vLLM weight-update transaction. Transporting and installing GPU weight tensors in a separate model process validates the Store data plane, not the administrative update protocol or an inference-serving rollout.
