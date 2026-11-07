---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Generation Artifact Delivery (Proposed)
subtitle: Proposed metadata_upload contract, backend-owned encoding, and delivery guarantees
---

This reference describes a proposed extension to NVIDIA Dynamo's existing `nvext.metadata_upload` contract. Dynamo would carry delivery controls and opaque artifact bytes; each backend would own capture and encoding.

> [!WARNING]
> Proposed, not available on main. This documentation can merge independently of the implementation. Examples describe the proposed API, not executable instructions for a released feature. No production-readiness claim or release version is implied.

## Compatibility and Scope

The existing SGLang `{ "url": "s3://bucket/prefix" }` request would remain valid, with its cumulative `meta_info` serialization and fsspec transport unchanged. An optional legacy `fallback_url` would attempt the same serialized bytes once at the fallback prefix after an ordinary primary failure; cancellation would not trigger fallback. That fallback extension is proposed rather than current main behavior.

An explicit `kind: "generation_artifact"` would opt into artifact capture and delivery through the same `nvext.metadata_upload` field. A separate `nvext.generation_artifact` request field is not proposed. The Python package would be `dynamo.artifacts`, not a separate deployable component or service.

| Responsibility | Proposed Owner |
|----------------|----------------|
| Request validation and receipt projection | Generic Rust protocol and frontend |
| Tensor capture and binary encoding | Backend adapter, initially vLLM |
| Immutable byte transport and operation lifecycle | Backend-independent Python artifact package |
| Managed blob storage | Native Python OpenDAL |
| Delegated exact HTTP PUT | Dedicated capability-aware HTTP transport, not OpenDAL |
| Legacy SGLang metadata upload | Existing Python fsspec uploader |

## Proposed Request Contract

| Artifact Field | Type | Constraint |
|----------------|------|------------|
| `kind` | string | `generation_artifact` |
| `contents` | string array | Unique bounded names; initial vLLM names: `moe_routes`, `selected_logprobs`, `topk_logprobs`. An empty array would still record exact sequence IDs. |
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

| Kind | Required Fields | Proposed Behavior |
|------|-----------------|-------------------|
| `managed_store` | `profile`, `object_key` | OpenDAL writes through a trusted server profile. Keys must be normalized, relative, and within configured prefixes. Conditional create must prevent replacement. |
| `presigned_http_put` | `url`, `required_headers`, `expires_at`, `max_bytes`, `object_id` | One exact PUT with an independently authorized read verifier. `verification_ref` would be optional at the protocol layer but required by the initial worker authority. |
| `mooncake_store` | `profile`, `consumer_group`, `retention_seconds` | Reserved destination for `relay_store`; public relay requests would remain rejected until a lease and consumer authority exists. |

Presigned `url` is a capability, not a prefix. The transport must preserve its escaped path and query, apply all required signed headers, and avoid appended filenames, redirects, proxy environment variables, and OpenDAL reinterpretation. Expiration would use RFC3339. `max_bytes` must be positive and at most 64 MiB. A signed `if-none-match: *` header would be required; signed content length must match the encoded bytes exactly. The authority must bind the exact PUT URL hash to an independently authorized read profile and object key.

The transport must enforce an exact host/port allowlist and connect-time address checks. Private addresses require explicit allowed networks; plain HTTP additionally requires an exact allowed authority. Receipts must never expose credentials, presigned URLs, or signed headers.

## Proposed Results and Operations

The final generation response would project the backend receipt into `nvext.metadata_upload` independently of `extra_fields`. Rust would not decode the tensor schema. Inline results would contain `kind`, `mode`, `format`, `contents`, `byte_count`, and `data_base64` for the complete binary artifact.

| State | Required Guarantee | Additional Fields |
|-------|--------------------|-------------------|
| `pending` | This worker owns source bytes and retains the encrypted destination grant after durable handoff | `operation_id`, `revision` |
| `ready` | Bounded readback confirms exact length and SHA-256; an available provider version is read explicitly | `actual_bytes`, `sha256`, `object_id`, optional `provider_identity` |
| `failed` | Capture or delivery cannot satisfy the readiness contract | `error_code` |

The proposed engine administrative route `artifact_operations` would accept `{ "operation_id": "ga_..." }` and query only the configured authority's operation records. It would not be a public HTTP endpoint or tenant authorization boundary; access would require an authenticated administrative network.

One worker and event loop would exclusively own each SQLite operation database, protected by a filesystem lock on its canonical path. Grants would remain encrypted while pending and be erased at terminal completion. Lost PUT acknowledgements would trigger exact-destination readback, not a second PUT. Restart recovery would mark lost volatile sources `failed/source_lost`; durable operation metadata does not make volatile payload bytes resumable or permit cross-worker takeover. Retention sweeping remains separate work.

## Proposed Operator Controls

Object delivery would require protected server configuration, native OpenDAL dependencies, and a persistent Fernet capability key from a secret manager. Inline-only delivery would not require storage dependencies. This page does not prescribe installation commands before implementation availability.

| Control | Proposed Default | Meaning |
|---------|------------------|---------|
| `DYNAMO_ARTIFACT_CONFIG` | Unset | Trusted JSON configuration file; object delivery disabled when unset |
| `DYNAMO_ARTIFACT_CAPABILITY_KEY` | Required with config | Persistent encryption key for pending grants |
| `DYNAMO_ARTIFACT_CAPTURE_BYTES` | `268435456` | Process-wide capture admission bytes, including source and scratch estimates |
| `DYNAMO_ARTIFACT_ROUTER_PROFILE` | `[]` | Trusted `[router_id, expert_count, top_k]` triples, required for route capture |
| Queued source capacity | `268435456` bytes | Separate delivery source quota |
| Pending operation capacity | `128` | Bound capability and task overhead independently of payload size |
| Delivery concurrency | `2` | Maximum owned concurrent transfers |
| Total operation timeout | `60` seconds | Bound transfer lifetime |

Trusted configuration would supply a fixed security-domain `owner`, dedicated persistent `operation_db`, named profiles, verifier bindings, and storage policy. Each managed profile would contain `service`, native OpenDAL `options`, `allowed_prefixes`, and `immutable: true`. Immutability certification would not configure the provider: provider policy and unique keys must enforce it. Requests must never supply storage credentials.

Storage policy would bound destination authorities and networks, single PUT bytes (64 MiB), upload timeout (60 seconds), and remaining presigned capability lifetime (3600 seconds). Capture and delivery quotas would be independent. Source reservations must survive cancellation cleanup until terminal persistence. Shutdown must stop new admissions and drain owned transfers before releasing the database lock.

## Proposed vLLM Producer and Binary Format

The initial producer would support aggregated, single-choice, token-mode vLLM generation with exact effective prompt IDs. It would reject multimodal prompts, native-generate inputs, disaggregated capture, frontend-local string stops, full-vocabulary logprobs, and custom logprob token selection before inference. Route capture would require enabled routed-expert recording and an explicit trusted router profile, not inferred router IDs. For vLLM 0.31 auxiliary capture, the proposed adapter requires Model Runner V2 (`VLLM_USE_V2_MODEL_RUNNER=1`) and prefix caching. Native capture currently requires uniform top-k width, while the binary format supports heterogeneous per-router widths. Logprobs would follow the configured vLLM logprob mode; RL-enabled Dynamo workers require `processed_logprobs`.

The proposed vLLM producer would emit `generation_artifact_v1`, not native vLLM serialization. The v1 format has a 26-byte prelude and a Zstandard-compressed record body containing exact effective prompt and completion IDs. Each route record is a per-router `[T, K_r]` tensor with explicit router ID, expert count, and `token_start`. Expert counts through 256 use unsigned 8-bit IDs; counts through 65536 use unsigned 16-bit IDs. Nonzero `token_start` is valid. Generated route extent excludes the last unconsumed generated token; prefill-only extent has its own bound. Selected logprobs are float32; top-k records pair candidate token IDs with float32 scores.

The codec must validate geometry, ranges, dtypes, component identity, framing, and bounded decompression. Transport would see immutable bytes, lengths, and digests only. Future SGLang or TensorRT-LLM adapters could use backend-specific serializers without teaching Rust their experience schemas.

## Proposed Mooncake Foundation

A native byte-relay adapter would publish scoped unique keys, verify bounded readback and SHA-256, and clean up only owned keys after readers finish. It would be independent of OpenDAL and could carry artifact or weight payloads. Its tests must exercise the actual Store client, not an in-process dictionary.

This foundation would not implement public relay reader groups, lease renewal, consumer acknowledgements, or a live vLLM weight-update transaction. Transporting and installing GPU tensors in a separate model process validates a Store data plane, not the inference-service update protocol. Public `relay_store` activation requires a separate lifecycle design and validation.
