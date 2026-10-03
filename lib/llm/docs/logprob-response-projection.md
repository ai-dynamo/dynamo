<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Finite logprob response projection

This change carries probabilities already returned by the engine through the
Rust frontend and the Python vLLM chat processor. It does not synthesize missing
prompt probabilities, advertise complete vLLM compatibility, or add backend
extension admission. Public prompt counts retain the existing unsigned type;
full-vocabulary requests and selected-token-ID extensions are separate work.

## Prompt probabilities

When a finite root `prompt_logprobs` is requested and the engine returns the
payload, unary chat responses expose it at the root. Unary text-completion
responses expose it on each corresponding choice. An absent engine payload
remains absent. Requested malformed payloads fail instead of disappearing.

`internal_prompt_logprobs` is transport inside the frontend's stream-to-unary
aggregation, including the Python/Rust bridge. It is not a public response
field and is never serialized into client-facing SSE chunks. Synthetic chunks
clear it. Numeric token keys survive JSON and direct Python deserialization.

The existing `nvext.extra_fields` selection is independent: requesting the
legacy `nvext.prompt_logprobs` response does not require the new unary field,
and both can coexist. This change does not remove that legacy streaming
extension or promise that it avoids prompt-sized payloads.

The Rust completion envelope now owns typed choice wrappers. Rust constructors
can convert existing protocol choices/envelopes with `Into`; ordinary JSON
without prompt probabilities is unchanged.

## Generated probabilities

With chat `logprobs=true`, omitted `top_logprobs` means zero alternatives;
explicit null disables generated probability capture. Null presence survives
request serialization without adding a private wire field. Positive counts
bound alternatives. Sampled-token bytes describe decoded text even when the
displayed token is a token-ID placeholder.

Text-completion alternatives use token-to-probability maps, not chat token
objects. Offsets count characters, track each streamed choice independently,
and are rebuilt from returned token strings for unary aggregation.

Generated HTTP probabilities have a -9999 floor. Prompt probabilities preserve
finite values below that floor. The vLLM worker maps negative infinity to -9999
before JSON transport and rejects NaN/positive infinity without echoing values.
Prompt text reconstruction reuses the pinned vLLM tokenizer algorithm and must
be checked when that upstream dependency changes.

## Validation scope

The branch includes finite-count Rust projection, aggregation and null-schema
tests, Python processor tests and bounded comparisons with native vLLM projection
helpers. Native helper calls are not full HTTP/model-serving parity evidence.
See the review validation summary for exact executed commands and revisions.
Full-vocabulary safeguards, selected-ID admission, compatibility profiles,
mixed-release deployment campaigns and broad model coverage are excluded.
