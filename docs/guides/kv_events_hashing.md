<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
-->

# KV Events & Hashing Consistency

This guide explains how Dynamo computes and consumes KV cache block hashes, and how to ensure consistent hashing across engines, processes, and nodes.

## Canonical Hashing (Router)

- Algorithm: xxh3-64
- Seed: 1337
- Token encoding: u32 tokens serialized via little-endian `to_le_bytes`
- Scope: Computes "local block hashes" used by the router/indexer to match cached prefixes.

Reference implementations:
- Rust (primary): `lib/llm/src/kv_router/indexer.rs` (`compute_block_hash_for_seq`)
- Python binding: `dynamo._core.compute_block_hash_for_seq_py` (delegates to the Rust implementation)

Note:
- `kv_block_size` must be identical between the engine that publishes KV events and the router. A mismatch will yield different local block hashes and break prefix matching.

Reference test vector check:
- Tokens `[1,2,3,4]`, `kv_block_size=4` → `14643705804678351452`

## Engine Block IDs vs Router Hashes

- LocalBlockHash (router): Canonical value used for KV matching.
- ExternalSequenceBlockHash (engine): Engine-provided block identifiers to link parent/child and removals; MUST be deterministic within a deployment.

The router recomputes LocalBlockHash from tokens on ingest. If parent links or removals reference unknown ExternalSequenceBlockHash, the router logs a warning (or error if `DYN_KV_ENFORCE_ENGINE_HASH_STABILITY=1`).

## Engine Configuration Tips

The goal is to ensure that emitted KV events are deterministic across ranks/restarts.

General:
- Set `PYTHONHASHSEED=0` for Python processes to eliminate hash randomization.

vLLM:
- If your version supports it, set a deterministic prefix-caching algorithm, e.g. `--prefix-caching-algo sha256`.
- Keep `enable_prefix_caching=True` when emitting KV events.

SGLang:
- Ensure events use deterministic block IDs across processes. If applicable, set `PYTHONHASHSEED=0`.

TensorRT-LLM:
- Use a stable `--random-seed` where applicable and validate that KV event block IDs are deterministic across launches.

## Observability and Enforcement

- Warnings on router when parent link is missing or a removal refers to an unknown block id include remediation hints.
- Set `DYN_KV_ENFORCE_ENGINE_HASH_STABILITY=1` to promote these warnings to error-level logs. This does not abort processing; the router still skips the offending operation.

## Quick Self-Check

From Python:

```python
from dynamo._core import compute_block_hash_for_seq_py
assert compute_block_hash_for_seq_py([1,2,3,4], 4)[0] == 14643705804678351452
```

If this check fails across nodes, verify environment and engine flags per above.
This self‑check only validates the router’s canonical hashing path (known‑answer test); it does not validate that engine‑emitted block IDs are deterministic.
