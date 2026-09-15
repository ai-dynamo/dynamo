<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KV DC Relay Protocol

The Relay publishes endpoint-local KV pool state through the transport-neutral
[`RelayPublicationSource`](../../../publication/source.rs). The universal publisher owns snapshot
bootstrapping, contiguous deltas, bounded queues, and producer-generation fencing. Transport
adapters consume its state watches and canonical `PublicationFrame`s.

`dynamo_llm::kv_dc_relay::wan::grpc::protocol` provides the Protobuf/gRPC representation of that state.
It adds generated messages, client/server interfaces, and wire validation; it does not own
another publication hub or CKF mirror. Its Cuckoo Bucket Images v1 (CBI1) helpers re-export
the [publisher's transport-neutral codec](../../../publication/cbi1.rs).

## Interfaces and Documentation

- [Universal publication API](../../../publication.rs): state watches, pool streams, canonical frames,
  and transport-neutral errors.
- [Producer architecture](../../../docs/architecture.md): discovery inputs, pool lifecycle,
  publication, and serving topology.
- [Protobuf schema](relay.proto) and [gRPC contract](../../../docs/grpc-contract.md): the
  `dynamo.kvrelay.v1` adapter's RPCs, wire identity, compatibility, errors, and client rules.
- [Component usage](../../../../../../../components/src/dynamo/kv_dc_relay/README.md#usage):
  minimal startup and links to deployment and configuration documentation.

The standard build includes the protocol package and plaintext gRPC server; `--bind` enables
the listener. The independent `ckf-diagnostics` Cargo feature enables optional diagnostics.

## Rust Protocol Helpers

The [protocol module](../protocol.rs) exports:

| API | Purpose |
| --- | --- |
| `KvEventRelayClient` | Generated tonic client for Relay RPCs. |
| `KvEventRelay`, `KvEventRelayServer` | Generated service trait and server wrapper. |
| `ProducerKey::try_from(&identity)` | Validate and compare the explicit v1 wire producer key, excluding descriptor metadata. |
| `validate_protocol_envelope`, `validate_*` | Validate envelopes, identities, descriptors, query semantics, and topology entries. |
| `WireIdentityError::is_unsupported()` | Distinguish unsupported semantics from malformed known data. |
| `relay_error_reason(&status)` | Decode the machine-readable gRPC error trailer; return `None` when absent, unknown, or unspecified. |
| `FILE_DESCRIPTOR_SET` | Compiled descriptors used by gRPC reflection. |

[`wire::images`](wire/images.rs) re-exports `encode_snapshot_chunks`, `encode_delta`, `decode`,
and `SnapshotAssembly` from the universal publisher. The package supplies validation and codec
primitives, not a consumer state machine or routing policy.

## CBI1 Payload

CBI1 encodes absolute Cuckoo-filter (CKF) bucket images independently of Protobuf or gRPC.
`PublicationFrame::payload()` carries these bytes; the gRPC adapter forwards them unchanged in
`FilterUpdate.payload`. Identity and sequencing remain in the enclosing frame. The CBI1 header
repeats the CKF format and data-center dimension so a decoder can reject drift before applying
bucket words. All multi-byte integers are little-endian.

### Header

Every snapshot chunk and delta begins with this 48-byte header:

```text
[0..4]   magic: b"CBI1"
[4..6]   wire_version: u16       (=1)
[6..8]   flags: u16              (1=snapshot chunk, 2=delta)
[8]      fingerprint_bits: u8    (=16)
[9]      slots_per_bucket: u8    (=4)
[10..12] format_version: u16     (=1)
[12..20] seed: u64
[20..28] bucket_count: u64
[28..36] dc_id: u64
[36..44] epoch: u64
[44..48] body_checksum: u32      (low 32 bits of xxh3_64(body))
```

The decoder validates the magic, wire version, flags, checksum, filter format, and bucket bounds.
Before applying the decoded frame, a consumer also verifies that its `dc_id` matches the
enclosing producer identity. `bucket_count` must be a power of two in `2..=16777216`.

### Snapshot Chunks

A complete CKF lane is encoded as dense `u64` bucket words. Each chunk body is:

```text
[0..4]   chunk_index: u32
[4..8]   chunk_count: u32
[8..16]  bucket_offset: u64
[16..]   bucket words: u64[]
```

Each word packs four 16-bit slots, least-significant slot first; zero means empty.
For example, bytes `34 12 00 00 cd ab 01 00` encode slots `[0x1234, 0, 0xabcd, 1]`.
See the shared [CKF addressing](../../../../../../kv-router/src/indexer/cuckoo/addressing.rs)
for fingerprint and candidate-bucket derivation from a canonical sequence hash.

One chunk contains at most 512 Ki buckets, or 4 MiB of bucket words. `SnapshotAssembly` accepts only
ordered contiguous chunks from one epoch that cover the declared bucket count. Do not expose a
filter until the complete snapshot has been validated and installed.

### Deltas and Sequencing

A delta body contains absolute bucket images:

```text
[0..8]   base_epoch: u64
[8..12]  image_count: u32
for each image:
  [0..4]   bucket: u32
  [4..12]  absolute bucket word: u64
```

The delta body's `base_epoch` equals `PublicationFrame::base_sequence()`; the header's `epoch`
equals `PublicationFrame::sequence()`. Snapshot chunks also carry their snapshot's sequence in
the header. These are publication sequence values, not separate generation counters. The gRPC
adapter preserves them as `FilterUpdate.base_sequence` and `FilterUpdate.sequence`; its added
heartbeats contain no CBI1 payload and do not advance the sequence.

Absolute images are idempotent at the bucket level, but consumers must still enforce contiguous
stream sequences. The frame limit is 4194368 bytes: a delta contains at most 349525 images
(`48 + 12 + 12 * image_count` bytes). A larger publication becomes a chunked snapshot.
