<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LPX topology protocol

`lpx/topology/v1/topology.proto` (package `lpx.topology.v1`) is the canonical
LPX partition-topology contract. The leader of an LPX Agent allocation serves
`TopologyService` as the allocation's only topology authority; the Dynamo LPX
sidecar relays its answers through Dynamo discovery to the sidecar in each
conductor pod, which serves the same service to the conductor. `build.rs`
generates the client and server from this file.

Repositories that implement or consume the service vendor a byte-identical copy
and record the Dynamo commit and the file's SHA-256. Never edit a vendored copy:
change this file and re-vendor it.

## Compatibility

- Changes within `v1` are additive: new fields, enum values, messages or RPCs.
- Never renumber or reuse a field or enum number. Reserve removed numbers and
  names.
- Any other change, such as a changed meaning, type or status code, needs a
  new package (`lpx.topology.v2`), served alongside `v1` until every consumer
  has moved.
- Format the file with the repository's pinned `clang-format` (16) and keep it
  clean under `buf lint` (STANDARD).
- Update `build.rs`'s `PROTOS` list when a file is added or removed.

## Golden encoding

`testdata/get_snapshot_response.txtpb` is a hand-written `GetSnapshotResponse`
in the protobuf text format. `testdata/get_snapshot_response.binpb` is its wire
encoding, produced by `protoc` rather than by the generated code, so the crate's
tests check the generated bindings against an independent encoder. Vendored
copies can reuse both files. After changing either the proto or the text file,
regenerate the encoding from this directory:

```bash
protoc --encode=lpx.topology.v1.GetSnapshotResponse lpx/topology/v1/topology.proto \
  < testdata/get_snapshot_response.txtpb > testdata/get_snapshot_response.binpb
```
