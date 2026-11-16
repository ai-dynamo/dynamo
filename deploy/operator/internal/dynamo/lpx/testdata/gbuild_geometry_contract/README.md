<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GBuild geometry contract fixtures

These deployment contracts exercise GBuild topology resolution, manifest construction,
and Cap'n Proto serialization through Dynamo's offline acquisition, normalization,
LPX Request projection, and runtime partition mapping. `fixtures.json` records each
payload's SHA-256, source geometry, compiler partition IDs, and selected chain.

The fixtures use synthetic runtime inputs and the existing `opaque-v3-topology` test
label. Geometry comes from explicit topology metadata, independently of that label.
The associated execution and report roots are encoded but omitted here: Dynamo reads
only the deployment root on this path. These files contain no compiled programs or
weights and prove artifact compatibility, not runtime execution or physical placement.

| Fixture | Logical chips and source shape | LPX reservation and Agents |
| --- | --- | --- |
| `subnode8` | 8; `[8,1,1,1]` | `[16,1,1,1]`; 1 |
| `cross_hemi32` | 32; `[16,1,2,1]` | Same shape; 2 |
| `rack256` | 256; `[16,8,2,1]` | Same shape; 16 |
| `connected_chain` | IDs 7 then 8; 32 then 16 chips | 3; connector devices 16–31 → 0–15 |

The Go expectations are independent of the projection code. The tests preserve the
original artifact bytes and verify stable source snapshot identity through acquisition
and projection.

## Updating the fixtures

Normal Dynamo CI reads these checked-in files offline. To refresh them, resolve the
four cases through a compatible GBuild producer, use `opaque-v3-topology` for topology
labels in its synthetic inputs, and encode the deployment and associated execution
and report roots together. Retain only the deployment files and update their hashes
and expected geometry in `fixtures.json`. Keep producer-specific setup outside this
repository. Do not edit encoded bytes or hashes by hand.

Review the decoded fields and run `go test ./internal/dynamo/lpx -count=1` from
`deploy/operator`. Each payload is limited to 64 KiB; the four-file corpus is limited
to 256 KiB.
