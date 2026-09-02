<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# llm-d Async image provenance

## Binary identity

The deployed container used a private POC registry. Its coordinates and image
digest are intentionally omitted from checked-in artifacts. The live pod's
`imageID` matched the recorded build output exactly. This proves which binary
was deployed; it does not by itself prove which dirty source tree produced it.

## Local BuildKit record

- Build reference: `lucidno9yuu3kule43czt2y66`
- Context: `.`
- Dockerfile: `Dockerfile`
- Platform: `linux/amd64`
- VCS repository: `https://github.com/llm-d/llm-d-async.git`
- VCS base revision: `877a068f9e22f6062c291ed36b2f8d14aa663a85`
- Started: 2026-08-28 09:20:36 America/New_York
- Duration: 1 minute 23 seconds
- Private-registry output identities: omitted from checked-in POC artifacts

## Source reconstruction

The current checkout has the same base revision and is the best available
reconstruction of the Go implementation used for the build. Reconstruction
identifiers are:

- Tracked worktree diff captured near build time:
  `36bfa4b96aa0897cfb748eacedb03cef6b3cae12a08efb8298093c560f48abc9`
- Current tracked Go-only binary diff:
  `26d06d9cda1a2cf27a9faeb7c42355e190e71bb6e3e12551206af92b2b12d423`
- `api/dispatch_control.go`:
  `e0d07938ca2d93ac6f50cb999b6f9ce5c6e8a01accda3118ba22e1c8bd5b898c`
- `api/dispatch_control_test.go`:
  `8ed45a14b7391392da12dadf5705d50f209741c8eee5ad2c7b5281807e5a513c`
- `pkg/redis/leased_rate_gate.go`:
  `621ece42935bf5ff67e495fe53d4b044d27dd1ae9cb76d43b8fdc591e0fc27ac`
- `pkg/redis/leased_rate_gate_test.go`:
  `dafc487e893cb2d9c21c7982bdb42139e3b8e905911183f46b562a92435f1f18`

The binary reports `version=dev`, `commit=unknown`, and `buildDate=unknown`.
Therefore the reconstruction above is strong local provenance but is not
cryptographically bound to the binary. A production image should embed the
source revision and a clean-tree or patch identity during the build.

## Validation tied to this implementation

The focused Go packages for API, pipeline, flow control, worker, metrics,
pub/sub, Redis, and server all passed. The tagged Redis pool-gate integration
test, `go vet`, Helm lint, and chart rendering also passed. The canonical live
run then dispatched exactly 100 attempts and completed all 100 successfully on
the identity-pinned pod.
