<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Cross-version component compatibility

Run real frontend and SGLang worker images independently, with an HTTP client
from this checkout. No checkout, wheel, or adapter is injected into either
component. This tests the component boundary used during upgrades; it does not
perform a Kubernetes rolling upgrade or measure routing capacity.

## Version matrix

For candidate release line N, run these five pairs for **each** scenario:

| Frontend | Worker |
| --- | --- |
| candidate | candidate |
| N-1 | candidate |
| candidate | N-1 |
| N-2 | candidate |
| candidate | N-2 |

`releases.json` selects one published patch per historical minor line. The
runner defaults N to the checkout's Cargo.toml version. Update this manifest
when cutting a minor release; missing entries fail, rather than reducing
coverage. The initial resolver requires two previous minors within the same
major. A major-version transition needs an explicitly reviewed window policy.
This samples the listed patch versions, not every historical patch.

Images are pulled once and pinned to their local content IDs before any run.
The report records input references, content IDs, registry digests, installed
component versions, and the model revisions. Models are pinned in the manifest
and copied into both containers at `/model`; this also works with DinD without
shared host bind paths. Both components run from `/tmp` with offline model
loading. Every pair/scenario has its own Docker network, etcd, and NATS. Only
one worker runs at a time, using GPU 0.

## Initial contracts

- **Embedding:** Qwen3-Embedding-0.6B; default encoding, explicit `float`, and
  a batch. Require JSON float arrays, finite values, expected dimensions,
  cardinality, and indices. An HTTP error or base64 string for a float request
  fails. The client never normalizes a string into a vector.
- **Chat Completions:** Qwen2.5-0.5B-Instruct, aggregated SGLang; unary,
  streaming, `max_tokens=1`, a repeated request, and `stop`. Check content,
  token limits, finish reasons, stream errors, and `[DONE]`. The stop string is
  derived from the prefix of the unary response and repeated at temperature 0;
  the stopped response must be empty with finish reason `stop`. This avoids
  assuming that a small model follows a particular instruction. It does assume
  deterministic greedy output for the same request on the same worker.

The stop case depends on a successful unary baseline; if that baseline fails,
its failure already fails the scenario. Remaining independent cases still run.
These are API/protocol checks, not numerical embedding parity or model-quality
benchmarks. V1 does not cover disaggregation, KV-aware routing, tools,
multimodal inference, other engines, or simultaneous mixed worker pools.

## Run locally or on a GPU runner

Requires Linux amd64, one NVIDIA GPU with at least 24 GiB VRAM, a driver
compatible with every selected CUDA image, and Docker with NVIDIA runtime.
The client and Docker daemon must share the network namespace (local Docker or
a DinD sidecar), so published loopback ports are reachable. Remote Docker over
TCP with a separate network namespace is not supported. Allow sufficient disk
for all runtime images and the two models. Registry authentication is external
to the runner; log in before invoking it if your candidate images are private.

```bash
python3 -m venv /tmp/n2-client
/tmp/n2-client/bin/pip install requests==2.32.5 huggingface-hub==0.34.4
/tmp/n2-client/bin/python -m unittest discover -s scripts/compatibility -v
/tmp/n2-client/bin/python scripts/compatibility/runner.py \
  --frontend-image YOUR_FRONTEND_IMAGE --worker-image YOUR_SGLANG_IMAGE \
  --output /tmp/n2-results
```

The output directory must not exist. Add `--plan` to print and validate the
matrix without Docker, Hugging Face access, or GPUs. Use `--config` to provide
another reviewed release/model manifest. To exercise the reported historical
embedding array/string incompatibility, explicitly target the 1.4 window:

```bash
/tmp/n2-client/bin/python scripts/compatibility/runner.py \
  --release-line 1.4 \
  --frontend-image nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.4.2 \
  --worker-image nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.4.2 \
  --output /tmp/n2-embedding-regression
```

That includes frontend 1.2.1 with worker 1.4.2. The default main window moves
with the development version, so it must not be used to imply continued
coverage of an older regression pair. There are no xfails or skips for known
protocol incompatibilities. This framework does not fix those incompatibilities.

## CI and evidence

`compatibility-contract-tests.yml` runs the CPU harness checks on relevant PRs.
`cross-version-compatibility.yml` is callable and manually dispatchable with
independent candidate image inputs and an optional release line. Nightly CI
calls it after the actual frontend and SGLang builds, using both SHA-tagged
runtime artifacts. It authenticates to ECR using the existing registry action.
A release workflow can call the same workflow with its release candidate images
and release line, with `secrets: inherit`. V1 does not change release promotion
gates. The GPU job defaults to `prod-tester-amd-gpu-v2`; `N2_GPU_RUNNER` can select
a dedicated compatible runner. Its timeout is 120 minutes.

Any startup, request, validation, or resource cleanup failure makes the run
fail. Cases and later pairs continue after a scenario failure. The artifact
includes `report.json`, raw HTTP requests/responses or SSE lines, container
logs/states, and installed versions; model weights are excluded. The report
separates startup failures from individual contract failures. Use the matching
candidate/candidate control to identify failures that are not specific to
version skew. For ambiguous historical failures, repeat the same scenario with
both candidate image inputs set to that historical release as a same-version
control before attributing the failure to N-2 compatibility.

To add another scenario, add its immutable model specification and request
cases/validators, then run it through the same version matrix. Additional
engines need their own explicit launcher and supported-version configuration;
keep image orchestration separate from HTTP assertions.
