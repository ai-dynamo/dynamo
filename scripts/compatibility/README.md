<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Cross-version component compatibility

Test a candidate frontend and SGLang worker against the previous two release
lines through an independent HTTP client. The candidate is the code being
validated: for a PR targeting 1.5, the historical release lines are 1.4 and 1.3.
This tests the frontend/worker protocol boundary used during upgrades; it does
not perform a Kubernetes rolling upgrade or measure routing capacity.

## Version matrix

Run all five pairs for both embedding and aggregated Chat Completions:

| Frontend | Worker |
| --- | --- |
| candidate | candidate |
| candidate | N-1 |
| N-1 | candidate |
| candidate | N-2 |
| N-2 | candidate |

The target release line defaults to the checkout's Cargo.toml version and can
be specified with `--release-line`. `releases.json` selects one published patch
per historical minor line (currently 1.4.2 and 1.3.1). Update the manifest as the
candidate advances. Missing historical entries fail rather than reducing
coverage. The resolver requires two previous minors within the same major; a
major transition needs an explicitly reviewed window policy. This samples the
listed patches, not every historical patch. There is no published-only matrix
or separate historical defect reproduction suite.

Both candidate images are mandatory. Images are pulled once and pinned to
local content IDs; the report records input references, IDs, registry digests,
installed component versions, and immutable model revisions. Both containers
receive the same model snapshot at `/model`, run from `/tmp`, and load models
offline. No current-checkout code or adapter is injected into either component.
Copying models into containers avoids shared-bind-path assumptions with DinD.
Each pair/scenario has its own Docker network, etcd and NATS. Only one worker
runs at a time, using GPU 0.

## Initial contracts

- **Embedding:** Qwen3-Embedding-0.6B; default encoding, explicit `float`, and
  batch inputs. Require finite JSON vectors, expected dimensions, cardinality
  and indices. HTTP errors and strings returned for float requests fail; the
  client does not normalize an incompatible response.
- **Chat Completions:** Qwen2.5-0.5B-Instruct; unary, streaming, `max_tokens=1`,
  and `stop`. Validate content, token limits, finish reasons,
  stream errors and `[DONE]`. Derive a stop string from the unary response's
  prefix and repeat at temperature 0; require an empty stopped response with
  finish reason `stop`. This assumes deterministic greedy output for the same
  request on the same worker, not a particular model-generated phrase.

The stop case depends on a successful unary baseline. If that fails, the
scenario already fails and remaining independent cases still run. There are
no xfails for known protocol incompatibilities. These are API/protocol checks,
not embedding numerical parity or model-quality benchmarks. V1 does not cover
disaggregation, KV-aware routing, tools, multimodal inference, other engines,
or simultaneous mixed worker pools.

## Run locally or on a GPU runner

Requires Linux amd64, one NVIDIA GPU with at least 24 GiB VRAM, a driver
compatible with every selected CUDA image, and Docker with NVIDIA runtime.
The client and daemon must share the network namespace (local Docker or a
DinD sidecar) so published loopback ports are reachable. Remote Docker with a
separate network namespace is unsupported. Allow disk for all runtime images
and models. Log in to private candidate registries before invoking the runner.

```bash
python3 -m venv /tmp/n2-client
/tmp/n2-client/bin/pip install requests==2.32.5 huggingface-hub==0.34.4
/tmp/n2-client/bin/python -m unittest discover -s scripts/compatibility -v
/tmp/n2-client/bin/python scripts/compatibility/runner.py \
  --frontend-image YOUR_FRONTEND_IMAGE --worker-image YOUR_SGLANG_IMAGE \
  --output /tmp/n2-results
```

The output directory must not exist. Add `--plan` to validate/print the matrix
without Docker or GPUs. Use `--config` for another reviewed release/model
manifest. The supplied images must correspond to the target release line.

## CI and evidence

`compatibility-contract-tests.yml` runs CPU harness tests on relevant PRs in
both normal and optimized (`python -O`) mode.
The GPU job lives in `pr.yaml`, on approved `pull-request/N` pushes, and waits
for both `frontend-build` and `sglang-build`. It passes the runtime images tagged
with that exact PR source SHA to `cross-version-compatibility.yml`; it never
substitutes published images for either candidate. Changes to the harness,
SGLang, frontend or shared core trigger both required component builds. The
GPU result participates in `dynamo-status-check`.

Nightly uses the same five-pair runner with its SHA-tagged nightly artifacts.
The reusable/manual workflow also accepts two candidate images and an optional
release line for release-candidate validation. V1 does not change release
promotion gates. CI authenticates to ECR via the existing registry action and
pulls historical images from NGC. The GPU lane defaults to
`prod-tester-amd-gpu-v2`; `N2_GPU_RUNNER` can select another compatible runner.
The job timeout is 120 minutes.

Any startup, request, validation or cleanup failure makes the job fail. Later
cases and pairs continue after a scenario failure. Artifacts contain the
report, raw requests/responses or SSE lines, container logs/states and installed
versions; model weights are excluded. Compare failures with the
candidate/candidate control before attributing them to version skew. A healthy
control is useful evidence but does not by itself rule out an issue specific
to a historical engine or its launch configuration.

Additional scenarios can reuse the version matrix and orchestration with their
own model specification and HTTP assertions. Additional engines need explicit
launchers and supported-version configuration.
