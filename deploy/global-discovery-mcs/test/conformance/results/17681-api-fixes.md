<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Conformance after the API fixes

The complete, unmodified upstream MCS suite ran on 2026-10-30 against
[`918d5228ab6dba417dc30e8c69c39df9eef615f4`](https://github.com/ai-dynamo/dynamo/commit/918d5228ab6dba417dc30e8c69c39df9eef615f4)
in [#17681](https://github.com/ai-dynamo/dynamo/pull/17681), stacked on
[#17615](https://github.com/ai-dynamo/dynamo/pull/17615). All 28 specifications
ran without selection filters or assertion changes in 1023.343 seconds
(17m03s), with **19 passed, 9 failed, and none skipped or pending**.

## Comparison with the original baseline

The [original baseline](README.md) and its generated report are preserved.

| Upstream group | Baseline passed / failed | After fixes passed / failed |
| --- | --- | --- |
| Required | 12 / 3 | 15 / 0 |
| Optional | 3 / 10 | 4 / 9 |
| Total | 15 / 13 | 19 / 9 |

All three formerly failing required checks now pass:

- `status.endpointSliceObjects` publishes `Present` or `Absent`.
- `spec.internalTrafficPolicy` matches the exported Service's value.
- `spec.trafficDistribution` matches the exported Service's value.

The optional EndpointSlice labels and unexport/deletion check now gets past
its former status-field prerequisite and passes. The headless StatefulSet
hostname check also gets past that prerequisite, finds the imported slice,
and then fails at its DNS lookup. It is not a newly introduced failure; the
baseline stopped earlier in that same specification.

## Remaining failures

The nine optional failures comprise five DNS cases (ClusterIP A/SRV,
headless A/SRV, and headless StatefulSet per-hostname A records) and four
DNS-dependent connectivity cases. This installation has no
`clusterset.local` DNS provider. The suite queries that upstream domain
unchanged; ordinary local `cluster.local` DNS passes.

These failures do not independently diagnose endpoint reachability or
per-port routing, because the prerequisite DNS lookup fails. Upstream
classifies all nine as execution/assertion failures with conformance status
unknown, rather than explicit non-conformance entries. A report entry with
`conformant: true` and `failed: true` is **not** a passing result.

The test binary exits 1 and `make conformance` exits 2. The full suite
therefore remains failing despite all required checks passing. This run
measures IPv4 with the memory registry; it does not establish IPv6,
DynamoDB, multi-region consistency, partition recovery, or complete MCS
conformance with DNS deployed.

## Regression validation

The race-enabled module unit/envtest suite passes at 95.7 percent internal
statement coverage. `TestImportTrafficPoliciesAndEndpointSliceStatus` uses
the real registry and API server and passes in 0.63 seconds. It verifies:

- Local/PreferClose propagation into ServiceImport and derived Service.
- An update to Cluster policy and removal of the owned distribution value.
- Omitted policy on ServiceImport and API-server Cluster defaulting on Service.
- Status correction on a new registry event when other status fields match.
- Absent status with no slices, and Present with only unready endpoints.

The pinned Kubernetes 1.30 envtest API server enables
`ServiceTrafficDistribution` so that it exercises the alpha field rather
than silently pruning it. The kind run uses Kubernetes 1.35.8.

## Reproduction and artifacts

| Item | Value |
| --- | --- |
| Implementation / harness commit | `918d5228ab6dba417dc30e8c69c39df9eef615f4` |
| Upstream root release / commit | `mcs-api v0.5.2` / `32672de8605f9719b9220992c75cb263ba6f3746` |
| Nested suite module | `sigs.k8s.io/mcs-api/conformance v0.5.1-0.20260713154219-32672de8605f` |
| Host / kind / Helm | macOS arm64, Docker Desktop / `v0.33.0` / `v3.21.0` |
| Kubernetes node | `kindest/node:v1.35.8@sha256:07b2536e30b803ed61d1677a79df6115f798ce64c80f9e22f6ed45afd09323c0` |
| Registry / controllers | Memory registry in hub; one controller per cluster |
| Runtime image | `global-discovery-mcs-conformance:e2e-0d629aebc7dc` |
| Image digest | `sha256:0d629aebc7dcac0b1d7924ae670e0d0d72ecbe3697f7296ccd20fb0a831493c3` |
| Ginkgo seed | `1793366913` |
| Start / end | `2026-10-30T14:28:33+01:00` / `2026-10-30T14:45:36+01:00` |

Command, from the module directory with the prerequisite tools on PATH:

```bash
CONFORMANCE_WORK_DIR="$PWD/bin/conformance-fixed" make conformance
```

[17681-api-fixes-v0.5.2.yaml](17681-api-fixes-v0.5.2.yaml) is the generated
upstream report with only an SPDX comment header added. Its entries and
metadata are unchanged. The original HTML, Ginkgo JSON/JUnit, suite log,
environment metadata and controller/registry diagnostics remain in the
ignored local `bin/conformance-fixed/reports` directory. Both dedicated
kind clusters were deleted after the run, while reports were retained.
Kubeconfigs are outside that reports directory. The GitHub workflow has
not been executed; these results come from the local run.
