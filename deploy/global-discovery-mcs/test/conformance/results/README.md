<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Baseline run against PR #17615

For the subsequent API fixes, see the [post-fix comparison](17681-api-fixes.md):
all 15 required checks pass and the overall result improves to 19 passed,
9 failed, and none skipped. The original baseline below is preserved.

The complete, unmodified upstream MCS conformance suite ran on 2026-10-30
against the registry/controller implementation from
[`0fd4c96b4ebc5321e0ecd2adc9627cee48b1e3ba`](https://github.com/ai-dynamo/dynamo/commit/0fd4c96b4ebc5321e0ecd2adc9627cee48b1e3ba),
the head of [#17615](https://github.com/ai-dynamo/dynamo/pull/17615).
No production controller or registry behavior was changed for this run.

The suite ran all 28 specifications in 1115.268 seconds (18m35s):
**15 passed, 13 failed, 0 skipped or pending**. The test binary exited 1,
and `make conformance-test` exited 2. This implementation does not pass
required MCS conformance in this configuration.

| Upstream group | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Required | 12 | 3 | 0 |
| Optional | 3 | 10 | 0 |
| Total | 15 | 13 | 0 |

## Required failures

| Field on ServiceImport | Expected by upstream | Observed |
| --- | --- | --- |
| `status.endpointSliceObjects` | `Present` or `Absent` | Empty. |
| `spec.internalTrafficPolicy` | Exported Service's `Cluster` policy | `nil`. |
| `spec.trafficDistribution` | Exported Service's `PreferClose` value | `nil`. |

All three failures are explicit non-conformance entries in the upstream
report. They occurred while reading API objects, independently of DNS.

Required passing checks cover import creation/deletion, headless type and
IP behavior, ClusterSetIP allocation, session affinity, exported ports, port
and headlessness conflicts, and rejection of ExternalName exports. The
negative connectivity check also passes; with exported DNS unresolved, that
pass alone does not establish correct cross-cluster isolation.

## Optional failures and blocked coverage

Two optional specifications fail with explicit non-conformance because their
prerequisite `status.endpointSliceObjects` is not `Present`:

- Imported EndpointSlice labels and deletion after unexport.
- Per-hostname DNS for a headless StatefulSet.

Their later assertions are therefore not established by this run. Do not
interpret the EndpointSlice case as proof that the slice objects or labels
are missing, or the StatefulSet case as an independently observed DNS failure.

The other eight optional failures are four DNS cases (ClusterIP A/SRV and
headless A/SRV records) and four DNS-dependent connectivity cases. The Helm
installation does not provide a `clusterset.local` DNS implementation. The
suite queried that upstream domain unchanged and could not resolve the
exported Services. Ordinary `cluster.local` DNS and pod exec worked. The
connectivity failures cannot independently diagnose endpoint reachability or
port routing until the DNS prerequisite is available.

Three optional specifications pass: local Service DNS isolation, exported
label/annotation propagation, and conflicting exported labels/annotations.

## Report interpretation

[17615-v0.5.2.yaml](17615-v0.5.2.yaml) is the upstream-generated YAML report
with only an SPDX comment header added. Test entries and implementation
metadata are unchanged. Upstream distinguishes five explicit non-conformance
entries (the three required and two optional cases above) from eight
execution/assertion failures with conformance status unknown. In particular,
`conformant: true` alongside `failed: true` in that schema is **not a pass**;
use `passed`, `failed`, and `skipped` together.

The original `report.html`, Ginkgo `results.json`/`junit.xml`, complete suite
log, and controller/registry diagnostics remain in the local ignored
`bin/conformance/reports` directory. Future manually dispatched CI runs
upload these artifacts even on failure. Kubeconfigs are outside that directory.

## Reproduction environment

| Item | Value |
| --- | --- |
| Upstream root release | `kubernetes-sigs/mcs-api v0.5.2` |
| Upstream commit | `32672de8605f9719b9220992c75cb263ba6f3746` |
| Nested suite module | `sigs.k8s.io/mcs-api/conformance v0.5.1-0.20260713154219-32672de8605f` |
| Host | macOS, arm64, Docker Desktop |
| kind | `v0.33.0` |
| Kubernetes node | `kindest/node:v1.35.8@sha256:07b2536e30b803ed61d1677a79df6115f798ce64c80f9e22f6ed45afd09323c0` |
| Helm | `v3.21.0` |
| Registry | Memory backend, one replica in the hub cluster |
| Controllers | One per cluster; cluster IDs match kubeconfig cluster names |
| Clusters | `mcs-conformance-hub`, `mcs-conformance-a`; separate IPv4 pod/Service subnets |
| Runtime image | `global-discovery-mcs-conformance:e2e-31a3b7ac9535` |
| Image digest | `sha256:31a3b7ac953577db23e4fc88a135c748a44b8500c5ff0723886ec225fb1ce92f` |
| Ginkgo seed | `1793359964` |
| Start / end | `2026-10-30T12:32:44+01:00` / `2026-10-30T12:51:19+01:00` |

The run used `make conformance-setup` followed by `make conformance-test`.
It began before the runner's reporting metadata was finalized, so its report
records the verified implementation source SHA above as `version`. The
committed runner now records the deployed content-tagged image there and
writes the harness checkout revision separately to `environment.json`.

To repeat with fresh clusters from a checkout, see the
[conformance instructions](../README.md). This IPv4 memory-registry run does
not validate IPv6, DynamoDB behavior, multi-region consistency, or recovery
under partitions.
