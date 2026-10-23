<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LPX discovery proof of concept

The allocation leader publishes deterministic placement through OpenEngine
`Control.GetServerInfo`. Its sidecar stores the complete snapshot in
`DynamoWorkerMetadata.spec.data.lpx_topologies`. Each Cyborg sidecar serves a
Unix socket from which its local workers acquire that snapshot once at startup.
No cross-pod topology RPC, topology watch, or model-card extension is used.

## Payload contract

The canonical schema is [topology.proto](proto/lpx/topology/v1/topology.proto).
`ServerInfo.extra["lpx_topology"]` is a protobuf `StringValue` containing standard,
padded RFC 4648 base64 of `TopologyPayload`. `schema_version` must equal `1`.
The same schema is vendored by the independent Cyborg consumer and Agent producer.

This required LPX metadata is an explicit PoC exception to OpenEngine's contract
that `extra` is optional and unsuitable for correctness dependencies. OpenEngine
protobuf definitions are unchanged. Model cards and KV-transfer metadata retain
their existing contracts.

The snapshot identifies the DGD UID, Grove allocation
`<PodCliqueScalingGroup name>-<replica index>`, and immutable build ID from the
compiled manifest. Compiler partition IDs identify execution entries or chain
members; execution entries carry the host count, control endpoint and optional
state-ingress endpoint. Producers emit a complete map in sorted ID order.
Consumers use the IDs and compiled execution order, never the record order.

The publisher validates the extension before atomically storing it. DWM lookup
reads the dedicated field independently of Pod readiness; ordinary inference
discovery retains its readiness requirements. Only allocation pod 0 publishes.
The DWM survives container restarts and remains bounded by its owner Pod's
lifetime. Deterministic Grove placement makes that snapshot usable without a
fresh Agent response. Connection readiness is checked separately by Cyborg.

`GetSnapshot(expected)` requires all three identity fields. Absence or discovery
unavailability returns `UNAVAILABLE`. Invalid payloads, wrong identities,
unsupported versions, and multiple authorities return `FAILED_PRECONDITION`.
The query deliberately includes wrong-build records so a build mismatch fails
immediately. No version downgrade or endpoint-file fallback occurs.

## Deployment

The operator example is
[the hybrid DGD fixture](../../../deploy/operator/internal/dynamo/lpx/testdata/from_dgd_yaml/node-local-v2-hybrid-topology.input.yaml).
Supply actual images and a compatible compiled model before applying it.
The deployment harness contains full Hydra/Cyborg aggregated and
vLLM-prefill/Cyborg-decode examples with digest-required image overrides.

For hybrid deployments, author a restartable init container named `runtime` in
both Agent and conductor roles. The operator adds:

| Container | Binding |
| --- | --- |
| Every Agent and Cyborg container | `LPX_DEPLOYMENT=<DGD UID>` |
| Agent main | `LPX_TOPOLOGY_SOCKET=/run/lpx-topology/agent.sock` |
| Agent sidecar | `DYN_LPX_TOPOLOGY_ROLE=publisher`, Grove pod-index rank |
| Cyborg main | `CYBORG_LPU_TOPOLOGY_SOCKET=/run/lpx-topology/topology.sock` |
| Cyborg sidecar, every rank | `DYN_LPX_TOPOLOGY_ROLE=consumer` |

Both containers share an `emptyDir` mounted at `/run/lpx-topology` and use the
DGD's Kubernetes discovery service account. Preserve the engine readiness probe.
The local snapshot server starts before waiting for engine initialization.
Non-serving inference ranks also run the local server.

Quasar `--serve` uses `publisher-and-serve` on its sidecar and sets
`LPX_TOPOLOGY_SOCKET` equal to `LPU_IE_OPENENGINE_SOCKET`. The same OpenEngine
socket carries metadata and inference. Only allocation pod 0 serves; other
sidecars stand by. Quasar retains its LP20, single-model, manifest-v2 limits.

Cyborg uses `CYBORG_LPU_TOPOLOGY_TIMEOUT` (default `5m`) only for snapshot
acquisition. Cancellation interrupts it; other startup and connection deadlines
are separate. Validated routes are pinned for the worker lifetime, including
pipeline reconnects. Explicit all-local execution skips acquisition. Standalone
endpoint-file mode remains available but cannot be combined with discovery.

## Operator rollout

The new operator supplies `LPX_MODEL_PATH` and explicit
`LPX_LOCAL_PARTITION_IDS`; Agents derive placement from the compiled manifest.
It creates no runtime ConfigMaps. Current-operator compatibility adapters read
legacy ConfigMaps at startup and convert their partition selections into the
same native configuration. They do not turn legacy endpoint files into topology.
Keep these adapters through rollout to supported deployments; the maintainer
decides their removal.

The current operator also needs authored topology bindings and socket volumes.
Name its Agent publisher `topology`, because that operator reserves `runtime`
for the serving role. The smoke harness applies these bindings when
`LPX_E2E_OPERATOR_CONTRACT=configmap`; the new operator uses `runtime-config`.
Speculative decoding is deferred: its existing conductor still depends on
legacy ConfigMaps and is outside this snapshot startup contract.

## Validation

Run the affected Rust tests with `cargo test -p dynamo-runtime --lib discovery::`
and `cargo test -p dynamo-lpx-sidecar`, plus the operator's `go test ./...` and
`make lint`. The producer integration test starts a real OpenEngine Unix server,
publishes through the actual sidecar, stops the Agent, and acquires the retained
snapshot without registering inference endpoints or models. Kubernetes discovery
tests prove the snapshot is visible before inference readiness.

Hardware acceptance additionally requires matching native images and deterministic
reference inference on both operator versions. CPU tests alone do not establish
GPU/LPU execution or KV handoff correctness.
