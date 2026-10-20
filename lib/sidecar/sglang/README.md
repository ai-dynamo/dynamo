<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SGLang sidecar

> [!WARNING]
> **Experimental.** These deployment examples and the sidecar image
> are experimental and not yet packaged for distribution (the launcher module
> ships inside `ai-dynamo-runtime`). The manifests, flags, and behavior may change
> without notice.

`dynamo-sglang-sidecar` connects Dynamo's unified worker lifecycle to an
out-of-process SGLang engine through SGLang's native gRPC service. It is a
standalone Rust executable and is also compiled into `ai-dynamo-runtime` for
the importable `dynamo.sglang.sidecar` launcher.

## Run

Build and run it directly from the Dynamo workspace:

```bash
cargo build --release -p dynamo-sglang-sidecar
./target/release/dynamo-sglang-sidecar \
    --grpc-endpoint http://127.0.0.1:30001
```

There is no published image yet; see
[Build the image](../README.md#build-the-image), which produces one image
containing all three sidecar executables. Official packaging is deferred to a
follow-up.

Use `DYN_SIDECAR_GRPC_ENDPOINT` instead of `--grpc-endpoint` when the endpoint is provided through the environment.

Start SGLang with `--incremental-streaming-output`. The sidecar's gRPC streaming path expects each response to contain only new tokens; cumulative output would duplicate tokens and inflate completion-token counts. The sidecar checks `GetServerInfo` during discovery and rejects startup unless `incremental_streaming_output` is explicitly `true`. Unlike the in-process Python worker, the sidecar cannot set launch options on an already-running engine.

Native Dynamo `/generate` requests are forwarded opaquely to SGLang's HTTP endpoint using the gRPC host and the HTTP port returned by `GetServerInfo`. The sidecar advertises this capability only after HTTP discovery and its health probe succeed; otherwise it continues serving the native gRPC path without advertising `/generate`.

The sidecar discovers the model and tokenizer paths, served model name, parser defaults, worker role, context length, KV capacity, scheduler limits, data-parallel topology, and KV-event sources through SGLang's native discovery RPCs. Explicit Dynamo parser options override parser names discovered from SGLang.

The sidecar also subscribes to SGLang's engine-state stream. A new SGLang
process gets a new instance ID. When that ID changes, the sidecar removes and
restores its discovery record so Dynamo clears stale KV-routing state. By
default, the sidecar also removes the worker from discovery while generation
is paused. Set `--unregister-on-pause=false` or
`DYN_SGLANG_UNREGISTER_ON_PAUSE=false` to keep a paused worker in discovery.
SGLang's computed health controls discovery. The engine-state stream requires
`WatchEngineState` support; the older versions listed in the deployment examples
below are not sufficient unless they include that RPC.

## Controller-managed admission (opt-in)

Ordinary inference auto-registers a healthy, unpaused engine. To start outside
discovery until a controller approves the worker:

```bash
DYN_SYSTEM_PORT=8081 dynamo-sglang-sidecar \
    --grpc-endpoint http://127.0.0.1:30001 \
    --namespace training-run-a \
    --controller-managed
```

The equivalent environment variable is `DYN_SGLANG_CONTROLLER_MANAGED=true`.
This mode requires `--unregister-on-pause=true`. Healthy/resumed state alone
does not grant membership. Use the individual sidecar's system listener for
the [admission API](../../backend-common/ADMISSION.md), not the frontend or
SGLang HTTP port. That reference covers status/admit/withdraw, retries, restart
fencing, update ordering, and security.

Admission brackets a fresh `GetModelInfo` with state snapshots checking the
expected healthy, unpaused engine instance, and compares the live weight version
with the controller's expected version. The shared worker checks observations
before/after verification and is the only discovery writer. State observations
do not replace controller approval.

For Miles, pin a compatible SGLang build from `sglang-miles`, including
`WatchEngineState` and the required weight-update transport. Use the
pause/update/commit/resume boundary and wait for a matching observation:
version-only metadata changes may not emit state notifications.

### Optional policy-version taints

`--policy-version-taints` / `DYN_SGLANG_POLICY_VERSION_TAINTS=true`
independently enables version labels; it is not required for controller admission.
It also requires `--unregister-on-pause=true`. For namespace `training-run-a`
and version `17`, the label is `dynamo.policy/version=training-run-a:17`;
both components are form-URL-encoded. Use a unique namespace per training run.

The worker withdraws membership, updates labels via `update_model_taints`,
then registers when eligible. It preserves unrelated labels, clears policy
labels while withdrawn, and stays withdrawn on missing versions or update
failures. Manual full-set `/engine/update/model_taints` is disabled only
in this opt-in mode to prevent competing writers.

Admission is not a per-request fence or proof of tensor/TP-rank consistency.
Discovery does not drain requests or stop cached/direct callers. Taint-aware
routing can require labels through `nvext.routing_constraints.required_taints`,
but native `/generate` constraint projection and round-robin enforcement are
not implemented here.

SGLang remains the source of truth for the worker's aggregated, prefill, or decode role. The inherited `--disaggregation-mode` option and `DYN_DISAGGREGATION_MODE` environment variable have no effect in this sidecar. The SGLang sidecar rejects `--route-to-encoder` because its native protocol does not support encoder workers. Disaggregated workers continue to register under their fixed role components; aggregated workers honor `--component` or `DYN_COMPONENT`.

The sidecar opens eight gRPC connections by default. Override the pool size with `--grpc-connections` or `DYN_SIDECAR_GRPC_CONNECTIONS`.

Connection startup uses a 30-second timeout per attempt, a one-second retry and readiness interval, and a 30-minute deadline for establishing the full connection pool. Override them with `--grpc-connect-attempt-timeout-secs`, `--grpc-retry-interval-secs`, and `--grpc-startup-deadline-secs`, or with the corresponding `DYN_SIDECAR_GRPC_*` environment variables.

## SGLang-managed module contract

SGLang can load the Python entry point and supply the gRPC endpoint arguments:

```bash
python3 -m sglang.launch_server \
    <args> \
    --grpc-port 30001 \
    --incremental-streaming-output \
    --sidecar dynamo.sglang.sidecar
```

The entry point configures Dynamo logging when `main()` runs, then calls the
private `dynamo._core.backend._run_sglang_sidecar(argv)` binding. The binding
prepends the executable name expected by clap, releases the GIL, and runs the
same unified worker lifecycle as the standalone executable.

## Deploy on Kubernetes (quick start)

`deploy/agg.yaml` runs an aggregated deployment (a frontend plus one worker pod
that colocates the sidecar with an SGLang engine). `deploy/agg_kv_router.yaml`
runs two aggregated workers behind Dynamo's KV-aware router.
`deploy/disagg.yaml` runs disaggregated prefill/decode with NIXL KV transfer;
`deploy/disagg_kv_router.yaml` expands it to two workers per role and publishes
KV-cache events for exact routing.

There is no published sidecar image yet, so build and push the image from
`lib/sidecar/Dockerfile`. It contains all three engine-specific sidecar
executables; these manifests run `dynamo-sglang-sidecar` as the container
command.

> [!NOTE]
> The engine image must be a stock SGLang **v0.5.16+** build: the native gRPC
> server (`--grpc-port`) landed there. The KV-routing examples require
> **v0.5.18+** because the sidecar discovers their structured KV-event
> descriptor through `GetServerInfo`. They use `lmsysorg/sglang:v0.5.19`.

### Prerequisites

- A Kubernetes cluster (**v1.29+**, or v1.28 with the `SidecarContainers` feature
  gate) with the Dynamo operator and a GPU node (two GPUs for
  `agg_kv_router.yaml`; two or four GPUs plus an RDMA fabric for `disagg.yaml` or
  `disagg_kv_router.yaml`, respectively). The engine runs as a native sidecar
  (`initContainers` with `restartPolicy: Always`), which requires that version.
- `kubectl` set to that cluster, and a namespace to deploy into.
- A Hugging Face token for the model.
- A container registry you can push to and the cluster can pull from.

### 1. Build and push the sidecar image

Build and push the image to a registry your cluster can pull from:

```bash
docker buildx build --platform linux/amd64,linux/arm64 \
  -f lib/sidecar/Dockerfile \
  -t <your-registry>/dynamo-sidecar:1.3.0 --push .
```

See [Build the image](../README.md#build-the-image) for a single-architecture
build. These manifests set the container `command` to
`dynamo-sglang-sidecar`.

### 2. Point the manifest at your image

In the selected manifest under `deploy/`, set the `main` worker image to the one
you just pushed. If your registry is private, add `imagePullSecrets` to the
worker pod spec.

### 3. Create the Hugging Face token secret

```bash
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="$HF_TOKEN" -n <namespace>
```

### 4. Deploy

```bash
kubectl apply -f lib/sidecar/sglang/deploy/agg.yaml -n <namespace>
```

Wait for the worker pod to reach `2/2 Running`:

```bash
kubectl get pods -n <namespace> -w
```

### 5. Send a request

```bash
kubectl port-forward -n <namespace> svc/sglang-sidecar-agg-frontend 8000:8000 &

curl -s localhost:8000/v1/models | jq .

curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}' | jq .
```

### KV routing

The KV-routing manifests run multiple workers and configure each SGLang engine
to publish ZMQ KV-cache events on all pod interfaces. Each sidecar connects to
the engine over its pod IP and advertises that routable address to the frontend
for exact KV-aware routing. Restrict the unauthenticated gRPC and ZMQ ports with
NetworkPolicy.

```bash
# Aggregated: two workers, two GPUs.
kubectl apply -f lib/sidecar/sglang/deploy/agg_kv_router.yaml -n <namespace>

# Disaggregated: two prefill + two decode workers, four GPUs and RDMA.
kubectl apply -f lib/sidecar/sglang/deploy/disagg_kv_router.yaml -n <namespace>
```

After deploying one of the KV-routing manifests, port-forward its frontend:

```bash
# Aggregated.
kubectl port-forward -n <namespace> svc/sglang-sidecar-agg-kv-router-frontend 8000:8000

# Disaggregated.
kubectl port-forward -n <namespace> svc/sglang-sidecar-disagg-kv-router-frontend 8000:8000
```

### Disaggregated

`deploy/disagg.yaml` runs prefill and decode as separate worker pods that hand
off KV cache over a bootstrap server + NIXL. It needs multiple GPUs and an RDMA
fabric, and both worker pods must reach `2/2 Running`.
`deploy/disagg_kv_router.yaml` uses two replicas per role and enables exact KV
routing from all four event streams.
