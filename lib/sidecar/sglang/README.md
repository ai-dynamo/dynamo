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

Native Dynamo `/generate` requests are forwarded opaquely to SGLang's HTTP
endpoint using the gRPC host and the HTTP port returned by `GetServerInfo`.
The sidecar advertises this capability only after the HTTP health probe passes
and discovery confirms `--incremental-streaming-output`; otherwise it continues
serving the native gRPC path without advertising `/generate`.

The sidecar discovers the model and tokenizer paths, served model name, parser defaults, worker role, context length, KV capacity, scheduler limits, data-parallel topology, and KV-event sources through SGLang's native discovery RPCs. Explicit Dynamo parser options override parser names discovered from SGLang.

SGLang remains the source of truth for the worker's aggregated, prefill, or decode role. The inherited `--disaggregation-mode` option and `DYN_DISAGGREGATION_MODE` environment variable have no effect in this sidecar. The SGLang sidecar rejects `--route-to-encoder` because its native protocol does not support encoder workers. Disaggregated workers continue to register under their fixed role components; aggregated workers honor `--component` or `DYN_COMPONENT`.

The full sidecar opens eight gRPC connections by default. Override the pool size with `--grpc-connections` or `DYN_SIDECAR_GRPC_CONNECTIONS`. Telemetry-only mode uses one metadata connection.

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
same mode dispatcher as the standalone executable.

## Node-local KV sidecars (multinode)

This opt-in path requires an SGLang build containing
[SGLang #39659](https://github.com/sgl-project/sglang/pull/39659)
(metadata contract at `c6c49ac1d2721b93becdfc2be7cab46ddf58c0e8`). It exposes
node-local `kv_event_sources` through `GetServerInfo`, including on followers
started with `--grpc-port`. Follower gRPC servers provide metadata only, not
an inference API.

Launch the engines and sidecars separately, with a sidecar colocated with each
node that owns KV publishers:

```bash
# Node 0: normal request serving and global DP registration.
python3 -m dynamo.sglang.sidecar --grpc-endpoint http://127.0.0.1:30001

# Follower node: local KV events only; no model or inference registration.
python3 -m dynamo.sglang.sidecar --grpc-endpoint http://127.0.0.1:30001 --telemetry-only
```

Both modes read the local engine's gRPC metadata. Telemetry-only mode calls
`GetServerInfo` without model discovery, inference, or health-check RPCs.
SGLang-managed follower sidecar launching is not implemented by this change;
the existing single-node `--sidecar dynamo.sglang.sidecar` path is unchanged.

Both modes consume `kv_event_sources` exactly as supplied: global `dp_rank`,
connectable `endpoint`, `topic`, and logical `block_size`. Dynamo does not
reconstruct local rank ranges or ports from `dp_size`. An explicit empty list
stays empty; a telemetry-only sidecar rejects it. A TP-only follower without a
local KV publisher should run just the engine, not a sidecar. Older leader
engines without this field retain legacy discovery, including its multinode
DP guard.

The leader advertises the existing `sglang_worker_group_id` derived from the
shared `dist_init_addr`, plus its KV block size and locally owned ranks in
runtime metadata. Followers use Dynamo's runtime-config discovery watch to
find that group's serving worker. Each follower retains its own publisher
identity, while its events carry the **leader's worker ID and the source's
global DP rank**. This associates cache locations with a routable target; it
does not send the events through the leader sidecar.

All sidecars in a group must use the same Dynamo namespace and endpoint.
Both modes discover the prefill/decode role from SGLang and use its fixed
component; aggregated workers must also share the same configured component.
Separate prefill/decode engine groups need distinct `dist_init_addr` values,
as in the in-process integration.

Followers may start before the leader; discovery is cancellable and bounded
by `--leader-discovery-timeout-secs` (default 1800). Missing, ambiguous, or
incompatible leader metadata produces an explicit startup error. If a running
follower loses its leader or local engine, or observes changed publishing
metadata, it exits. The external launcher must coordinate restarting the
distributed engine group and its sidecars, rather than restarting a sidecar
independently with stale router state.

This change adds **KV events only**, not forward-pass metrics or scheduler
load publishing. It reuses the existing advisory ZMQ relay: it does not add a
readiness handshake, lossless startup, or independent sidecar restart/replay.
`replay_endpoint` in engine metadata is not consumed by this relay.

### Externally managed multinode example

[launch/multinode_kv_router.sh](launch/multinode_kv_router.sh) starts one local
engine and one sidecar. Run it once per node, using the same model, topology,
namespace, and shared Dynamo discovery/event services (`ETCD_ENDPOINTS` and
`NATS_SERVER`). Start one Dynamo frontend separately:

```bash
python3 -m dynamo.frontend --router-mode kv

# Node 0 (replace 10.0.0.10 with its reachable address).
NODE_RANK=0 DIST_INIT_ADDR=10.0.0.10:5000 \
    bash lib/sidecar/sglang/launch/multinode_kv_router.sh

# Node 1, in a separate terminal on that host.
NODE_RANK=1 DIST_INIT_ADDR=10.0.0.10:5000 \
    bash lib/sidecar/sglang/launch/multinode_kv_router.sh
```

The default is two nodes with one GPU each: global TP=2, attention DP=2.
SGLang multinode DP requires `--enable-dp-attention`; it is not an arbitrary
collection of independent full-model replicas. For two eight-GPU nodes with
four attention-DP ranks per node, set `TP_SIZE=16 DP_SIZE=8` on both hosts.
The example is limited to layouts where every node owns a publisher.

For PD disaggregation, run a separate group with `ROLE=prefill` and another
with `ROLE=decode`, using distinct rendezvous addresses. Set
`SGLANG_BOOTSTRAP_HOST` to the prefill leader's reachable address on its node 0.
The script passes the role and NIXL transfer configuration to SGLang; sidecars
discover that role over gRPC. See `--help` for per-node port overrides and pass
additional engine/network options as script arguments. This example has not
been validated on a multinode GPU deployment.

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
