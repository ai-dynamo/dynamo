# SGLang sidecar

> [!WARNING]
> **Experimental.** These deployment examples and the unified sidecar image
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
    --sglang-endpoint http://127.0.0.1:30001
```

There is no published image yet; see
[Build the image](../README.md#build-the-image) for the unified sidecar image.
Official packaging is deferred to a follow-up.

Use `SGLANG_GRPC_ENDPOINT` instead of `--sglang-endpoint` when the endpoint is provided through the environment.

The sidecar discovers the model and tokenizer paths, served model name, parser defaults, worker role, context length, KV capacity, scheduler limits, data-parallel topology, and KV-event sources through SGLang's native discovery RPCs. Explicit Dynamo parser options override parser names discovered from SGLang.

SGLang remains the source of truth for the worker's aggregated, prefill, or decode role. The inherited `--disaggregation-mode` option and `DYN_DISAGGREGATION_MODE` environment variable have no effect in this sidecar. The SGLang sidecar rejects `--route-to-encoder` because its native protocol does not support encoder workers. Disaggregated workers continue to register under their fixed role components; aggregated workers honor `--component` or `DYN_COMPONENT`.

The sidecar opens eight gRPC connections by default. Override the pool size with `--grpc-connections` or `DYN_SIDECAR_GRPC_CONNECTIONS`.

Connection startup uses a 30-second timeout per attempt, a one-second retry and readiness interval, and a five-minute deadline for establishing the full connection pool. Override them with `--grpc-connect-attempt-timeout-secs`, `--grpc-retry-interval-secs`, and `--grpc-startup-deadline-secs`, or with the corresponding `DYN_SIDECAR_GRPC_*` environment variables.

## SGLang-managed module contract

SGLang can load the Python entry point and supply the gRPC endpoint arguments:

```bash
python3 -m sglang.launch_server \
    <args> \
    --grpc-port 30001 \
    --sidecar dynamo.sglang.sidecar
```

The entry point configures Dynamo logging when `main()` runs, then calls the
private `dynamo._core.backend._run_sglang_sidecar(argv)` binding. The binding
prepends the executable name expected by clap, releases the GIL, and runs the
same unified worker lifecycle as the standalone executable.

## Deploy on Kubernetes (quick start)

`deploy/agg.yaml` runs an aggregated deployment (a frontend plus one worker pod
that colocates the sidecar with an SGLang engine). `deploy/disagg.yaml` runs
disaggregated prefill/decode with NIXL KV transfer.

There is no published sidecar image yet, so build and push the unified image
from `lib/sidecar/Dockerfile`. It contains all three engine-specific sidecar
executables; this deployment selects `dynamo-sglang-sidecar`.

> [!NOTE]
> The engine image must be a stock SGLang **v0.5.16+** build: the native gRPC
> server (`--grpc-port`) landed there. The deployment examples use
> `lmsysorg/sglang:v0.5.17`, matching Dynamo main's SGLang pin.

### Prerequisites

- A Kubernetes cluster (**v1.29+**, or v1.28 with the `SidecarContainers` feature
  gate) with the Dynamo operator and a GPU node (multiple GPUs plus an RDMA fabric
  for `disagg.yaml`). The engine runs as a native sidecar (`initContainers` with
  `restartPolicy: Always`), which requires that version.
- `kubectl` set to that cluster, and a namespace to deploy into.
- A Hugging Face token for the model.
- A container registry you can push to and the cluster can pull from.

### 1. Build and push the sidecar image

Follow [Build the image](../README.md#build-the-image) and push the result to
a registry your cluster can pull from. This deployment selects
`dynamo-sglang-sidecar`.

### 2. Point the manifest at your image

In `deploy/agg.yaml`, set the `main` worker image to the one you just pushed.
If your registry is private, add `imagePullSecrets` to the worker pod spec.

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

### Disaggregated

`deploy/disagg.yaml` runs prefill and decode as separate worker pods that hand
off KV cache over a bootstrap server + NIXL. It needs multiple GPUs and an RDMA
fabric, and both worker pods must reach `2/2 Running`.
