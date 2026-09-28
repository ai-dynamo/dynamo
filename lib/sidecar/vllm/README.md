<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# vLLM sidecar

The sidecar and its mock server use the official
[`vllm-proto`](https://crates.io/crates/vllm-proto) crate for Rust gRPC bindings.
The version is pinned in the workspace `Cargo.toml` and released independently
of vLLM. Update it together with compatibility checks against the supported
vLLM runtime. Dynamo does not need copied schemas, a generation script, or Buf credentials.

> [!WARNING]
> **Experimental.** This sidecar and its deployment examples are experimental.
> The Python launcher ships in the `ai-dynamo` and `ai-dynamo-runtime` wheel
> pair, but the container image is not yet packaged for distribution. The
> manifests, flags, and behavior may change without notice.

`dynamo-vllm-sidecar` connects a Dynamo worker to vLLM's native gRPC services:

- `vllm.Inference` for generation
- `vllm.Control` for model and server discovery
- Standard gRPC health for startup readiness

It is a standalone Rust executable and is also compiled into
`ai-dynamo-runtime` for the importable `dynamo.vllm.sidecar` launcher.

## Supported

- Aggregated generation
- NIXL prefill/decode generation
- Encoder/prefill/decode generation in E+PD and E+P+D topologies
- Token and text requests through Dynamo preprocessing
- Sampling, stop conditions, structured output, logprobs, cache options, and priority
- Opaque `kv_transfer_params` handoff
- Data-parallel rank routing and KV-event source discovery
- Capability-gated RL pause/resume, sleep/wake, weight-transfer, and weight-version controls through native gRPC
- Image, video, and audio URL and data-URI inputs; cache UUIDs remain image-only
- Preprocessed image features with dense placeholders, plus sparse placeholders when the renderer preserves `is_embed`. Sparse layouts such as Nemotron-H Omni require a vLLM revision containing [vllm-project/vllm#54548](https://github.com/vllm-project/vllm/pull/54548) on Python renderer round-trips.
- Dynamic LoRA load, unload, list, discovery, and request selection when vLLM enables LoRA
- Opaque encoder-cache handoff through vLLM `ec_transfer_params`
- Multimodal-aware KV routing for images

Audio and video gRPC inputs are not available in vLLM `0.28.0`. They require a later vLLM release.

The sidecar does not support beam search, `n > 1`, or Dynamo tool-call and reasoning parsers. The sidecar does not support `input_audio`, `file://` media, `use_audio_in_video` or other `mm_processor_kwargs`, decoded RDMA media, UUID-only media, or audio/video cache UUIDs. Encoder disaggregation is image-only in this release. The Dynamo frontend accepts inline media through OpenAI-compatible `data:` URLs; it does not expose a separate raw-byte media variant. Parser defaults returned by Control are intentionally not advertised to the Dynamo frontend because the current inference protocol does not preserve all parser-related request semantics.

In prefill/decode deployments, both engines independently prepare the original media. Reusing only the prefill-expanded prompt IDs is insufficient because KV transfer does not carry model-specific multimodal position metadata.

The official `Qwen/Qwen3-ASR-1.7B` repository currently needs Rust-frontend-compatible tokenizer and config metadata (`tokenizer.json` and a top-level `vocab_size`). Dynamo's Rust chat renderer also does not yet insert the model-native audio placeholder for `audio_url` content parts; callers can supply those prompt token IDs through `nvext.token_data`. These are model-loading and request-rendering gaps rather than sidecar media-transport limitations.

### LoRA

LoRA management is exposed only when `DYN_LORA_ENABLED` permits it, vLLM advertises
`supports_lora`, and the server reports `max_loras > 0`.

The sidecar resolves `file://`, `hf://`, and `s3://` LoRA sources through Dynamo's shared
LoRA downloader. Local `file://` adapters pass through at their canonical absolute path,
S3 adapters use `DYN_LORA_PATH`, and Hugging Face adapters use an immutable snapshot under
the configured Hugging Face cache. vLLM and the sidecar must see every resolved directory at
the same absolute path, so mount all local paths and cache roots identically when they run in
separate containers. Set vLLM's `VLLM_RUNTIME_LORA_ALLOWED_PATH_PREFIXES` to a platform
path-list containing those shared roots, for example
`/shared/local-loras:/shared/dynamo-loras:/shared/huggingface` on Linux.

Three behaviors intentionally differ from the legacy Python vLLM worker, because the gRPC
control surface does not expose the primitives they need:

- **Adapter IDs are server-assigned and opaque.** The Python worker derives the ID
  deterministically from the adapter name; `LoadLora` assigns it instead. Dynamo always
  reports the ID returned by vLLM and never generates or infers one.
- **Prefill workers load adapters eagerly.** The Python worker can defer loading until a
  request arrives because it forwards an adapter path in its internal `LoRARequest`.
  `GenerateRequest` carries only `lora_name`, and vLLM rejects names it has not already
  loaded, so every worker that may receive the adapter must load it up front.
- **Hot swap is not supported.** Loading a name that is already loaded is idempotent
  and returns the existing ID, and `hot_swap` is reported as `false`. The gRPC API
  has no atomic adapter replacement. Its pause-and-clear operation affects the
  whole worker and is not coordinated with adapter lifecycle operations here.

Custom Python-only LoRA source schemes are not available in the sidecar implementation.

LoRA requests retain normal prefix caching. Use a new adapter name for different weights:
unloading an adapter does not invalidate KV cached under its name, so reusing that name
for different weights can reuse stale results. Safe same-name replacement is not supported.

LoRA lifecycle mutations are serialized per worker, including source resolution. Requests
using other loaded adapters can continue during a load or unload. Multiple workers can
publish the same adapter independently.

LoRA requires a vLLM build containing
[vllm-project/vllm#52840](https://github.com/vllm-project/vllm/pull/52840).
NIXL prefill/decode also requires the gRPC numeric-conversion fix in
[vllm-project/vllm#54814](https://github.com/vllm-project/vllm/pull/54814).

For local LoRA serving, use [`launch/agg_lora.sh`](launch/agg_lora.sh) or
[`launch/disagg_lora.sh`](launch/disagg_lora.sh). Both print adapter loading examples
and accept `--help` for GPU, port, and cache settings. Load each adapter on both
workers before sending prefill/decode traffic.

S3 downloads use your existing AWS configuration. To use a local MinIO server,
set its connection details explicitly (also applies to `disagg_lora.sh`):

```bash
AWS_ENDPOINT_URL_S3=http://localhost:9000 \
AWS_ACCESS_KEY_ID=minioadmin \
AWS_SECRET_ACCESS_KEY=minioadmin \
AWS_REGION=us-east-1 \
AWS_ALLOW_HTTP=true \
    lib/sidecar/vllm/launch/agg_lora.sh
```

The disaggregated launcher uses a private local IPC socket for KV events, with one
data-parallel rank per engine. Custom multi-rank deployments must set
`VLLM_PREFILL_KV_EVENT_ENDPOINT`, replacing `VLLM_PREFILL_KV_EVENT_PORT`.
For TCP, this vLLM publisher requires a wildcard bind address such as
`tcp://*:20081`; a concrete IP makes it connect instead of listen. Restrict access
to that port to trusted consumers because KV events contain request token IDs.

## Run

### Native Generate compatibility

`vllm-proto 0.3.0` does not include the native sampling JSON extension proposed in [vLLM #56421](https://github.com/vllm-project/vllm/pull/56421), so the sidecar projects typed controls into the gRPC request and advertises `vllm_inference_v1_generate`. During rolling upgrades, v1.4 frontends can supply the legacy `extra_args.vllm_tito.sampling_params` envelope; canonical typed fields take precedence when both are present. Requests that rely on distinctions proto 0.3 cannot represent, such as explicit `top_k=0`, `top_k=-1`, or `min_p=0`, fail explicitly instead of silently changing sampling behavior.

Prefill and encode use their canonical one-token request and do not apply decode sampling controls.

### Runtime compatibility

The Python `vllm` package and `vllm-rs` must come from compatible vLLM revisions. Do not combine a wheel from one nightly with a binary from another. The sidecar's `vllm-proto` dependency is pinned in the workspace `Cargo.toml`.

vLLM-Omni changes the engine response format, causing `vllm-rs` to reject
responses. The Dynamo vLLM runtime image provides a `vllm-rs` wrapper that
disables Omni by default. Use `vllm-rs` from `PATH` when starting the engine.

The wrapper enables only ModelExpress when installed; otherwise it disables
all plugins. An exported `VLLM_PLUGINS` overrides this default. The `dev` and
`local-dev` images do not install Omni and retain normal plugin discovery.

Start vLLM with its gRPC listener:

```bash
vllm-rs serve Qwen/Qwen3-0.6B --host 127.0.0.1 --grpc-port 50051
```

This listener is unauthenticated and plaintext. Keep colocated deployments on
loopback or a private interface. Remote access requires network controls or a
secure proxy.

Start the Dynamo worker explicitly:

```bash
dynamo-vllm-sidecar \
  --grpc-endpoint 127.0.0.1:50051
```

After installing `ai-dynamo`, the Python module runs the same native worker:

```bash
python -m dynamo.vllm.sidecar \
  --grpc-endpoint 127.0.0.1:50051
```

Use `DYN_SIDECAR_GRPC_ENDPOINT` instead of `--grpc-endpoint` when the endpoint is
provided through the environment.

### RL workflows

Start vLLM with the capabilities required by the workflow, then opt the sidecar into RL discovery. This example targets vLLM 0.28:

```bash
vllm-rs serve Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 8000 \
  --grpc-port 50051 \
  --enable-sleep-mode \
  --weight-transfer-config '{"backend":"nccl"}'

DYN_SYSTEM_PORT=8081 dynamo-vllm-sidecar \
  --grpc-endpoint 127.0.0.1:50051 \
  --vllm-http-endpoint http://rollout-0.rl.svc.cluster.local:8000 \
  --vllm-rl-world-size 1 \
  --enable-rl
```

For newer vLLM releases, use the same command without `--vllm-rl-world-size`; the nonzero value reported over gRPC is authoritative.

Replace `rollout-0.rl.svc.cluster.local` with a private address that the RL controller can route to. Binding vLLM to `0.0.0.0` exposes both its HTTP and gRPC listeners, so restrict both ports with host firewall rules, Kubernetes NetworkPolicy, or an equivalent trusted-network control. A colocated sidecar can continue to use loopback for `--grpc-endpoint`; the advertised HTTP URL must be routable from the controller, not merely from the worker.

`--enable-rl` (or `DYN_ENABLE_RL=true`) requires the Dynamo system server (`DYN_SYSTEM_PORT=0` or a positive port) and registers `dyn://<namespace>.<component>.rl`, which lets the Dynamo frontend discover this worker and its `/engine/control/*` and `/engine/update/*` routes through `/v1/rl/workers`. The sidecar advertises pause/resume, sleep-status, and weight-version controls when the vLLM server reports the RL gRPC API; mutating sleep/wake routes require `--enable-sleep-mode`, weight-transfer routes require `--weight-transfer-config`, and draft updates require speculative decoding support. The sidecar publishes `--vllm-http-endpoint` (or `VLLM_HTTP_ENDPOINT`) only as part of this RL worker metadata.

Native vLLM lifecycle and weight-update operations use the typed gRPC Control service and do not require `--vllm-http-endpoint`. Configure the HTTP base URL only when an RL framework needs a compatibility operation that is not represented by the typed service, such as a custom `worker_extension_cls` method invoked through `/collective_rpc`.

vLLM 0.28 does not report its engine world size over production gRPC or HTTP routes. When RL is enabled against that release, pass `--vllm-rl-world-size` (or `DYN_VLLM_RL_WORLD_SIZE`) with the total process count across tensor, pipeline, prefill-context, and data parallelism (`TP * PP * PCP * DP`); the example uses the default one-rank topology. Do not substitute decode-context parallelism for PCP because those settings do not have a one-to-one relationship. Newer vLLM releases report the per-data-parallel engine value over gRPC, so omit the compatibility option for them.

The HTTP value must be a controller-routable `http://` or `https://` base URL. Path prefixes are preserved, so a reverse proxy can advertise a value such as `https://rollout.example.internal/vllm-admin`; downstream clients append the compatibility route beneath that prefix. User information, query strings, and fragments are rejected. Do not place credentials or tokens in the URL.

The update request bodies match vLLM's RL HTTP schemas: `init_weight_transfer_engine` requires `{"init_info": {...}}`, `update_weights` requires `{"update_info": {...}}`, `finish_weight_update` accepts `{"weight_version": "..."}`, and `update_weight_version` requires `{"new_version": "..."}`. Weight tensors remain on the configured NCCL, IPC, or sparse-NCCL transport; only backend metadata crosses gRPC.

The RL endpoint, engine routes, and raw HTTP compatibility surface are administrative interfaces that can pause serving, release GPU memory, and replace model weights. The sidecar does not add HTTP authentication to the advertised URL. Enable these interfaces only on trusted request and system networks, or place the HTTP endpoint behind an authenticated private proxy without embedding credentials in the published URL.

The sidecar discovers `model_id`, the served name, context length, KV capacity, scheduler limits, data-parallel topology, and KV-event sources through `vllm.Control`. `model_id` must be readable locally or fetchable by Dynamo for tokenization and chat templates. Parser defaults are not advertised because the current inference protocol cannot preserve all parser-related request semantics.

For hybrid data parallelism, run one vLLM gRPC frontend and sidecar per node with `--data-parallel-hybrid-lb` and the node's local DP size and starting rank. Point each sidecar's `--grpc-endpoint` at its local frontend. This requires a vLLM build that reports local DP size.

For a multimodal model, the sidecar resolves the model's chat image-placeholder
token from its local or Hugging Face configuration and attaches it to every ZMQ
source. This lets backend events use the same canonical image hash as frontend
routing. If the model configuration or exact-routing prerequisites cannot be
resolved, inference remains available but falls back to ordinary text-prefix KV
routing.

The forwarded 64-hex vLLM media identifier contains Dynamo's 64-bit routing hash
and becomes part of vLLM's encoder and prefix-cache identity. Treat passed-through
HTTP URLs as immutable: changing the bytes behind one URL can reuse stale media
cache state. User-supplied media UUIDs retain their vLLM semantics and disable
exact Dynamo multimodal credit.

Aggregated serving is the default. The sidecar role is configured explicitly because the current Control API does not report it:

- Encoder: `--disaggregation-mode encode`
- E+PD downstream: aggregated mode plus `--route-to-encoder`
- E+P+D prefill: `--disaggregation-mode prefill --route-to-encoder`
- E+P+D decode: `--disaggregation-mode decode`

### Encoder disaggregation

Encoder disaggregation uses Dynamo's Encode worker discovery and routing contract. All media items in one request are sent together to one Encode worker; per-item fan-out is not supported. Text-only requests bypass Encode workers. If the encoder hop fails, the downstream request retains its original media and vLLM encodes it inline.

The encoder vLLM instance must use an EC producer connector and the aggregated or prefill instance must use the matching EC consumer connector. The sidecar treats the connector metadata as an opaque JSON object and carries it over the existing gRPC `KVCacheParameters.ec_transfer_params` and `FinishInfo.ec_transfer_params` fields. In E+P+D, decode's vLLM gRPC frontend uses that metadata with the original media description to reconstruct model-specific positions such as Qwen-VL mRoPE, then removes the EC parameters before EngineCore consumes the prefill KV handoff. Decode therefore uses NIXL without an EC connector and does not load the encoder embedding again. This path requires vLLM Rust frontend support for metadata-only remote-prefill decode from [vLLM #54814](https://github.com/vllm-project/vllm/pull/54814) or a later release containing it.

The local examples use `Qwen/Qwen2.5-VL-3B-Instruct` with vLLM's `ECExampleConnector` and a shared directory. The directory must be accessible at the same path from the producer and consumer. Each script creates and removes an isolated temporary directory unless `EC_SHARED_STORAGE_PATH` names a caller-managed directory:

In the filenames, `e_pd` means one Encode worker plus one aggregated Prefill/Decode worker, while `epd` means separate Encode, Prefill, and Decode workers.

```bash
# Encoder + aggregated prefill/decode (2 GPUs)
lib/sidecar/vllm/launch/disagg_multimodal_e_pd.sh

# Encoder + NIXL prefill + decode (3 GPUs)
lib/sidecar/vllm/launch/disagg_multimodal_epd.sh
```

The examples run one `vllm-rs` process and one Dynamo sidecar for each role. The encoder uses `--mm-encoder-only`, eager execution, and disabled prefix caching. The sidecar rejects media UUIDs that contain path separators, NUL bytes, or dot path components before forwarding them as connector keys. `ECExampleConnector` is a validation connector; production deployments should select an EC connector whose transport and storage semantics fit the deployment.

The sidecar opens eight gRPC connections by default. This avoided
connection-level throttling in high-concurrency sidecar tests. Override the
pool size with `--grpc-connections` or `DYN_SIDECAR_GRPC_CONNECTIONS`.

Connection startup uses a 30-second timeout per attempt, a one-second retry
interval, and a 30-minute deadline for establishing the full connection
pool. Override them with `--grpc-connect-attempt-timeout-secs`,
`--grpc-retry-interval-secs`, and `--grpc-startup-deadline-secs`, or with the
corresponding `DYN_SIDECAR_GRPC_*` environment variables.

Each request owns its response stream but borrows a channel from the shared pool. Aggregate and prefill cancellation drops only that request's stream. Decode cancellation first submits the decode request and retains its stream until the first output token or a response containing `finish_info`, so a NIXL receiver can complete and release the transferred KV; it then drops the stream. If the stream ends early, returns a gRPC error, or produces an invalid response after cancellation, the sidecar logs the failure and reports the request as cancelled. vLLM automatically aborts the corresponding engine request while the pooled HTTP/2 connection remains available to other requests. The sidecar does not call the Control `Abort` RPC.

## Test without vLLM or a GPU

Use the CPU-only `dynamo-vllm-mocker-server` to exercise the same Inference, Control, and health contracts:

```bash
cargo run -p dynamo-vllm-mocker --bin dynamo-vllm-mocker-server -- \
  --listen 127.0.0.1:50051 \
  --model mocker-model \
  --extra-engine-args '{"speedup_ratio":1000}'

cargo run -p dynamo-vllm-sidecar --bin dynamo-vllm-sidecar -- \
  --grpc-endpoint 127.0.0.1:50051
```

The mocker does not advertise RL capabilities; use a compatible vLLM server for RL route testing.

See [`../../mocker/servers/vllm/README.md`](../../mocker/servers/vllm/README.md)
for aggregated and prefill/decode examples, supported Mocker configuration,
and fidelity limits.

## Deploy on Kubernetes (quick start)

`deploy/agg.yaml` runs an aggregated deployment (a frontend plus one worker pod
that colocates the sidecar with a vLLM engine). `deploy/disagg.yaml` runs
disaggregated prefill/decode with NIXL KV transfer.

There is no published sidecar image yet, so build and push the image from
`lib/sidecar/Dockerfile`. It contains the vLLM, SGLang, and TensorRT-LLM
sidecar executables; these manifests run `dynamo-vllm-sidecar` as the container
command.

The sidecar waits for both the Control and Inference services through the standard gRPC health API before registering the worker. The deployment manifests retain lightweight socket probes for container lifecycle monitoring. The engine image must include a `vllm-rs` build compatible with the pinned `vllm-proto` crate.

The Dynamo vLLM runtime image exposes `vllm-rs` through the
[wrapper described above](#runtime-compatibility). On CPU and XPU, check that
the binary is available with `command -v vllm-rs`. The example manifests use
upstream vLLM images and locate the binary inside the Python package.

### Prerequisites

- A Kubernetes cluster (**v1.29+**, or v1.28 with the `SidecarContainers` feature
  gate) with the Dynamo operator and a GPU node (multiple GPUs plus an RDMA fabric
  for `disagg.yaml`). The engine runs as a native sidecar (`initContainers` with
  `restartPolicy: Always`), which requires that version.
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
`dynamo-vllm-sidecar`.

### 2. Point the manifest at your image

In `deploy/agg.yaml` (and `deploy/disagg.yaml`), set the `main` worker image to
the one you pushed. Add `imagePullSecrets` if your registry is private.

### 3. Create the Hugging Face token secret

```bash
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="$HF_TOKEN" -n <namespace>
```

### 4. Deploy

```bash
kubectl apply -f lib/sidecar/vllm/deploy/agg.yaml -n <namespace>
```

Wait for the worker pod to reach `2/2 Running`:

```bash
kubectl get pods -n <namespace> -w
```

### 5. Send a request

```bash
kubectl port-forward -n <namespace> svc/vllm-sidecar-agg-frontend 8000:8000 &

curl -s localhost:8000/v1/models | jq .

curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}' | jq .
```

### Disaggregated

`deploy/disagg.yaml` runs prefill and decode as separate worker pods with NIXL
KV transfer. It needs multiple GPUs and an RDMA fabric, and both worker pods
must reach `2/2 Running`. Apply it the same way and call the frontend as above.

## Packaging

There is no published sidecar image yet. See
[Build the image](../README.md#build-the-image). The image contains the vLLM,
SGLang, and TensorRT-LLM executables; each deployment sets its container
`command` to the one it needs. Official packaging is deferred to a follow-up
change.
