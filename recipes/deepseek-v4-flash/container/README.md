# DeepSeek-V4-Flash Reference Container

DeepSeek-V4-Flash is not in a stock vLLM release yet, so the recipe ships with its own reference Dockerfile that overlays the Dynamo runtime on top of the upstream dsv4 vLLM image.

- **Base:** [`vllm/vllm-openai:deepseekv4-cu130`](https://hub.docker.com/r/vllm/vllm-openai/tags) — vLLM from PR [#40760](https://github.com/vllm-project/vllm/pull/40760) (`zyongye/vllm:dsv4`) with the DeepSeek-V4 kernels, `tokenizer_mode`, tool + reasoning parsers, hybrid CSA + HCA attention, MTP speculative decoding, and the FP4 indexer.
- **Overlay:** pre-built Dynamo artifacts (wheels, static `nats`/`etcd` binaries, NIXL, UCX, the `dynamo.vllm` Python worker) copied from a locally-built Dynamo vLLM runtime image.

Both layers use Python 3.12; no vLLM reinstall is performed.

## Build flow

Two Docker images are involved:

1. **Dynamo vLLM runtime** — built from this repo using the instructions in [`<repo_root>/container/README.md`](../../../container/README.md). This image contains the Dynamo Rust runtime, wheels, and the `dynamo.vllm` worker.
2. **DeepSeek-V4-Flash overlay** — built here, using the image from step 1 as the source stage (`DYNAMO_SRC_IMAGE`) and the upstream dsv4 vLLM image as the final base (`DSV4_BASE_IMAGE`).

## Step 1 — Build the Dynamo vLLM runtime

From the **repo root**, render and build the runtime image per [`container/README.md`](../../../container/README.md):

```bash
# From <repo_root>
container/render.py --framework vllm --target runtime --output-short-filename
docker build -t dynamo:latest-vllm-runtime -f container/rendered.Dockerfile .
```

This produces the local tag `dynamo:latest-vllm-runtime`, which is what Step 2 expects by default.

## Step 2 — Build the DeepSeek-V4-Flash overlay

Still from the **repo root**:

```bash
docker build \
  -f recipes/deepseek-v4-flash/container/Dockerfile.dsv4 \
  -t <your-registry>/vllm-dsv4:<tag> \
  .
```

The Dockerfile takes no files from the build context (everything comes from `FROM` / `COPY --from=`), so any context directory works — using the repo root keeps the `-f` path straightforward.

### Build args

Both can be overridden with `--build-arg`:

| Arg | Default | Purpose |
|-----|---------|---------|
| `DYNAMO_SRC_IMAGE` | `dynamo:latest-vllm-runtime` | Source image for the Dynamo overlay. The default matches the tag produced by Step 1. Override with a pinned released tag (e.g. `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.2`) for reproducible builds without rebuilding locally. |
| `DSV4_BASE_IMAGE` | `vllm/vllm-openai:deepseekv4-cu130` | The dsv4 vLLM base. The `cu129` tag is also available for CUDA 12.9 hosts. |

Example — pin the overlay source to a released Dynamo tag on a CUDA 12.9 host:

```bash
docker build \
  -f recipes/deepseek-v4-flash/container/Dockerfile.dsv4 \
  --build-arg DYNAMO_SRC_IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.2-cuda13 \
  --build-arg DSV4_BASE_IMAGE=vllm/vllm-openai:deepseekv4-cu129 \
  -t <your-registry>/vllm-dsv4:<tag> \
  .
```

## Push

```bash
docker push <your-registry>/vllm-dsv4:<tag>
```

## Wire into the recipe

Once the image is pushed, update the `image:` fields in
[`../vllm/agg/vllm-dgd.yaml`](../vllm/agg/vllm-dgd.yaml) (both the Frontend and the `VllmDecodeWorker`) to point at `<your-registry>/vllm-dsv4:<tag>`, then follow the recipe's [Quick Start](../README.md#quick-start) to deploy.

## What the Dockerfile does

1. Installs the RDMA / UCX runtime deps on top of the dsv4 vLLM image (`libibverbs1`, `rdma-core`, `ibverbs-utils`, `libibumad3`, `libnuma1`, `librdmacm1`, `ibverbs-providers`, plus `ca-certificates`, `jq`, `curl`).
2. Applies a small upstream vLLM patch to the sparse attention indexer (drops the unsupported `topk=1024`). Remove once [vLLM PR #40760](https://github.com/vllm-project/vllm/pull/40760) lands in the base image.
3. Copies the static `nats-server` and `etcd` binaries from the Dynamo runtime image.
4. Copies UCX into `/usr/local/ucx` and NIXL into `/opt/nvidia/nvda_nixl`, with `LD_LIBRARY_PATH` set so NIXL's plugins resolve at runtime.
5. Installs the Dynamo Python wheels (`ai_dynamo_runtime`, `ai_dynamo`, NIXL Python bindings) into the dsv4 image's system Python 3.12.
6. Copies the `dynamo` Python package tree into `/workspace/components/src/dynamo` and puts it on `PYTHONPATH` so `python3 -m dynamo.vllm` resolves.
7. Keeps vLLM's FlashInfer sampler enabled (`VLLM_USE_FLASHINFER_SAMPLER=1`) and clears `ENTRYPOINT` so the Dynamo CRD operator's `command` / `args` take effect.

## Troubleshooting

- **`pull access denied for dynamo:latest-vllm-runtime`** — Step 1 has not been run (or produced a different tag). Build the Dynamo vLLM runtime image locally per [`<repo_root>/container/README.md`](../../../container/README.md), or override `--build-arg DYNAMO_SRC_IMAGE=<your-image>`.
- **`no matching manifest for linux/amd64`** — the dsv4 base is amd64-only today; build on an x86_64 host.
- **CUDA version mismatch on the host** — use `DSV4_BASE_IMAGE=vllm/vllm-openai:deepseekv4-cu129` if your node is still on CUDA 12.9.
- **NIXL plugins not found at runtime** — confirm `LD_LIBRARY_PATH` includes `/opt/nvidia/nvda_nixl/lib64/plugins` (set in the Dockerfile; don't unset it in the pod spec).
