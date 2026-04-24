# DeepSeek-V4-Flash Reference Container

DeepSeek-V4-Flash is not in a stock vLLM release yet, so the recipe ships with its own reference Dockerfile that overlays the Dynamo runtime on top of the upstream dsv4 vLLM image.

- Base: [`vllm/vllm-openai:deepseekv4-cu130`](https://hub.docker.com/r/vllm/vllm-openai/tags) — vLLM from PR [#40760](https://github.com/vllm-project/vllm/pull/40760) (`zyongye/vllm:dsv4`) with the DeepSeek-V4 kernels, `tokenizer_mode`, tool + reasoning parsers, hybrid CSA + HCA attention, MTP speculative decoding, and the FP4 indexer.
- Overlay: pre-built Dynamo artifacts copied from the public `nvcr.io/nvidia/ai-dynamo/vllm-runtime` image — wheels, static `nats`/`etcd` binaries, NIXL, UCX, and the `dynamo.vllm` Python worker tree.

Both layers are Python 3.12; no vLLM reinstall is performed.

## Prerequisites

- Docker with BuildKit.
- Access to the public Dynamo vLLM runtime image (`nvcr.io/nvidia/ai-dynamo/vllm-runtime`) — `docker login nvcr.io` with an NGC API key if you haven't already.
- Push access to wherever you intend to store the built image (private NGC org, Artifactory, ECR, etc.).

## Build

Run from the **repo root**:

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
| `DYNAMO_SRC_IMAGE` | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.2` | Source image for the Dynamo overlay (wheels + worker + NIXL/UCX). Pin to a released tag (`1.0.2`, `1.0.2-cuda13`, etc.) for reproducible builds. |
| `DSV4_BASE_IMAGE` | `vllm/vllm-openai:deepseekv4-cu130` | The dsv4 vLLM base. The `cu129` tag is also available for CUDA 12.9 hosts. |

Example:

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

1. Installs the RDMA/UCX runtime deps on top of the dsv4 vLLM image (`libibverbs1`, `rdma-core`, `ibverbs-utils`, `libibumad3`, `libnuma1`, `librdmacm1`, `ibverbs-providers`, plus `ca-certificates`, `jq`, `curl`).
2. Applies a small upstream vLLM patch to the sparse attention indexer (drops the unsupported `topk=1024`). This will be removed once [vLLM PR #40760](https://github.com/vllm-project/vllm/pull/40760) lands in the base image.
3. Copies the static `nats-server` and `etcd` binaries from the Dynamo source image.
4. Copies UCX into `/usr/local/ucx` and NIXL into `/opt/nvidia/nvda_nixl`, with `LD_LIBRARY_PATH` set so NIXL's plugins resolve at runtime.
5. Installs the Dynamo Python wheels (`ai_dynamo_runtime`, `ai_dynamo`, NIXL Python bindings) into the dsv4 image's system Python 3.12.
6. Copies the `dynamo` Python package tree into `/workspace/components/src/dynamo` and puts it on `PYTHONPATH` so `python3 -m dynamo.vllm` resolves.
7. Keeps vLLM's FlashInfer sampler enabled (`VLLM_USE_FLASHINFER_SAMPLER=1`) and clears `ENTRYPOINT` so the Dynamo CRD operator's `command` / `args` take effect.

## Troubleshooting

- **`failed to resolve source metadata for nvcr.io/nvidia/ai-dynamo/...`** — run `docker login nvcr.io` with an NGC API key (username `$oauthtoken`, password is the API key). The `nvcr.io/nvidia/ai-dynamo/*` images are public but still require an authenticated pull.
- **`no matching manifest for linux/amd64`** — the dsv4 base is amd64-only today; build on an x86_64 host.
- **CUDA version mismatch on the host** — use `DSV4_BASE_IMAGE=vllm/vllm-openai:deepseekv4-cu129` if your node is still on CUDA 12.9.
- **NIXL plugins not found at runtime** — confirm `LD_LIBRARY_PATH` includes `/opt/nvidia/nvda_nixl/lib64/plugins` (set in the Dockerfile; don't unset it in the pod spec).
