# DeepSeek-V4 Reference Containers

Shared reference Dockerfiles for the DeepSeek-V4 family — used by both [`deepseek-v4-flash`](../deepseek-v4-flash/) and [`deepseek-v4-pro`](../deepseek-v4-pro/). Nothing in either image is recipe-specific; the model is selected at runtime via `--model` (vLLM) or `--model-path` (SGLang).

| Backend | Dockerfile | Base image | Hardware target |
|---------|-----------|-----------|-----------------|
| vLLM | [`vllm/Dockerfile.dsv4.vllm.b200`](vllm/Dockerfile.dsv4.vllm.b200) | `vllm/vllm-openai:deepseekv4-cu130` | B200 |
| SGLang | [`sglang/Dockerfile.dsv4.sglang.b200`](sglang/Dockerfile.dsv4.sglang.b200) | `lmsysorg/sglang:deepseek-v4-blackwell` | B200 |

For SGLang, NVIDIA also publishes the prebuilt image at `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.0-sglang-deepseek-v4-b200-dev.1`, which both `sglang/agg/deploy.yaml` manifests pull directly. **Most users do not need to build from source.**

## vLLM (`vllm/Dockerfile.dsv4.vllm.b200`)

Two-step build: a Dynamo vLLM runtime image as the donor, layered onto the upstream dsv4 vLLM base.

### Step 1 — Build the Dynamo vLLM runtime

From the **repo root**:

```bash
container/render.py --framework vllm --target runtime --output-short-filename
docker build -t dynamo:latest-vllm-runtime -f container/rendered.Dockerfile .
```

This produces the local tag `dynamo:latest-vllm-runtime`, which Step 2 expects as `DYNAMO_SRC_IMAGE`. See [`<repo_root>/container/README.md`](../../../container/README.md) for details and alternative tags.

### Step 2 — Build the dsv4 overlay

Still from the **repo root**:

```bash
docker build \
  -f recipes/deepseek-v4/container/vllm/Dockerfile.dsv4.vllm.b200 \
  -t <your-registry>/vllm-dsv4:<tag> \
  .
```

The Dockerfile takes nothing from the build context (everything comes from `FROM` / `COPY --from=`), so any context directory works — running from the repo root just keeps the `-f` path concise.

### Build args

| Arg | Default | Purpose |
|-----|---------|---------|
| `DYNAMO_SRC_IMAGE` | `dynamo:latest-vllm-runtime` | Source for the Dynamo overlay. The default matches the tag produced by Step 1. Override with a pinned released tag (e.g. `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.2`) for reproducible builds without rebuilding locally. |
| `DSV4_BASE_IMAGE`  | `vllm/vllm-openai:deepseekv4-cu130` | The dsv4 vLLM base. The `cu129` tag is also available for CUDA 12.9 hosts. |

### Wire into a recipe

Push:

```bash
docker push <your-registry>/vllm-dsv4:<tag>
```

Set the `image:` field (Frontend + decode worker) in the recipe's vLLM manifest, then follow the recipe's Quick Start:

- Flash → [`../../deepseek-v4-flash/vllm/agg/deploy.yaml`](../deepseek-v4-flash/vllm/agg/deploy.yaml) — see [Quick Start](../deepseek-v4-flash/README.md#quick-start).
- Pro → [`../../deepseek-v4-pro/vllm/agg/deploy.yaml`](../deepseek-v4-pro/vllm/agg/deploy.yaml) — see [Quick Start](../deepseek-v4-pro/README.md#quick-start).

## SGLang (`sglang/Dockerfile.dsv4.sglang.b200`)

Three-stage build: stage 1 builds V4-aware Dynamo wheels from the `release/deepseekv4` branch on `manylinux_2_28_x86_64`; stage 2 pins a Dynamo SGLang runtime donor; stage 3 layers them onto the upstream SGLang dsv4 base.

### Step 1 — Build the Dynamo SGLang runtime

From the **repo root**:

```bash
container/render.py --framework sglang --target runtime --output-short-filename
docker build -t dynamo:latest-sglang-runtime -f container/rendered.Dockerfile .
```

This produces the local tag `dynamo:latest-sglang-runtime`, which Step 2 expects as `DYNAMO_SRC_IMAGE` for the runtime artifacts (`nats`, `etcd`, UCX, NIXL).

### Step 2 — Build the dsv4 overlay

Still from the **repo root**:

```bash
docker build \
  -f recipes/deepseek-v4/container/sglang/Dockerfile.dsv4.sglang.b200 \
  -t <your-registry>/sglang-dsv4:<tag> \
  .
```

The Dockerfile clones `ai-dynamo/dynamo` at `release/deepseekv4` in stage 1, so the build context contents are not used — running from the repo root just keeps the `-f` path concise.

### Build args

| Arg | Default | Purpose |
|-----|---------|---------|
| `DYNAMO_SRC_IMAGE` | `dynamo:latest-sglang-runtime` | Source for the runtime artifacts (`nats`, `etcd`, UCX, NIXL). The default matches the tag produced by Step 1; override with a published Dynamo SGLang runtime tag for reproducible builds. |
| `DSV4_BASE_IMAGE`  | `lmsysorg/sglang:deepseek-v4-blackwell` | The DeepSeek-V4 SGLang base. Pin a digest if you need byte-stable rebuilds. |

### Wire into a recipe

Push:

```bash
docker push <your-registry>/sglang-dsv4:<tag>
```

Set the `image:` field (Frontend + decode worker) in the recipe's SGLang manifest, then follow the recipe's Quick Start:

- Flash → [`../../deepseek-v4-flash/sglang/agg/deploy.yaml`](../deepseek-v4-flash/sglang/agg/deploy.yaml) — see [Quick Start](../deepseek-v4-flash/README.md#quick-start).
- Pro → [`../../deepseek-v4-pro/sglang/agg/deploy.yaml`](../deepseek-v4-pro/sglang/agg/deploy.yaml) — see [Quick Start](../deepseek-v4-pro/README.md#quick-start).
