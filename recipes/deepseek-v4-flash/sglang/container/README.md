# DeepSeek-V4-Flash SGLang Reference Container

The DeepSeek-V4 SGLang stack is published on NGC as `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.2.0-sglang-deepseek-v4-b200-dev.1` and is pulled directly by `deepseek-v4-flash/sglang/agg/deploy.yaml`. **Most users do not need to build this image.**

This directory ships a reference `Dockerfile.dsv4-sglang` for users who want to rebuild the image from source — for example to pin a custom Dynamo branch or a different SGLang base. The Dockerfile is identical to the V4-Pro recipe's copy — DeepSeek-V4-Flash and DeepSeek-V4-Pro share the same SGLang dsv4 stack — but is duplicated here so each recipe is self-contained. The model is selected at runtime via `--model-path`.

- **Base:** `lmsysorg/sglang:deepseek-v4-blackwell` — SGLang's DeepSeek-V4 enablement build with the V4 kernels, MXFP4 MoE backend, and EAGLE MTP.
- **Overlay:** Dynamo wheels built from the `release/deepseekv4` branch (V4 tool / reasoning parsers), plus `nats`, `etcd`, UCX, and NIXL copied from a locally-built Dynamo SGLang runtime image.

## Build flow

Two source images feed the final overlay:

1. **Dynamo SGLang runtime** — built from this repo using the instructions in [`<repo_root>/container/README.md`](../../../../container/README.md). This produces `dynamo:latest-sglang-runtime`, which the overlay copies `nats`/`etcd`/UCX/NIXL out of.
2. **DeepSeek-V4 overlay** — built here. Stage 1 builds the V4-aware Dynamo wheels (cloned from `release/deepseekv4`) on `manylinux_2_28_x86_64`; stage 2 layers them plus the runtime artifacts onto the SGLang base.

## Step 1 — Build the Dynamo SGLang runtime

From the **repo root**:

```bash
# From <repo_root>
container/render.py --framework sglang --target runtime --output-short-filename
docker build -t dynamo:latest-sglang-runtime -f container/rendered.Dockerfile .
```

This produces the local tag `dynamo:latest-sglang-runtime`, which is what Step 2 expects by default.

## Step 2 — Build the DeepSeek-V4-Flash overlay

Still from the **repo root**:

```bash
docker build \
  -f recipes/deepseek-v4-flash/sglang/container/Dockerfile.dsv4-sglang \
  -t <your-registry>/sglang-dsv4:<tag> \
  .
```

The Dockerfile clones the Dynamo `release/deepseekv4` branch in stage 1, so the build context contents are not used — running from the repo root just keeps the `-f` path concise.

> If you have already built the dsv4 SGLang overlay for the V4-Pro recipe, you can reuse the same image tag here — there is nothing model-specific in the container.

### Build args

| Arg | Default | Purpose |
|-----|---------|---------|
| `DYNAMO_SRC_IMAGE` | `dynamo:latest-sglang-runtime` | Source image for the runtime artifacts (`nats`, `etcd`, UCX, NIXL). The default matches the tag produced by Step 1; override with a published Dynamo SGLang runtime tag for reproducible builds. |
| `DSV4_BASE_IMAGE`  | `lmsysorg/sglang:deepseek-v4-blackwell` | The DeepSeek-V4 SGLang base. Pin a digest if you need byte-stable rebuilds. |

## Push

```bash
docker push <your-registry>/sglang-dsv4:<tag>
```

## Wire into the recipe

Once the image is pushed, set the `image:` field (Frontend + decode worker) in [`../agg/deploy.yaml`](../agg/deploy.yaml) to point at `<your-registry>/sglang-dsv4:<tag>`, then follow the recipe's [Quick Start](../../README.md#quick-start) to deploy.

## What the Dockerfile does

1. **Stage 1 — `wheel_builder`** on `quay.io/pypa/manylinux_2_28_x86_64`: installs Rust, `protoc`, and `maturin`; clones `ai-dynamo/dynamo` at `release/deepseekv4`; builds the `ai_dynamo_runtime` and `ai_dynamo` wheels with the V4 tool/reasoning parsers.
2. **Stage 2 — `dynamo_src` donor**: pins `${DYNAMO_SRC_IMAGE}` so stage 3 can `COPY --from=` the runtime binaries.
3. **Stage 3 — final image** on `${DSV4_BASE_IMAGE}`: copies `nats-server` and `etcd` into `/usr/local/bin`, the UCX shared libs into the system `lib/x86_64-linux-gnu`, and NIXL into the system Python `dist-packages`. Reinstalls the V4-aware Dynamo wheels with `--force-reinstall --no-deps`, then sanity-checks the parser is registered:

   ```python
   from dynamo._core import get_tool_parser_names
   assert 'deepseek_v4' in get_tool_parser_names()
   ```

4. Copies the Dynamo Python component tree into `/workspace/components/src/dynamo` and prepends both `/workspace/sglang/python` and `/workspace/components/src` to `PYTHONPATH` so `python3 -m dynamo.sglang` resolves and the SGLang repo directory at `/workspace/sglang` does not shadow the Python package.
5. Sets the DeepGEMM JIT env vars (`SGLANG_JIT_DEEPGEMM_PRECOMPILE=0`, `SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1`) and clears `ENTRYPOINT` so the Dynamo CRD operator's `command` / `args` take effect.

## Troubleshooting

- **`pull access denied for dynamo:latest-sglang-runtime`** — Step 1 has not been run (or produced a different tag). Build the Dynamo SGLang runtime image locally per [`<repo_root>/container/README.md`](../../../../container/README.md), or override `--build-arg DYNAMO_SRC_IMAGE=<your-image>`.
- **`V4 parser missing!`** raised during build — stage 1's clone of `release/deepseekv4` is stale or the branch was force-pushed without the parser. Re-run with `--no-cache` to refresh the clone.
- **`no matching manifest for linux/amd64`** — the SGLang dsv4 base is amd64-only today; build on an x86_64 host.
- **NIXL plugins not found at runtime** — confirm the destination paths inside the container still exist (`/usr/local/lib/python3.12/dist-packages/nixl*`, `/usr/lib/x86_64-linux-gnu/ucx`); a base-image bump can move them.
