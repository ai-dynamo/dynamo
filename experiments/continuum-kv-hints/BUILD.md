# Building the Continuum KV Hints container

Two Docker images are needed: a Dynamo+vLLM base built from this branch, and a thin
patch layer that overlays the custom vLLM onto it.

## Image names

| Image | Tag |
|---|---|
| Dynamo base (this branch) | `nvcr.io/nvidian/dynamo-dev/karenc:dynamo-kv-hints-55667792-vllm` |
| Dynamo + vLLM image | `nvcr.io/nvidian/dynamo-dev/karenc:dynamo-kv-hints-<dynamo-sha>-vllm-<vllm-sha>` |

The image tag identifies the exact Dynamo and vLLM revisions used for the build.

## Native vLLM request path

The standard Dynamo vLLM runtime installs vLLM-Omni. When `VLLM_PLUGINS` is
unset, vLLM loads every general plugin, and importing the Omni plugin globally
patches the native `vllm.v1.request.Request`. This text-only experiment must use
the native request path so `kv_hints` reaches the scheduler. The base-image
template and vLLM overlay Dockerfile set the following image-wide allowlist:

```bash
VLLM_PLUGINS=modelexpress,lora_filesystem_resolver,lora_hf_hub_resolver
```

## Step 1 — Dynamo base image

```bash
cd /home/scratch.karenc_coreai/dynamo
git checkout karenc/continuum-kv-hints-poc

# Render the Dockerfile
python3 container/render.py --framework vllm --output-short-filename

# Apply the experiments/ patch — needed because this branch adds Cargo workspace
# members under experiments/ but the Dockerfile template only copies lib/ and
# components/. The patch adds COPY experiments/ at both build stages.
patch container/rendered.Dockerfile \
  experiments/continuum-kv-hints/rendered-dockerfile-experiments.patch

docker build \
  --build-arg ENABLE_MEDIA_FFMPEG=false \
  -t nvcr.io/nvidian/dynamo-dev/karenc:dynamo-kv-hints-55667792-vllm \
  -f container/rendered.Dockerfile \
  .

docker push nvcr.io/nvidian/dynamo-dev/karenc:dynamo-kv-hints-55667792-vllm
```

> **Note:** `render.py` regenerates `container/rendered.Dockerfile` and wipes any
> manual edits. Always re-apply the patch after re-running `render.py`.

## Step 2 — vLLM patch layer

The vLLM overlay Dockerfile ([`container/Dockerfile.vllm-kv-hints-patch`](../../container/Dockerfile.vllm-kv-hints-patch))
copies only the Python files changed by the feature branch — no C extensions are
touched. The build context is the vLLM checkout.

```bash
cd ~/vllm
git checkout karenc/kv-hints-g1-actions

DYNAMO_DIR=/home/scratch.karenc_coreai/dynamo
DYNAMO_SHA=$(git -C "$DYNAMO_DIR" rev-parse --short=10 karenc/continuum-kv-hints-poc)
VLLM_SHA=$(git rev-parse --short=10 HEAD)
IMAGE=nvcr.io/nvidian/dynamo-dev/karenc:dynamo-kv-hints-${DYNAMO_SHA}-vllm-${VLLM_SHA}
docker build \
  -t "$IMAGE" \
  -f "$DYNAMO_DIR/container/Dockerfile.vllm-kv-hints-patch" \
  .

docker push "$IMAGE"
```

### vLLM files patched

The overlay copies these files from `karenc/kv-hints-g1-actions` into the container's
`/usr/local/lib/python3.12/dist-packages/vllm/`:

| File | Change |
|---|---|
| `v1/kv_hints/__init__.py` | new — KV hint protocol types |
| `v1/kv_hints/protocol.py` | new — envelope/action dataclasses |
| `v1/kv_hints/actions.py` | new — evict/retain action parsing |
| `v1/core/retained_block_queue.py` | new — deferred-free block queue |
| `v1/core/block_pool.py` | modified — external hash index, retained queue integration, session_id on BlockStored |
| `v1/core/kv_cache_manager.py` | modified — apply_request_completion_retention/eviction hooks |
| `v1/core/sched/scheduler.py` | modified — call retention/eviction hooks on request completion |
| `v1/engine/__init__.py` | modified — kv_hints field on EngineCoreRequest |
| `v1/engine/async_llm.py` | modified — forward kv_hints from request |
| `v1/engine/input_processor.py` | modified — forward kv_hints from request |
| `v1/engine/llm_engine.py` | modified — forward kv_hints from request |
| `v1/request.py` | modified — kv_hints field on Request |
| `v1/kv_offload/base.py` | modified — kv_hints in ReqContext |
| `v1/kv_offload/tiering/kvcr/manager.py` | modified — KVCR hint contract rebased onto envelope |
| `distributed/kv_events.py` | modified — session_id on BlockStored (upstream PR #51381) |
| `distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` | modified — hints plumbing |
| `engine/protocol.py` | modified — kv_hints in EngineClient protocol |

## Why two stages?

vLLM's C extensions (`_C.abi3.so`, flash-attn, etc.) are compiled into the base
`vllm/vllm-openai:v0.29.0-ubuntu2404` image. The feature changes are pure Python, so
they can be layered on top without recompilation. The vLLM branch
(`karenc/kv-hints-g1-actions`) was cherry-picked onto v0.29.0 specifically to isolate
the feature diff from upstream-main changes that could break API compatibility with the
compiled extensions.
