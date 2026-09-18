# Building the Continuum KV hints container

The tested container consists of a Dynamo base built from the experiment branch and a Python-only vLLM overlay. The overlay does not replace vLLM native extensions.

## Tested revisions

| Component | Revision |
|---|---|
| Dynamo | `3c5a01b51370a01902744939a147cf99605579ca` |
| vLLM | `9b6e116be2d9efdd044df1d738bba5aabdfbbd56` |
| Final image | `nvcr.io/nvidian/dynamo-dev/karenc:dynamo-kv-hints-3c5a01b513-vllm-9b6e116be2` |
| Published digest | `sha256:26619aa378b001207f91b93d560a065d77a295eb7d87e71d471a98462889cb4f` |

Pull the tested image without rebuilding:

```bash
docker pull nvcr.io/nvidian/dynamo-dev/karenc:dynamo-kv-hints-3c5a01b513-vllm-9b6e116be2@sha256:26619aa378b001207f91b93d560a065d77a295eb7d87e71d471a98462889cb4f
```

## Prerequisites

The build requires Docker, Python 3, GitHub access, and permission to pull and push `nvcr.io/nvidian/dynamo-dev/karenc` images.

```bash
export BUILD_ROOT="${BUILD_ROOT:-$HOME/continuum-kv-hints-build}"
export DYNAMO_COMMIT=3c5a01b51370a01902744939a147cf99605579ca
export VLLM_COMMIT=9b6e116be2d9efdd044df1d738bba5aabdfbbd56
export DYNAMO_DIR="$BUILD_ROOT/dynamo"
export VLLM_DIR="$BUILD_ROOT/vllm"

mkdir -p "$BUILD_ROOT"
git clone https://github.com/ai-dynamo/dynamo.git "$DYNAMO_DIR"
git -C "$DYNAMO_DIR" fetch origin "$DYNAMO_COMMIT"
git -C "$DYNAMO_DIR" checkout --detach "$DYNAMO_COMMIT"
git clone https://github.com/karen-sy/vllm.git "$VLLM_DIR"
git -C "$VLLM_DIR" fetch origin "$VLLM_COMMIT"
git -C "$VLLM_DIR" checkout --detach "$VLLM_COMMIT"

test -z "$(git -C "$DYNAMO_DIR" status --porcelain)"
test -z "$(git -C "$VLLM_DIR" status --porcelain)"
```

## Native vLLM request path

The standard Dynamo vLLM runtime installs vLLM-Omni. When `VLLM_PLUGINS` is unset, importing the Omni plugin replaces `vllm.v1.request.Request` and drops this prototype's `kv_hints` field. The base template and vLLM overlay use this image-wide allowlist:

```bash
VLLM_PLUGINS=modelexpress,lora_filesystem_resolver,lora_hf_hub_resolver
```

## Build the Dynamo base

The overlay Dockerfile currently references the legacy intermediate tag `dynamo-kv-hints-55667792-vllm`. The tag is rebuilt from the pinned Dynamo commit below and is not the final image name.

```bash
cd "$DYNAMO_DIR"
python3 container/render.py --framework vllm --output-short-filename
patch container/rendered.Dockerfile experiments/continuum-kv-hints/rendered-dockerfile-experiments.patch

export DYNAMO_BASE_IMAGE=nvcr.io/nvidian/dynamo-dev/karenc:dynamo-kv-hints-55667792-vllm
docker build --build-arg ENABLE_MEDIA_FFMPEG=false -t "$DYNAMO_BASE_IMAGE" -f container/rendered.Dockerfile .
docker push "$DYNAMO_BASE_IMAGE"
```

`container/render.py` regenerates `container/rendered.Dockerfile`; apply the experiment patch after every render. The patch adds `COPY experiments/` to the build stages because this branch adds experiment crates to the Cargo workspace.

## Build the vLLM overlay

```bash
cd "$VLLM_DIR"
export DYNAMO_SHA=$(printf '%s' "$DYNAMO_COMMIT" | cut -c1-10)
export VLLM_SHA=$(printf '%s' "$VLLM_COMMIT" | cut -c1-10)
export IMAGE=nvcr.io/nvidian/dynamo-dev/karenc:dynamo-kv-hints-${DYNAMO_SHA}-vllm-${VLLM_SHA}

docker build -t "$IMAGE" -f "$DYNAMO_DIR/container/Dockerfile.vllm-kv-hints-patch" .
docker push "$IMAGE"
```

## Verify the image

```bash
docker inspect "$IMAGE" --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -Fx 'VLLM_PLUGINS=modelexpress,lora_filesystem_resolver,lora_hf_hub_resolver'
docker run --rm --entrypoint python3 "$IMAGE" -c 'import inspect; from vllm.v1.kv_hints import KvHintAction, KvHintsEnvelope; from vllm.v1.request import Request; assert "kv_hints" in inspect.signature(Request).parameters; assert KvHintsEnvelope(protocol_version="0.1", message_id="smoke", actions=[KvHintAction(action_id="a", action_type="kv.retain", action_version="0.1", payload={})])'
```

## vLLM overlay files

| File | Change |
|---|---|
| `v1/kv_hints/__init__.py` | KV hint exports |
| `v1/kv_hints/protocol.py` | Envelope and action types |
| `v1/kv_hints/actions.py` | Retain and evict parsing |
| `v1/core/retained_block_queue.py` | Retention lease queue |
| `v1/core/block_pool.py` | External-hash index, retained queue, and `BlockStored.session_id` |
| `v1/core/kv_cache_manager.py` | Request-completion retention and eviction |
| `v1/core/sched/scheduler.py` | Request-completion action timing |
| `v1/engine/__init__.py` | `EngineCoreRequest.kv_hints` |
| `v1/engine/async_llm.py` | Request propagation |
| `v1/engine/input_processor.py` | Request propagation |
| `v1/engine/llm_engine.py` | Request propagation |
| `v1/request.py` | `Request.kv_hints` |
| `v1/kv_offload/base.py` | `ReqContext.kv_hints` |
| `v1/kv_offload/tiering/kvcr/manager.py` | KVCR envelope integration |
| `distributed/kv_events.py` | `BlockStored.session_id` |
| `distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` | Connector propagation |
| `engine/protocol.py` | Engine client protocol field |
