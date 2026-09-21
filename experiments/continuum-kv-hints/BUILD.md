# Building the Continuum KV hints container

The tested container consists of a Dynamo image built from the experiment branch and a Python-only vLLM overlay. The overlay does not replace vLLM native extensions.

## Last validated image

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

## Build inputs

The build requires Docker, Python 3, GitHub access, and permission to pull and push the selected image repository. `DYNAMO_REF` may be a branch, tag, or commit containing this experiment and its container files. `VLLM_REF` must resolve to a vLLM `v0.29.0`-compatible patch payload; the known-good payload is `9b6e116be2`. Each ref is resolved to an immutable full commit before checkout and image naming.

```bash
export BUILD_ROOT="${BUILD_ROOT:-$HOME/continuum-kv-hints-build}"
export DYNAMO_REF="${DYNAMO_REF:-karenc/continuum-kv-hints-poc}"
export VLLM_REF="${VLLM_REF:-9b6e116be2d9efdd044df1d738bba5aabdfbbd56}"
case "$(uname -m)" in
  aarch64|arm64) HOST_ARCH=arm64 ;;
  x86_64|amd64) HOST_ARCH=amd64 ;;
  *) HOST_ARCH=$(uname -m) ;;
esac
export TARGET_ARCH="${TARGET_ARCH:-$HOST_ARCH}"
export IMAGE_REPOSITORY="${IMAGE_REPOSITORY:-nvcr.io/nvidian/dynamo-dev/karenc}"
export DYNAMO_DIR="$BUILD_ROOT/dynamo"
export VLLM_DIR="$BUILD_ROOT/vllm"

mkdir -p "$BUILD_ROOT"
git clone https://github.com/ai-dynamo/dynamo.git "$DYNAMO_DIR"
git -C "$DYNAMO_DIR" fetch origin "$DYNAMO_REF"
export DYNAMO_COMMIT=$(git -C "$DYNAMO_DIR" rev-parse FETCH_HEAD)
git -C "$DYNAMO_DIR" checkout --detach "$DYNAMO_COMMIT"
git clone https://github.com/karen-sy/vllm.git "$VLLM_DIR"
git -C "$VLLM_DIR" fetch origin "$VLLM_REF"
export VLLM_COMMIT=$(git -C "$VLLM_DIR" rev-parse FETCH_HEAD)
git -C "$VLLM_DIR" checkout --detach "$VLLM_COMMIT"

test -z "$(git -C "$DYNAMO_DIR" status --porcelain)"
test -z "$(git -C "$VLLM_DIR" status --porcelain)"

export DYNAMO_SHA=$(printf '%s' "$DYNAMO_COMMIT" | cut -c1-10)
export VLLM_SHA=$(printf '%s' "$VLLM_COMMIT" | cut -c1-10)
export DYNAMO_IMAGE="${IMAGE_REPOSITORY}:dynamo-kv-hints-${TARGET_ARCH}-${DYNAMO_SHA}-vllm-v0.29.0"
export IMAGE="${IMAGE_REPOSITORY}:dynamo-kv-hints-${TARGET_ARCH}-${DYNAMO_SHA}-vllm-${VLLM_SHA}"
```

Override `DYNAMO_REF`, `VLLM_REF`, `TARGET_ARCH`, or `IMAGE_REPOSITORY` before running this block to build another compatible revision or target. Use a full commit SHA for an immutable input; the default Dynamo branch is convenient for rebuilding its latest pushed tip.

## Native vLLM request path

The standard Dynamo vLLM runtime installs vLLM-Omni. When `VLLM_PLUGINS` is unset, importing the Omni plugin replaces `vllm.v1.request.Request` and drops this prototype's `kv_hints` field. The base template and vLLM overlay use this image-wide allowlist:

```bash
VLLM_PLUGINS=modelexpress,lora_filesystem_resolver,lora_hf_hub_resolver
```

## Build Dynamo from source

```bash
cd "$DYNAMO_DIR"
python3 container/render.py --framework vllm --output-short-filename
patch container/rendered.Dockerfile experiments/continuum-kv-hints/rendered-dockerfile-experiments.patch

docker build --build-arg ENABLE_MEDIA_FFMPEG=false -t "$DYNAMO_IMAGE" -f container/rendered.Dockerfile .
```

`container/render.py` regenerates `container/rendered.Dockerfile`; apply the experiment patch after every render. The patch adds `COPY experiments/` to the build stages because this branch adds experiment crates to the Cargo workspace.

## Build the vLLM overlay

```bash
cd "$VLLM_DIR"
docker build \
  --build-arg DYNAMO_IMAGE="$DYNAMO_IMAGE" \
  -t "$IMAGE" \
  -f "$DYNAMO_DIR/container/Dockerfile.vllm-kv-hints-patch" \
  .
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
