# Custom Image Build Guide for DGDR Deployments

How to build, tag, and layer custom Dynamo images for DGDR-based deployments on a custom registry (e.g. Docker Hub `jont828/`).

## Image Naming and Derivation

The DGDR spec has a single `spec.image` field. The profiler uses this as its own container image and derives the other image names from it by **replacing the image name component** while preserving the registry prefix and tag.

The derivation logic lives in `components/src/dynamo/profiler/utils/profile_common.py` (`_replace_image_name`):

| Derived image | Name substitution |
|---|---|
| Planner | `dynamo-planner` |
| vLLM runtime | `vllm-runtime` |
| SGLang runtime | `sglang-runtime` |
| TRT-LLM runtime | `tensorrtllm-runtime` |

**Example:** Given `spec.image: jont828/dynamo-planner:main-short-names`:

| Component | Derived image |
|---|---|
| Profiling job (uses spec.image as-is) | `jont828/dynamo-planner:main-short-names` |
| Planner service | `jont828/dynamo-planner:main-short-names` |
| Backend runtime | `jont828/tensorrtllm-runtime:main-short-names` |
| Frontend service | `jont828/tensorrtllm-runtime:main-short-names` (set by `update_image`) |

**Key implication:** All images under the same registry prefix **must share the same tag** for the derivation to find them. If you push `dynamo-planner:foo` you must also push `tensorrtllm-runtime:foo` and `dynamo-frontend:foo` (if used separately).

> **Note:** The `spec.image` field is documented as the "frontend image" in the CRD, but in practice the profiling job container runs `python -m dynamo.profiler`, which is packaged in the `dynamo-planner` image. The non-planner services (frontend, workers) all get the backend runtime image via `update_image()`. So setting `spec.image` to the planner image works correctly for DGDR deployments.

## Known Build Issues

### EPP Image Not Found (Frontend Build)

The frontend Dockerfile includes the EPP (Endpoint Policy Processor) binary, pulled from a base image. The pre-rendered Dockerfiles checked into the repo may reference an old EPP version that has been removed from the staging registry:

```
ERROR: us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/epp:v0.5.1: not found
```

**Fixes (pick one):**

1. **Re-render the Dockerfile from `main`** — `context.yaml` on `main` has the updated EPP version (`v1.5.0-rc.2`):
   ```bash
   python container/render.py --framework dynamo --target frontend --cuda-version 12.9 --platform linux/amd64
   ```

2. **Use `registry.k8s.io`** — the old version still exists there. Edit the rendered Dockerfile's `EPP_IMAGE` ARG:
   ```
   # Change this:
   ARG EPP_IMAGE=us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/epp:v0.5.1
   # To this:
   ARG EPP_IMAGE=registry.k8s.io/gateway-api-inference-extension/epp:v0.5.1
   ```

3. **Pass it as a build arg**:
   ```bash
   docker build --build-arg EPP_IMAGE=registry.k8s.io/gateway-api-inference-extension/epp:v0.5.1 ...
   ```

### Transient NVCR Pull Failures

Pulls from `nvcr.io` occasionally fail with connection errors:
```
nvcr.io/nvidia/cuda-dl-base:25.06-cuda12.9-devel-ubuntu24.04: Unavailable: connection error
```
Simply retry the build — Docker layer caching means only the failed pull is retried.

### Pre-Rendered vs Re-Rendered Dockerfiles

Pre-rendered Dockerfiles checked into the repo (e.g. `container/dynamo-frontend-cuda12.9-amd64-rendered.Dockerfile`) may be stale if they were generated from an older branch. When in doubt, re-render from `main`:

```bash
python container/render.py --framework dynamo --target frontend --cuda-version 12.9 --platform linux/amd64
# This overwrites the rendered Dockerfile with current context.yaml values
```

## Building Images from Source

### Prerequisites

Images are built using a Jinja2-based Dockerfile renderer. From the repo root:

```bash
# Render a Dockerfile for a specific framework/target
python container/render.py \
    --framework <dynamo|vllm|sglang|trtllm> \
    --target <frontend|planner|runtime|dev> \
    --cuda-version <12.9|13.0|13.1> \
    --platform linux/amd64
```

### Building the Three Core Images

```bash
# 1. Frontend (CUDA 12.9)
python container/render.py --framework dynamo --target frontend --cuda-version 12.9 --platform linux/amd64
docker build -t jont828/dynamo-frontend:main -f container/dynamo-frontend-cuda12.9-amd64-rendered.Dockerfile .

# 2. Planner (CUDA 12.9)
python container/render.py --framework dynamo --target planner --cuda-version 12.9 --platform linux/amd64
docker build -t jont828/dynamo-planner:main -f container/dynamo-planner-cuda12.9-amd64-rendered.Dockerfile .

# 3. TRT-LLM Runtime (CUDA 13.1)
# Pre-rendered Dockerfile is checked in
docker build -t jont828/tensorrtllm-runtime:main -f container/trtllm-runtime-cuda13.1-amd64-rendered.Dockerfile .
```

Pre-rendered Dockerfiles are checked into the repo:
- `container/dynamo-frontend-cuda12.9-amd64-rendered.Dockerfile`
- `container/trtllm-runtime-cuda13.1-amd64-rendered.Dockerfile`

The planner Dockerfile needs to be rendered on demand (not checked in by default).

### Push to Registry

```bash
docker push jont828/dynamo-frontend:main
docker push jont828/dynamo-planner:main
docker push jont828/tensorrtllm-runtime:main
```

## Layering Changes Without Full Rebuilds

Full image builds are slow (especially the ~34GB TRT-LLM runtime). When iterating on fixes, you can **layer changes on top of existing images** to avoid rebuilding from scratch.

### Method: Cherry-pick and Rebuild Only Affected Images

1. **Start from a known-good base build** on `main`:
   ```bash
   git checkout main
   # Build all three images with :main tag (one-time cost)
   ```

2. **Cherry-pick the fix** onto your working tree:
   ```bash
   git cherry-pick <commit-sha> --no-commit
   ```

3. **Rebuild only the affected image** with a new tag:
   ```bash
   # Example: planner-only change (service name shortening)
   docker build -t jont828/dynamo-planner:main-short-names \
       -f container/dynamo-planner-cuda12.9-amd64-rendered.Dockerfile .
   ```

4. **Tag the unchanged images** to match:
   ```bash
   # Runtime and frontend didn't change, just retag
   docker tag jont828/tensorrtllm-runtime:main jont828/tensorrtllm-runtime:main-short-names
   docker tag jont828/dynamo-frontend:main jont828/dynamo-frontend:main-short-names
   ```

5. **Push all images with the new tag**:
   ```bash
   docker push jont828/dynamo-planner:main-short-names
   docker push jont828/tensorrtllm-runtime:main-short-names
   docker push jont828/dynamo-frontend:main-short-names
   ```

6. **Reset your working tree** so `main` stays clean:
   ```bash
   git reset HEAD --hard
   ```

### Why This Works

- The `dynamo-planner` image is lightweight (~8GB) and rebuilds in a few minutes.
- The `tensorrtllm-runtime` image is ~34GB and takes much longer to build. Retagging avoids the rebuild entirely.
- Docker layer caching means even when you do rebuild, only the layers affected by your code change are rebuilt (the Dynamo Python package install layer), not the base image or dependency layers.

### When You Must Rebuild the Runtime Image

You need to rebuild `tensorrtllm-runtime` (or `vllm-runtime`/`sglang-runtime`) when:
- The change touches Rust code in `lib/runtime/` or `lib/llm/` (Python bindings)
- The change modifies `components/src/dynamo/trtllm/` (backend-specific Python code)
- The change modifies dependencies in `pyproject.toml`

You do **not** need to rebuild the runtime when:
- The change is only in `components/src/dynamo/planner/` or `components/src/dynamo/profiler/` (planner image only)
- The change is only in `components/src/dynamo/frontend/` (frontend image only)
- The change is in deploy/operator Go code (operator image, separate build)

## Identifying What Changed Between Image Versions

The only Python package that differs between image builds from different branches is `ai-dynamo` itself. The base dependencies (TRT-LLM, torch, flashinfer) come from the base container image and don't change between branch builds.

To compare versions:
```bash
# Check ai-dynamo version
docker run --rm --entrypoint "" <image> bash -c \
    'grep "^Version" /opt/dynamo/venv/lib/python3.12/site-packages/ai_dynamo*.dist-info/METADATA'

# Check TRT-LLM version
docker run --rm --entrypoint "" <image> bash -c \
    'cat /opt/dynamo/venv/lib/python3.12/site-packages/tensorrt_llm/version.py | grep __version__'

# Check torch version
docker run --rm --entrypoint "" <image> bash -c \
    'grep "^Version" /opt/dynamo/venv/lib/python3.12/site-packages/torch-*.dist-info/METADATA'
```

## Example DGDR Spec

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: qwen3-235b-fp8-disagg
spec:
  model: Qwen/Qwen3-235B-A22B-FP8
  backend: trtllm
  searchStrategy: rapid
  autoApply: true

  # All images derived from this: dynamo-planner, tensorrtllm-runtime, dynamo-frontend
  image: jont828/dynamo-planner:main-short-names

  modelCache:
    pvcName: model-cache
    pvcMountPath: /home/dynamo/.cache/huggingface
    pvcModelPath: hub/models--Qwen--Qwen3-235B-A22B-FP8/snapshots/<snapshot-hash>

  # ... rest of spec
```
