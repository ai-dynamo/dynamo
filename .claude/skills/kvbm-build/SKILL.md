---
name: kvbm-build
description: Render and build the Dynamo vLLM+KVBM container image
user-invocable: true
disable-model-invocation: true
---

# Build KVBM Container

Render a Dockerfile from Jinja2 templates and build a Dynamo container with vLLM + KVBM support.

> **For local development on a dev box, prefer `/dynamo:kvbm:sandbox-venv` + `/dynamo:kvbm:maturin-dev`** — far faster iteration loop. This skill is the right choice when you need a hermetic image for CI, benchmarks, or multi-host deploys.

## Arguments

`/dynamo:kvbm:build [--target TARGET] [--cuda-version VER] [--tag TAG]`

- **--target** (default: `runtime`): `runtime`, `dev`, `local-dev`
- **--cuda-version** (default: `12.9`): `12.9` or `13.0`
- **--tag** (default: `dynamo:latest-vllm`): Custom image tag

## Step 1: Parse Arguments & Detect Architecture

Detect the host architecture:

```bash
ARCH=$(uname -m)
# Map to platform string used by render.py
if [ "$ARCH" = "x86_64" ]; then
    PLATFORM="amd64"
elif [ "$ARCH" = "aarch64" ]; then
    PLATFORM="arm64"
else
    echo "Unsupported architecture: $ARCH" && exit 1
fi
```

Defaults:
- target: `runtime`
- cuda-version: `12.9`
- platform: detected from `uname -m` (`amd64` or `arm64`)
- tag: `dynamo:latest-vllm` (append `-dev` for dev target)

For `local-dev` target, note that `--build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g)` will be needed at build time.

## Step 2: Check Prerequisites

```bash
docker --version
python -c "import jinja2, yaml; print('render deps OK')"
```

If jinja2/yaml not available: `pip install jinja2 pyyaml`

## Step 3: Render Dockerfile

```bash
python container/render.py \
    --framework vllm \
    --target <target> \
    --device cuda \
    --platform <platform> \
    --cuda-version <cuda-version>
```

This generates: `container/vllm-<target>-cuda<ver>-<platform>-rendered.Dockerfile`

Show the user the rendered filename and confirm they want to proceed with the build.

KVBM is enabled by default in `container/context.yaml` (`enable_kvbm: "true"`). No extra flags needed.

## Step 4: Build Image

**This takes ~1-2 hours.** Confirm with the user before proceeding.

```bash
docker build \
    -f container/vllm-<target>-cuda<ver>-<platform>-rendered.Dockerfile \
    -t <tag> \
    --build-arg MAX_JOBS=$(nproc) \
    .
```

For `local-dev` target, add:
```bash
    --build-arg USER_UID=$(id -u) \
    --build-arg USER_GID=$(id -g)
```

Stream the build output so the user can monitor progress.

## Step 5: Verify

After build completes:

```bash
# Check KVBM is importable
docker run --rm <tag> python -c "import kvbm; print(f'KVBM OK: {kvbm.__version__}')"

# Check vLLM is importable
docker run --rm <tag> python -c "import vllm; print(f'vLLM OK: {vllm.__version__}')"
```

Report success with the image tag, or diagnose any import failures.

## Step 6: Next Steps

Tell the user:

```
Image built: <tag>

Run validation tests:
  /dynamo:kvbm:run-validation --image <tag>

Run performance benchmarks:
  /dynamo:kvbm:run-perf vllm --image <tag>

Launch container interactively:
  container/run.sh --framework vllm --image <tag> --gpus all --mount-workspace -it
```

## Reference: Build Targets

| Target | Description | Use case |
|--------|-------------|----------|
| `runtime` | Production image (vLLM + Dynamo + KVBM) | Testing, benchmarks, deployment |
| `dev` | Runtime + debuggers, vim, tmux, rustup, cargo | Development, debugging |
| `local-dev` | Dev + user namespace matching | Local development with mounted workspace |

## Reference: Key Files

- `container/render.py` — Dockerfile renderer
- `container/context.yaml` — Build configuration (base images, versions, flags)
- `container/run.sh` — Container launcher with correct GPU/IPC/SHM flags
- `container/templates/wheel_builder.Dockerfile` — KVBM wheel compilation stage
- `container/templates/vllm_runtime.Dockerfile` — Final runtime image assembly
