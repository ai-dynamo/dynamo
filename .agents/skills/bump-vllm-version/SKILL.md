---
name: bump-vllm-version
description: Bump Dynamo to a newer upstream vLLM runtime image, update the matching dependency pins, and validate from image metadata through render checks and the vLLM test ladder.
---

# Bump vLLM Version

Use this skill when upgrading Dynamo to a newer vLLM release. Current Dynamo
main layers Dynamo-owned wheels onto upstream `vllm/vllm-openai` images; do not
use the legacy vLLM framework/source-build path for CUDA runtime bumps.

## Start Here

1. Work from a clean worktree based on latest `origin/main`; do not reuse a
   dirty development worktree for the bump.
2. Read the workspace/root instructions and any repo-local instructions that
   exist in the worktree.
3. Run a quick baseline/status pass:

```bash
git status --short --branch
python3 deploy/sanity_check.py
```

`deploy/sanity_check.py` may report missing runtime packages when run on the
host instead of inside a Dynamo runtime image. Record that as environment
context, not automatically as a regression.

## Gather Version Targets

Collect:

- Target vLLM version.
- CUDA runtime image tags for both supported CUDA families.
- vLLM-Omni policy: stable match, release candidate, keep previous, or skip.
- NIXL version required by the target vLLM KV connector path.
- Test-only dependency minimums from vLLM metadata.

Verify before editing:

```bash
python3 - <<'PY'
import json, urllib.request

for tag in ["v0.21.0", "v0.21.0-cu129"]:
    url = f"https://hub.docker.com/v2/repositories/vllm/vllm-openai/tags/{tag}"
    data = json.load(urllib.request.urlopen(url, timeout=20))
    print(tag, data.get("last_updated"), [(i.get("architecture"), i.get("os")) for i in data.get("images", [])])

for pkg, ver in [("vllm", "0.21.0"), ("nixl", "1.1.0"), ("vllm-omni", "0.21.0rc1")]:
    data = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/{ver}/json", timeout=20))
    print(pkg, ver, data["info"].get("requires_python"))
PY
```

Also check upstream vLLM `requirements/kv_connectors.txt` for the target tag.
For vLLM `0.21.0`, it requires `nixl >= 1.1.0`.

NIXL uses separate spellings for the Git ref and PyPI package version. For the
vLLM `0.21.0` bump, `container/context.yaml` uses the NIXL Git tag `v1.1.0`,
while `pyproject.toml` uses the Python package pin `nixl[cu12]==1.1.0`.

Confirm Dynamo's hwloc pin remains in sync with upstream NIXL:

```bash
curl -fsSL https://raw.githubusercontent.com/ai-dynamo/nixl/main/contrib/Dockerfile.manylinux \
  | rg -n "HWLOC_VERSION|Build latest hwloc" -C 2
```

## Files to Update

Update CUDA-only pins unless the task explicitly includes CPU or XPU:

- `container/context.yaml`
  - `vllm.cuda13.0.runtime_image_tag`: `v0.21.0`
  - `vllm.cuda12.9.runtime_image_tag`: `v0.21.0-cu129`
  - `vllm.nixl_ref`: target NIXL Git ref, e.g. `v1.1.0`
  - `vllm_omni_ref`: chosen vLLM-Omni ref
- `pyproject.toml`
  - `vllm[flashinfer,runai,otel]==<target>`
  - `nixl[cu12]==<target-nixl>`
- `container/deps/requirements.test.txt`
  - bump `mistral-common` if vLLM metadata requires a higher minimum.
- `container/templates/vllm_runtime.Dockerfile`
  - remove release-specific upstream-image workarounds once the new upstream
    image no longer needs them.
- `container/compliance/README.md` and current-version docs
  - update references that describe current default base images or main ToT
    backend versions. Do not rewrite historical release entries.

Do not reintroduce the legacy `vllm_framework` or `install_vllm.sh` source-build
flow for CUDA runtime images.

## vLLM-Omni Policy

`vllm-omni` often lags the core vLLM release. It is layered only for CUDA images
and installed with constraints so the upstream `vllm/vllm-openai` Python solve is
not replaced.

For the vLLM `0.21.0` bump, use:

```yaml
vllm_omni_ref: "v0.21.0rc1"
```

This is intentional because stable `vllm-omni==0.21.0` is not available on PyPI.
Keep core vLLM validation separate from Omni validation, and do not let Omni
block non-Omni compatibility work unless the PR scope says otherwise.

## Upstream Image Smoke Checks

Before broad pytest, inspect the upstream image behavior directly when Docker is
available:

```bash
docker run --rm --entrypoint python3 vllm/vllm-openai:v0.21.0 \
  -c 'import vllm, nixl, lmcache; print(vllm.__version__)'

docker run --rm --entrypoint python3 vllm/vllm-openai:v0.21.0-cu129 \
  - <<'PY'
import vllm
import nixl._api as api
import lmcache
print(vllm.__version__)
print(api.__file__)
assert "nixl_cu12" in api.__file__, api.__file__
PY
```

If LMCache linkage is broken in one upstream CUDA family, keep the LMCache xfail
scoped to that family and cite the observed `ldd` output in the PR. For the
vLLM `0.21.0` bump, CUDA 13 links against `libcudart.so.13` correctly, while
the CUDA 12.9 image's LMCache `c_ops` links against `libcudart.so.13` even
though the image provides CUDA 12.9.

## Render Checks

Render the vLLM matrix before building:

```bash
for cuda in 12.9 13.0; do
  for target in runtime dev local-dev; do
    for platform in linux/amd64 linux/arm64 linux/amd64,linux/arm64; do
      python3 container/render.py \
        --framework vllm \
        --target "$target" \
        --device cuda \
        --cuda-version "$cuda" \
        --platform "$platform" \
        --output-short-filename
    done
  done
done
```

Inspect rendered Dockerfiles for stale target-version workarounds, stale
runtime-image tags, and accidental dependency solves that would overwrite the
upstream vLLM stack.

## Test Ladder

Use this prefix for vLLM-heavy runs:

```bash
FLASHINFER_WORKSPACE_BASE=/tmp
```

Use `python3 -m pytest`, not bare `pytest`. For GPU serve tests, prefer the
VRAM-aware runner:

```bash
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/tests/serve/test_vllm.py \
  -m "vllm and gpu_1 and pre_merge" \
  --max-vram-gib=24 -n auto --dry-run --tb=short
```

Then run narrow to broad:

```bash
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/components/src/dynamo/vllm/tests/test_vllm_chat_message_utils.py --tb=short
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/components/src/dynamo/vllm/tests/test_vllm_renderer_api.py --tb=short
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/components/src/dynamo/vllm/tests/test_vllm_kv_events_api.py --tb=short
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/components/src/dynamo/vllm/tests/test_vllm_logging.py --tb=short
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/components/src/dynamo/vllm/tests/test_vllm_worker_factory.py --tb=short
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/components/src/dynamo/vllm/tests/test_vllm_engine_monitor_stats.py --tb=short
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/components/src/dynamo/vllm/tests/test_vllm_prompt_embeds.py --tb=short
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/components/src/dynamo/vllm/tests/test_vllm_sleep_wake_handlers.py --tb=short
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/components/src/dynamo/vllm/tests/test_vllm_unit.py --tb=short
```

Then:

```bash
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/components/src/dynamo/vllm/tests \
  --ignore=/workspace/components/src/dynamo/vllm/tests/omni/test_omni_handler.py --tb=short

FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/tests/frontend/test_vllm.py --tb=short
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/tests/frontend/test_prepost.py --tb=short
FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/tests/frontend/test_prepost_mistral.py --tb=short

FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/tests/kvbm_integration/test_kvbm_vllm_integration.py \
  -m "vllm and gpu_0 and integration" --tb=short

FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/components /workspace/tests \
  -m "vllm and unit" \
  --ignore=/workspace/target \
  --ignore=/workspace/components/src/dynamo/vllm/tests/omni/test_omni_handler.py \
  --tb=short

FLASHINFER_WORKSPACE_BASE=/tmp python3 -m pytest /workspace/tests/serve/test_vllm.py \
  -m "vllm and gpu_1 and pre_merge" \
  --max-vram-gib=24 -n auto -v --tb=short --durations=10
```

If the environment lacks Docker, CUDA, NVML, or the full runtime dependencies,
record the limitation and stop at static/render validation rather than turning
environment failures into code changes.
