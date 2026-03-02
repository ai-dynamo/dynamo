# Create separate `dynamo-planner` Docker image for planner/profiler components

## Summary

- Creates a new `dynamo-planner` Docker image target that extends the dynamo runtime with planner/profiler-specific Python dependencies
- Moves 8 planner/profiler-only packages out of the shared `requirements.txt` into a dedicated `requirements.planner.txt`
- Adds proper `planner`/`profiler` pytest markers to all planner and profiler test files
- Updates DGD generation to derive the planner service image as `dynamo-planner` (instead of reusing the frontend image)

## Motivation

The planner and profiler components have heavy dependencies (`prophet`, `pmdarima`, `scikit-learn`, `filterpy`, `aiconfigurator[webapp]`, etc.) that are not needed by the frontend, backend runtime, or other framework images. Bundling them into every image increases build time and image size unnecessarily. This PR separates them into a dedicated `dynamo-planner` image so that:

1. Frontend and backend images remain lean
2. The planner k8s service runs from its own purpose-built image
3. Test markers clearly identify which tests belong to planner vs profiler

## Changes

### New files

- **`container/deps/requirements.planner.txt`** — Planner/profiler-only dependencies: `aiconfigurator[webapp]`, `filterpy`, `matplotlib`, `pmdarima`, `prometheus-api-client`, `prophet`, `scikit-learn`, `scipy`
- **`container/templates/planner.Dockerfile`** — Extends the `runtime` stage with planner-specific deps

### Modified files

- **`container/deps/requirements.txt`** — Removed the 8 packages listed above (moved to `requirements.planner.txt`)
- **`container/render.py`** — Added `"planner"` as a valid target under `framework=dynamo`
- **`container/Dockerfile.template`** — Added `{% elif target == "planner" %}` branch that chains: `dynamo_base` → `wheel_builder` → `dynamo_runtime` → `planner`
- **`components/src/dynamo/profiler/utils/profile_common.py`** — Added `PLANNER_IMAGE_NAME`, `_replace_image_name()` shared helper, `derive_planner_image()` function; refactored `derive_backend_image()` to use the shared helper
- **`components/src/dynamo/profiler/utils/dgd_generation.py`** — Planner service image now uses `derive_planner_image(dgdr.image)` instead of `dgdr.image`
- **`components/src/dynamo/profiler/utils/config.py`** — `update_image()` now skips services with `componentType == "planner"` to prevent backend config modifiers from overwriting the planner image
- **`tests/planner/test_replica_calculation.py`** — Added `pytest.mark.planner` to `pytestmark`
- **`tests/planner/test_scaling_e2e.py`** — Added `pytestmark = [pytest.mark.planner]` (had no markers)
- **`tests/profiler/test_helpers_thorough.py`** — Added `pytestmark = [pytest.mark.profiler]`
- **`tests/profiler/test_helpers_profile_sla.py`** — Added `pytestmark = [pytest.mark.profiler]`
- **`tests/profiler/test_helpers_rapid.py`** — Added `pytestmark = [pytest.mark.profiler]`
- **`tests/profiler/test_profile_sla_dgdr.py`** — Added `pytestmark = [pytest.mark.profiler]`
- **`pyproject.toml`** — Registered `"profiler"` marker (`"planner"` was already registered)

## How to build

```bash
# Build the planner image
python container/render.py --framework=dynamo --target=planner --output-short-filename
docker build -t dynamo:planner -f rendered.Dockerfile .
```

## TODOs for Operations Team

### CI pipeline updates

1. **Build the `dynamo-planner` image in CI** — Create a workflow (similar to `build-frontend-image.yaml`) that builds and pushes the planner image using `--framework=dynamo --target=planner`. Publish it as `dynamo-planner` alongside the existing `dynamo-frontend` image.

2. **Update `container-validation-dynamo.yml`** — Planner/profiler tests currently run in the dynamo runtime image via `pre_merge and not parallel and not (vllm or sglang or trtllm) and gpu_0`. Since those deps were removed from the base requirements, these tests now need to run in the **planner** image. Options:
   - Add a separate pytest job using the planner image with markers `pre_merge and (planner or profiler) and gpu_0`
   - Exclude `planner`/`profiler` markers from the existing dynamo runtime test jobs

3. **Update framework pipeline test markers** — Planner tests tagged with `vllm`/`sglang` (e.g., `test_prometheus.py`, `test_virtual_connector.py`) currently run in framework runtime images. Those images no longer have planner-specific deps (`filterpy`, `prometheus-api-client`, etc.). Either:
   - Run those tests in the planner image instead
   - Or install `requirements.planner.txt` in the framework runtime images too

4. **Check `requirements.test.txt` overlap** — The test requirements file still includes `pmdarima`, `prometheus-api-client`, and `matplotlib` which overlap with `requirements.planner.txt`. Decide whether to also remove them from test requirements (they would only be needed in the planner image).

5. **Published image naming** — Ensure the planner image is published as `dynamo-planner` in the container registry (e.g., `nvcr.io/nvidia/ai-dynamo/dynamo-planner:<tag>`) so that `derive_planner_image()` correctly resolves it from the frontend image name.
