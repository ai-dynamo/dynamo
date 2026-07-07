# Dynamo on Vera Rubin (VR200) — bring-up container builds

Dockerfiles for building **Dynamo + {TRT-LLM, vLLM, SGLang}** images for **Vera Rubin
(VR200, sbsa/arm64)** during bring-up. These intentionally **do not** use `render.py`; per
@Ishan Dhanani's guidance we **install Dynamo from source inside the published Rubin framework
image** — render.py's recipes assume a *release*-image layout that doesn't match the Rubin
images yet (see Gotchas).

> Status: **trtllm validated** (`…tzulingk-trtllm-rubin:sbsa-fromsrc-main-7c8b4b19` — builds,
> pushes, `import dynamo` + `import tensorrt_llm` OK). vLLM/SGLang building with the same recipe.
> Tracked under DIS-2292 (trtllm) / DIS-2294 (vLLM) / DIS-2293 (SGLang).

## What these build

| File | Base image (CUDA 13.4, sbsa) | Framework installed? | Extra |
|------|------------------------------|----------------------|-------|
| `Dockerfile.trtllm` | `urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm:pytorch-rubin-py3-sbsa-…trt10.15.1.29-20260512` | **No** (build base) | installs the prebuilt trtllm wheel from urm Artifactory |
| `Dockerfile.vllm`   | `gitlab-master.nvidia.com:5005/dl/dgx/vllm:rubin-py3-devel` | **Yes** | — |
| `Dockerfile.sglang` | `gitlab-master.nvidia.com:5005/dl/dgx/sglang:rubin-py3-devel` | **Yes** | `blake3` |

All three then: install `nats`+`etcd` → Rust + maturin → build & `pip install` the Dynamo wheels
from source → **build & install UCX + NIXL from source** → `ENTRYPOINT bash`.

### UCX + NIXL (built from source)

The Rubin framework bases ship **no** UCX/NIXL, so each image builds them from source using
the exact same recipe as `container/templates/wheel_builder.Dockerfile`'s CUDA path
(`hwloc` → `gdrcopy` → `libfabric` → `UCX` → `NIXL` via meson), with the `dnf` package list
mapped to its Ubuntu/`apt` equivalents. Versions come from `container/context.yaml`:

| Component | Version (context.yaml key) | trtllm | vllm | sglang |
|-----------|----------------------------|--------|------|--------|
| NIXL      | `<framework>.nixl_ref`     | `v1.0.1` | `v1.1.0` | `v1.0.1` |
| UCX       | `nixl_ucx_ref`             | `v1.20.x` | `v1.20.x` | `v1.20.x` |
| gdrcopy   | `nixl_gdrcopy_ref`         | `v2.5.2` | `v2.5.2` | `v2.5.2` |
| libfabric | `nixl_libfabric_ref`       | `v2.5.1` | `v2.5.1` | `v2.5.1` |
| hwloc     | `hwloc_version`            | `2.12.2` | `2.12.2` | `2.12.2` |

Override any of these per build with `--build-arg NIXL_REF=… / NIXL_UCX_REF=… / …`. NIXL
installs to `/opt/nvidia/nvda_nixl` (libs under `lib64`, matching the templated CUDA layout)
and UCX to `/usr/local/ucx`; both are wired via `NIXL_PREFIX`/`NIXL_LIB_DIR`/`NIXL_PLUGIN_DIR`
and `LD_LIBRARY_PATH` so the Rust `nixl-sys` `dlopen("libnixl.so")` resolves the real library
at runtime instead of falling into stub mode. The build is placed **after** the Dynamo wheel
build so Dynamo itself still compiles in the validated stub-mode path (unchanged); NIXL is
supplied purely at runtime, exactly as the runtime templates do.

## Build (on a native arm64 Docker host, e.g. `computelab-armbuild-1`)

```bash
# login to the registries the bases live on (one-time):
#   nvcr.io (push target): docker login nvcr.io -u '$oauthtoken' -p <NGC_API_KEY>
#   gitlab-master:5005 (vllm/sglang bases): docker login gitlab-master.nvidia.com:5005 -u <user> -p <PAT>
#   urm.nvidia.com pulls anonymously from the build host.

docker build --platform linux/arm64 -f Dockerfile.trtllm -t <registry>/<you>-trtllm-rubin:<tag> .
docker build --platform linux/arm64 -f Dockerfile.vllm   -t <registry>/<you>-vllm-rubin:<tag> .
docker build --platform linux/arm64 -f Dockerfile.sglang -t <registry>/<you>-sglang-rubin:<tag> .
```

Override the base or the trtllm wheel via `--build-arg BASE=…` / `--build-arg TRTLLM_WHEEL_URL=…`,
and the Dynamo source via `--build-arg DYNAMO_REF=<branch-or-sha>`.

## Run (on Hecate — Enroot/Pyxis, no Docker)

```bash
# import + serve TinyLlama (default model) aggregated on one VR200 node, then curl it:
srun -A <account> -p batch-spx -N1 -t 00:30:00 \
  --container-image=nvcr.io#<registry>/<you>-trtllm-rubin:<tag> --pty bash
#  in-container:  nats-server & etcd & \
#                 python3 -m dynamo.trtllm --disaggregation-mode agg & \
#                 python3 -m dynamo.frontend --http-port 8000 & \
#                 curl localhost:8000/v1/chat/completions -d '{"model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","messages":[{"role":"user","content":"hi"}]}'
```

DeepSeek-R1 (NVFP4, single node, aggregated — per the vLLM B200 recipe):
`python3 -m dynamo.trtllm --model nvidia/DeepSeek-R1-FP4 --tensor-parallel-size 4 --expert-parallel-size 4 --enable-attention-dp --disaggregation-mode agg`.

## Gotchas (hard-won on Rubin)

1. **`cudarc` rejects CUDA 13.4** (its allow-list stops at 13.1) → the maturin/cargo build panics
   with `Unsupported cuda toolkit version: 13.4`. Fix: `ENV CUDARC_CUDA_VERSION=13010` (bind to the
   13.1 API, which is forward-compatible with the 13.4 driver). **Required for all three.**
2. **dl/dgx vLLM & SGLang bases lack `libclang`** → bindgen (via `nixl-sys`) fails with
   `Unable to find libclang`. Fix: `apt-get install clang libclang-dev`. (The trtllm pytorch-rubin
   base already ships it.)
3. **The trtllm "image" is a *build base*, not a release image** — no `tensorrt_llm` installed.
   Install the prebuilt VR200 wheel from urm Artifactory first (there is no Rubin trtllm release
   image yet — confirmed with #ghw-dl-trtllm-rubin).
4. `libcuda.so.1: cannot open shared object file` when testing `import tensorrt_llm` on a CPU
   build host is **expected** (driver lib only exists on a GPU) — not a build error.

## Future

Productize into `render.py` as a Rubin variant (encode the above as `context.yaml` data knobs +
Jinja gates) so that, once a Rubin trtllm *release* image and upstreamed vLLM/SGLang Rubin support
exist, the standard `FROM <published image>` flow works on Rubin unchanged.
