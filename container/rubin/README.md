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
from source → **wire up UCX + NIXL** → `ENTRYPOINT bash`.

### UCX + NIXL

**What these are:** **NIXL** is NVIDIA's library for moving KV-cache data quickly between GPUs or
nodes — Dynamo needs it for *disaggregated serving* (running prefill on one GPU and decode on
another). **UCX** is the lower-level networking library NIXL is built on top of.

**The key idea:** how each image gets NIXL/UCX depends on its **base image** (the `FROM …` layer it
starts from). Two of our bases already ship NIXL/UCX; the third does not — so we handle them
differently. Think of it like moving in:

- The **vllm** and **sglang** bases are a *furnished apartment* — NIXL/UCX are already installed, so
  we just point Dynamo at them.
- The **trtllm** base is an *empty apartment* — we build and install NIXL/UCX ourselves.

| Image | Base ships NIXL/UCX? | What the Dockerfile does |
|-------|----------------------|--------------------------|
| `Dockerfile.vllm`   | **Yes** — NIXL 1.3.0 + UCX 1.21.0, pre-installed at `/usr/local/nixl` + `/usr/local/ucx` | **Reuse the base's copy.** Set `NIXL_PREFIX` / `NIXL_LIB_DIR` / `NIXL_PLUGIN_DIR` and add UCX to `LD_LIBRARY_PATH` so Dynamo can find the existing files. We deliberately **don't rebuild** — that would install an older NIXL and shadow the good one the base ships. |
| `Dockerfile.sglang` | **Yes** — same as vllm | Same as vllm. |
| `Dockerfile.trtllm` | **No** — its `pytorch-rubin` base is a *build base*; NIXL isn't installed | **Build from source**, in order `hwloc → gdrcopy → libfabric → UCX → NIXL`, following the same recipe Dynamo's runtime templates use (`container/templates/wheel_builder.Dockerfile`). |

**Why "reuse" still needs some wiring:** when a program needs a shared library (a `.so` file),
Linux searches the folders listed in `LD_LIBRARY_PATH`. Dynamo's Rust code loads `libnixl.so`
*by name* at runtime, so even though the file already exists in the base image, we still have to
point the environment variables and `LD_LIBRARY_PATH` at it — otherwise the OS can't find it.

**The trtllm from-source versions** come from `container/context.yaml` (NIXL `v1.0.1`, UCX
`v1.20.x`, gdrcopy `v2.5.2`, libfabric `v2.5.1`, hwloc `2.12.2`), and each can be overridden at
build time, e.g. `--build-arg NIXL_REF=…`. The built NIXL lands in `/opt/nvidia/nvda_nixl` and UCX
in `/usr/local/ucx`, wired up with the same environment variables.

**Why the build order matters (trtllm):** the NIXL build is placed **after** the Dynamo wheel
build on purpose. Dynamo compiles against a lightweight *stub* of NIXL — a placeholder that lets
the code build *without* the real library present — because the real NIXL is only needed **at
runtime**. So we build Dynamo (using the stub) first, then add the real NIXL afterward, exactly how
Dynamo's runtime images already do it. How Dynamo itself compiles doesn't change.

**A built-in safety net (vllm/sglang):** the Dockerfile checks that the base's NIXL files are
where we expect (`test -f` / `test -d`) at build time. If a future base image ever **moves** NIXL,
the build **fails immediately and loudly** — printing a directory listing of what's actually there
— instead of quietly producing an image that looks fine but can't do disaggregated serving.

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
