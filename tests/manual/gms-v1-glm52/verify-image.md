<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# verify-image — pre-flight checks

Run these **before** applying `20-dynamocheckpoint.yaml`. Each failure here would
otherwise surface as a confusing mid-run crash and be misclassified as a driver
problem when it is really **(P) plumbing**.

```bash
CTX=nv-prd-dgxc.teleport.sh-dynamo-nscale-dev-cluster
NS=schwinns
# TODO: confirm this matches the tag produced by build-on-demand (06-placeholder-image.md).
IMG=dynamoci.azurecr.io/ai-dynamo/dynamo:1.4.0-ci-05dac0c7da0372312819e256e1b0cd4a07a61eab-vllm-placeholder
```

Checks run **inside the cluster**, not with local docker. Start one throwaway
pod on s2877 and exec into it for checks 1-4:

```bash
kubectl --context "$CTX" -n "$NS" run imgcheck \
  --image="$IMG" --restart=Never --command -- sleep 3600 \
  --overrides='{"spec":{
    "nodeSelector":{"kubernetes.io/hostname":"cluster-0967a26d-pool-14bee067-prctr-s2877"},
    "tolerations":[{"operator":"Exists"}],
    "imagePullSecrets":[{"name":"acr-token-secret"}]}}'

kubectl --context "$CTX" -n "$NS" wait --for=condition=Ready pod/imgcheck --timeout=600s

# Helper used by every check below.
R() { kubectl --context "$CTX" -n "$NS" exec imgcheck -- bash -lc "$1"; }
```

Delete it when done: `kubectl --context "$CTX" -n "$NS" delete pod imgcheck`.

> Checks 1-4 need no GPU. Check 5 does — it uses a separate GPU-claiming pod.

## 1. Placeholder tooling (runtime-vs-placeholder discriminator) ⚠️ MOST IMPORTANT

If any of these are missing you have a **runtime** image, and the 8-GPU
checkpoint Job will die instantly with
`exec: "cuda-checkpoint": executable file not found in $PATH`.
See `06-placeholder-image.md`.

```bash
R 'command -v cuda-checkpoint            && cuda-checkpoint --help | head -3'
R 'command -v cuda-checkpoint-helper'
R 'command -v nsrestore'
R 'command -v criu                       && criu --version'
R 'echo "ORIGINAL_BASE_IMAGE=$ORIGINAL_BASE_IMAGE"'
```

| Expected | Source |
|---|---|
| `/usr/local/sbin/cuda-checkpoint` | `deploy/snapshot/Dockerfile:407` |
| `/usr/local/bin/cuda-checkpoint-helper` | `Dockerfile:409` |
| `/usr/local/bin/nsrestore` | `Dockerfile:413` — must equal `nsRestorePath` (`snapshot/values.yaml:176`) |
| `criu --version` prints a version | `Dockerfile:402-404` |
| `ORIGINAL_BASE_IMAGE` = the runtime tag | `Dockerfile:375` |

## 2. GMS V1 package + compiled allocator extension ⚠️ SECOND MOST IMPORTANT

```bash
# V1 worker module must import.
R 'python3 -c "import gpu_memory_service.v1.integrations.vllm.worker as w; print(w.GMSV1Worker)"'

# The compiled CUDAPluggableAllocator extension. This is the one that fails
# SILENTLY: extensions/__init__.py:13-19 swallows ImportError and sets None,
# then backend.py:64-65 raises "GPU Memory Service allocator extension is not built"
# only once a GPU worker starts - i.e. 30+ minutes into a run.
R 'python3 -c "
import ctypes
from gpu_memory_service.core.client.torch.extensions import _allocator_ext
assert _allocator_ext is not None, \"FAIL: _allocator_ext is None (extension not built)\"
print(\"OK\", _allocator_ext.__file__)

# init_module IS a Python-level method (PyMethodDef in allocator.cpp:140).
assert hasattr(_allocator_ext, \"init_module\"), \"FAIL: init_module missing\"

# my_malloc/my_free are extern \"C\" symbols (allocator.cpp:50-89). They are
# resolved by torch via dlsym on the .so path, NOT as Python attributes.
# hasattr(_allocator_ext, \"my_malloc\") is EXPECTED to be False - that is
# not a failure. Check dlsym resolution instead.
lib = ctypes.CDLL(_allocator_ext.__file__)
for sym in (\"my_malloc\", \"my_free\"):
    getattr(lib, sym)  # raises AttributeError if dlsym cannot resolve it
    print(\"dlsym OK:\", sym)
"'
```

> [!WARNING]
> **Do NOT assert `hasattr(_allocator_ext, "my_malloc")`.** `my_malloc`/`my_free`
> are `extern "C"` symbols (`allocator.cpp:50-89`), consumed by
> `torch.cuda.CUDAPluggableAllocator(_allocator_ext.__file__, "my_malloc", "my_free")`
> (`backend.py:67-71`) which resolves them **by dlsym on the shared object**.
> Only `init_module` is exported as a Python method (`allocator.cpp:140`).
> So the module's `dict_keys` legitimately contains just
> `init_module` + dunders. A previous run aborted the whole experiment on this
> false negative — the image was fine.

## 3. GMS V1 server CLI accepts `--use-v1`

```bash
R 'python3 -m gpu_memory_service.cli.server --help | grep -A2 use-v1'
R 'python3 -c "
from gpu_memory_service.cli.server import _child_command
print(_child_command(0, \"cuda\", use_v1=True))
"'
```

Expect `['<python>', '-m', 'gpu_memory_service', '--use-v1', '--device', '0']`
(`cli/server.py:30-38`). Note `--device-type` is **omitted** when `use_v1` is
true — that is correct.

```bash
# V1 serves exactly these two domains (v1/cli.py:19).
R 'python3 -c "from gpu_memory_service.v1.cli import _DOMAINS; print(_DOMAINS)"'
# -> ('weights', 'kv_cache')
```

## 4. Saver/loader CLIs are the **V1** (per-device) ones

This is the check the last run needed and did not have: an image with #12011 but
without #12392 has only the V0 saver, and `gms-saver` dies with
`msgspec.ValidationError: Object missing required field 'success'`.

```bash
R 'python3 -m gpu_memory_service.cli.snapshot.saver  --use-v1 --help'
R 'python3 -m gpu_memory_service.cli.snapshot.loader --use-v1 --help'
```

| CLI | Must list **exactly** | Must **not** list |
|---|---|---|
| saver `--use-v1` | `--checkpoint-dir` (required), `--device`, `--shard-size-bytes` | `--sharded-ssd-roots`, `--max-workers`, `--save-lock-timeout-ms`, `--device-type` |
| loader `--use-v1` | `--checkpoint-dir` (required), `--device`, `--max-workers` | `--sharded-ssd-roots`, `--transfer-backend`, `--sharded-ssd-queues-per-root`, `--device-type` |

Verified at `v1/saver.py:22-28` and `v1/loader.py:22-30`. Both parsers are
`allow_abbrev=False`, so a stray V0 flag is a hard error rather than a silent
no-op. If either shows `--sharded-ssd-roots`, **the image predates #12392 —
stop and rebuild.**

```bash
# The modules must exist at all (they are what #12392 adds).
R 'python3 -c "
import gpu_memory_service.v1.saver, gpu_memory_service.v1.loader
from gpu_memory_service.v1.snapshot import save_weights, hydrate_weights
print(\"V1 saver/loader present\")
"'

# Artifact layout is per-device: <checkpoint-dir>/device-<N>.
R 'python3 -c "
import os
print(os.path.join(\"/mnt/gms-ssd/nvme2/schwinns/gmsv1-glm52-tep8\", \"device-0\"))
"'
```

Expect `/mnt/gms-ssd/nvme2/schwinns/gmsv1-glm52-tep8/device-0`
(`v1/saver.py:30-35`, `v1/loader.py:32-34`). The saver creates
`device-N/shards/` and writes `device-N/manifest.json`
(`v1/snapshot.py:138-140,167-169`).

## 5. GLM-5.2 architecture is registered in this vLLM (needs a GPU)

This check needs a real GPU, so it runs as a short-lived pod that claims one via
DRA. Simplest option: reuse the `imgcheck` pod only if it holds a GPU claim;
otherwise run the check inside the checkpoint Job's `main` container on the first
attempt and read it from the logs.

```bash
R 'python3 -c "
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()
print(\"GLM-ish archs:\", [a for a in archs if \"glm\" in a.lower()])
"'
```

`get_supported_archs()` reads a static registry and generally works without a
GPU; if it fails for lack of CUDA, defer this check to the first run's `main`
container logs, where an unsupported architecture surfaces immediately as
`Model architectures [...] are not supported`.

Then check it against the cached model's own `config.json`:

```bash
# Inside a pod that mounts shared-model-cache, or on the node:
python3 -c "
import json,glob
p = glob.glob('/home/dynamo/.cache/huggingface/**/nvidia--GLM-5.2-NVFP4/**/config.json', recursive=True)
print(p[:1])
print(json.load(open(p[0]))['architectures'])
"
```

The architecture string must appear in the registry list. If not, vLLM will fail
with `Model architectures [...] are not supported` — a clear **(P)** result.

## 6. Model is actually cached (offline mode depends on it)

`20-dynamocheckpoint.yaml` sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
If the model is not in `shared-model-cache`, load fails with a confusing offline
error rather than a download.

```bash
kubectl --context "$CTX" -n schwinns exec nvme-janitor -- \
  sh -c 'ls /mnt/nvme2 >/dev/null'   # (janitor pod from 07-disk-cleanup.md)

# Better: a throwaway pod mounting the PVC.
kubectl --context "$CTX" -n schwinns run hfcheck --rm -it --restart=Never \
  --image=busybox:1.36 --overrides='{"spec":{
    "nodeSelector":{"kubernetes.io/hostname":"cluster-0967a26d-pool-14bee067-prctr-s2877"},
    "tolerations":[{"operator":"Exists"}],
    "containers":[{"name":"c","image":"busybox:1.36","command":["sh","-c",
      "du -sh /hf/hub/models--nvidia--GLM-5.2-NVFP4 2>/dev/null || ls /hf/hub | head -30"],
      "volumeMounts":[{"name":"hf","mountPath":"/hf"}]}],
    "volumes":[{"name":"hf","persistentVolumeClaim":{"claimName":"shared-model-cache"}}]}}'
```

Expect ~372 GB under `models--nvidia--GLM-5.2-NVFP4`.

<!-- UNVERIFIED: the exact on-PVC layout (whether HF_HOME root is the PVC root or
     a subdir) was not confirmed. The reference mounts the PVC at
     /home/dynamo/.cache/huggingface and sets HF_HOME to the same path, so the hub
     dir should be <mount>/hub/models--nvidia--GLM-5.2-NVFP4. -->

## Summary gate

Do not proceed unless all of these are true:

- [ ] `cuda-checkpoint`, `cuda-checkpoint-helper`, `nsrestore`, `criu` present (check 1)
- [ ] `_allocator_ext is not None` (check 2)
- [ ] `GMSV1Worker` imports (check 2)
- [ ] `--use-v1` accepted, domains `('weights','kv_cache')` (check 3)
- [ ] all four saver flags exist, 7 roots parse (check 4)
- [ ] GLM-5.2 arch registered (check 5)
- [ ] model present in `shared-model-cache` (check 6)
