---
name: kvbm-sandbox-venv
description: Set up or repair the .sandbox venv for KVBM development (especially GB10/sm_121 Blackwell)
user-invocable: true
disable-model-invocation: true
---

# KVBM Sandbox Venv Setup

Set up or repair the `.sandbox/` uv venv used for local KVBM iteration. Targets the tricky path — NVIDIA GB10 / Blackwell (sm_121) — where stock `vllm 0.19.0` dies with `cudaErrorNoKernelImageForDevice` because its bundled torch has no sm_120+ kernels.

The known-good recipe was derived during ACTIVE_PLAN phase 3 and recorded in `.sandbox/requirements.after-phase3-wheels.txt`.

After this skill completes, run `/dynamo:kvbm:maturin-dev` to (re)build `kvbm-py3` against the new torch ABI.

## Arguments

`/dynamo:kvbm:sandbox-venv [--repair] [--fresh] [--skip-vllm]`

- **--repair** (default): Operate on the existing `.sandbox/` venv in place.
- **--fresh**: Delete `.sandbox/` first and rebuild from scratch via `uv venv .sandbox --clear`.
- **--skip-vllm**: Only install torch/nccl/openblas; skip the vllm nightly step (use when iterating on non-vllm paths).

## Step 1: Preflight

Check the host is what we think it is:

```bash
uname -m                                             # expect aarch64 on GB10, x86_64 on DGX/H100
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
nvcc --version | grep release
```

Report to user:
- GPU name + compute capability (e.g. `GB10, 12.1` = sm_121 Blackwell)
- nvcc version (kernels crate expects CUDA 13.x on GB10)
- Existing `.sandbox/` state (`ls -ld .sandbox` and `.sandbox/bin/python --version` if present)

If compute_cap is `12.1` (sm_121): **this is the Blackwell path below**. Otherwise the stock pyproject.toml install should work; tell the user and exit unless they explicitly asked for `--fresh`.

## Step 2: Snapshot Existing Env (Rollback Escape)

If `.sandbox/` already exists and `--fresh` was not passed:

```bash
.sandbox/bin/python -m pip freeze > .sandbox/requirements.before.txt
ls -la .sandbox/requirements.*.txt
```

Tell the user: "If anything goes sideways, roll back with `uv pip install -r .sandbox/requirements.before.txt`".

There are already two historical snapshots in `.sandbox/`:
- `.sandbox/requirements.before-phase3.txt` — stock vllm 0.19.0 / torch 2.10.0+cu126 (pre-GB10 fix)
- `.sandbox/requirements.after-phase3-wheels.txt` — known-good cu130 nightly set (post-GB10 fix)

## Step 3: System Packages (GB10 only)

torch 2.11.0+cu130 and the cypheritai diagnostic alpha both need libopenblas on the host:

```bash
sudo apt install -y libopenblas0 libopenblas0-pthread
```

Skip if already installed (`dpkg -l libopenblas0 &>/dev/null`).

## Step 4: Create Or Enter the Venv

If `--fresh`:
```bash
rm -rf .sandbox
uv venv .sandbox --clear --python 3.12
VIRTUAL_ENV=.sandbox PATH=.sandbox/bin:$PATH \
    uv pip install -r tests/kvbm_integration/requirements.txt
```

Otherwise:
```bash
source .sandbox/bin/activate
```

## Step 5: Install vllm cu130 Nightly + Pinned NCCL

This is the gotcha-heavy step. Order matters.

```bash
# 1. Remove the stale cu12 nccl that comes with torch 2.10.0+cu126 — it
#    stomps the install path if left around.
uv pip uninstall nvidia-nccl-cu12 || true

# 2. Remove any existing vllm so the nightly resolver runs fresh.
uv pip uninstall vllm || true

# 3. Install the cu130 nightly vllm. The extra index is live nightly wheels
#    — this pulls torch 2.11.0+cu130 + torchvision 0.26 + torchaudio 2.11
#    as transitive deps, targeting sm_80..sm_120 (fwd-compat to sm_121).
uv pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly/cu130

# 4. Force-bump nvidia-nccl-cu13. torch 2.11.0 calls ncclDevCommDestroy which
#    exists only in nccl >= 2.29; vllm pins it to 2.28.9, so we overwrite.
uv pip install --force-reinstall --no-deps 'nvidia-nccl-cu13>=2.29'
```

If `--skip-vllm` was passed, do only step 4 after ensuring torch 2.11+cu130 is otherwise present.

## Step 6: Verify

```bash
# Torch arch list must include sm_100/sm_110/sm_120 for Blackwell fwd-compat
python -c "import torch; print('torch', torch.__version__, 'archs', torch.cuda.get_arch_list())"

# NCCL version must be >= 2.29 (DevCommDestroy availability)
python -c "import torch; print('nccl', torch.cuda.nccl.version())"

# Tiny matmul on the GPU — catches cudaErrorNoKernelImageForDevice immediately
python -c "import torch; x = torch.randn(64, 64, device='cuda'); print('matmul', (x @ x.T).mean().item())"

# vllm boots (30 sec smoke — expect "Application startup complete")
timeout 60 vllm serve Qwen/Qwen3-0.6B \
    --max-model-len 256 --gpu-memory-utilization 0.5 2>&1 | tail -30
```

Stop the smoke server with Ctrl-C after the "Application startup complete" line.

Expected output highlights:
- `torch 2.11.0+cu130`, archs containing `sm_100`, `sm_110`, `sm_120`
- `nccl (2, 29, 7)` or newer
- matmul prints a finite float (no `cudaErrorNoKernelImageForDevice`)
- vllm reaches `Application startup complete`

## Step 7: Snapshot and Next Steps

Capture the final pip state for reproducibility:

```bash
.sandbox/bin/python -m pip freeze > .sandbox/requirements.after.txt
diff .sandbox/requirements.before.txt .sandbox/requirements.after.txt | head -30
```

Tell the user:

```
.sandbox venv ready. Next steps:

  1. Rebuild kvbm-py3 against the new torch ABI:
       /dynamo:kvbm:maturin-dev

  2. Run the decomposed determinism flow:
       /dynamo:kvbm:decomposed-run v1-Qwen3-0.6B --fast

  3. If anything looks wrong, tail a vllm log:
       /dynamo:kvbm:diagnose
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `cudaErrorNoKernelImageForDevice` during matmul | torch has no sm_120+ kernels | Re-run step 5; confirm you're on the cu130 nightly index, not the stable cu126 index |
| `undefined symbol: ncclDevCommDestroy` at vllm startup | nccl rolled back to 2.28.9 | Re-run step 5 command 4 (force-reinstall nccl >= 2.29) |
| `nvidia-nccl-cu12 reappeared` in `pip freeze` | Some transitive dep pulled it back | `uv pip uninstall nvidia-nccl-cu12` and move on — it's a dead symlink on GB10 |
| FP8 crashes on sm_121 | vllm FP8 kernels incomplete on sm_121 (eugr/spark-vllm-docker#143) | Avoid FP8 model variants. Qwen3-0.6B and DeepSeek-R1-Distill-Llama-8B are safe |
| `libopenblas.so.0: cannot open shared object file` | Missing system package | Re-run step 3 |
| `maturin develop` fails with `cudarc` link errors | `CUDA_PATH` / `CUDA_HOME` not exported | Use `/dynamo:kvbm:maturin-dev` — it exports everything correctly |

## Reference: Known-Good Pinned Set

`.sandbox/requirements.after-phase3-wheels.txt` — captured 2026-04-13 on GB10. Notable pins:

| Package | Version |
|---|---|
| `torch` | `2.11.0+cu130` |
| `torchvision` | `0.26.0` |
| `torchaudio` | `2.11.0` |
| `vllm` | `0.19.1rc1.dev232+g0e39202ca.cu130` |
| `nvidia-nccl-cu13` | `2.29.7` |
| `flashinfer-python` | `0.6.7` |

If a future wheel install corrupts the venv, `uv pip install -r .sandbox/requirements.after-phase3-wheels.txt` is the recovery path.
