---
name: kvbm-maturin-dev
description: Rebuild kvbm-py3 Python bindings with correct env ordering (post-maturin nccl re-bump trap)
user-invocable: true
disable-model-invocation: true
---

# KVBM maturin develop

Rebuild the `kvbm-py3` PyO3 extension against the current `.sandbox/` torch, with the correct CUDA env vars exported and the post-maturin nccl re-bump that the phase 3 bring-up learned about the hard way.

**The gotcha**: `maturin develop` runs a pip install step that respects vllm's `nvidia-nccl-cu13==2.28.9` pin. torch 2.11.0 calls `ncclDevCommDestroy` which only exists in nccl 2.29+. Every time you run `maturin develop`, nccl silently rolls back — you **must** re-bump it after.

## Arguments

`/dynamo:kvbm:maturin-dev [--clean] [--features FEATS]`

- **--clean** (default for ABI-change runs): `cargo clean -p kvbm-py3` before rebuilding. Required when torch ABI changed (e.g. after `/dynamo:kvbm:sandbox-venv`).
- **--features FEATS** (default: `v1,v2`): Cargo feature flags. Options: `v1`, `v2`, `dynamo`, `kernels`, `nccl`. The `v1` feature implies `dynamo`.

## Step 1: Preflight

Confirm the venv is present and torch is sm_120+ capable:

```bash
test -x .sandbox/bin/python || { echo "no .sandbox venv — run /dynamo:kvbm:sandbox-venv first"; exit 1; }
.sandbox/bin/python -c "import torch; archs = torch.cuda.get_arch_list(); print('archs:', archs); assert any('sm_10' in a or 'sm_11' in a or 'sm_12' in a for a in archs), 'torch has no sm_100+ kernels; run /dynamo:kvbm:sandbox-venv'"
```

Also confirm CUDA is on disk at the expected location:

```bash
test -d /usr/local/cuda/bin || { echo "CUDA toolkit not at /usr/local/cuda"; exit 1; }
```

## Step 2: Export Env

All of this must be in the same shell as `maturin develop`:

```bash
source .sandbox/bin/activate
export CUDA_PATH=/usr/local/cuda
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export KVBM_REQUIRE_CUDA=1
```

`KVBM_REQUIRE_CUDA=1` makes the kernels build fail loud rather than silently producing a stub.

## Step 3: (Optional) Clean

If `--clean` or if the last rebuild was against a different torch:

```bash
cd lib/bindings/kvbm
cargo clean -p kvbm-py3
cd -
```

Signs you need `--clean`: `undefined symbol` at `import kvbm`, PyO3 ABI mismatch errors, unfamiliar torch-libc symbols in ldd output.

## Step 4: maturin develop

```bash
cd lib/bindings/kvbm
maturin develop --features v1,v2
cd -
```

Stream the output. The build takes ~2-5 minutes cold, under 1 minute incremental. Watch for:
- `Finished dev [unoptimized + debuginfo] target(s)` — rust build OK
- `📦 Built wheel` — maturin packaging OK
- `Installed kvbm-1.0.0` — site-packages install OK

## Step 5: Post-Maturin NCCL Re-Bump (CRITICAL)

maturin's install step just rolled nvidia-nccl-cu13 back to 2.28.9 to satisfy vllm's pin. Undo it:

```bash
uv pip install --force-reinstall --no-deps 'nvidia-nccl-cu13>=2.29'
```

Verify:

```bash
python -c "import torch; print('nccl', torch.cuda.nccl.version())"
```

Expect `(2, 29, 7)` or newer. If you see `(2, 28, 9)`, re-run the install.

## Step 6: Smoke Verification

```bash
python - <<'PY'
import kvbm
print('kvbm version:', kvbm.__version__)
from kvbm.v1 import BlockManager, KvbmLeader, KvbmWorker  # noqa: F401
from kvbm.v2 import KvbmRuntime  # noqa: F401
from kvbm.v1.vllm.connector import DynamoConnector as V1  # noqa: F401
from kvbm.v2.vllm.connector import DynamoConnector as V2  # noqa: F401
assert kvbm._V1_AVAILABLE and kvbm._V2_AVAILABLE
print('OK — all v1/v2 imports resolved')
PY

# Legacy shim regression test — expect 6 passed, 1 skipped
pytest lib/bindings/kvbm/python/tests/test_legacy_imports.py -q
```

Expected: prints `OK — all v1/v2 imports resolved` and pytest shows `6 passed, 1 skipped`.

## Step 7: Next Steps

Tell the user:

```
kvbm-py3 rebuilt. Next:

  Run the determinism flow:
    /dynamo:kvbm:decomposed-run v1-Qwen3-0.6B --fast
    /dynamo:kvbm:decomposed-run v2-Qwen3-0.6B-intra --fast
    /dynamo:kvbm:decomposed-run v2-Qwen3-0.6B-inter --fast

  If imports fail:
    /dynamo:kvbm:diagnose
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `undefined symbol: _ZN3c10...` at `import kvbm` | PyO3 / torch ABI drift | Re-run with `--clean`; if still broken, re-run `/dynamo:kvbm:sandbox-venv` first |
| `ncclDevCommDestroy` error at `import torch` / `import kvbm` | nccl rolled back to 2.28.9 (step 5 skipped or silently failed) | Re-run step 5 |
| `cudarc` link failures during build | `CUDA_PATH` or `CUDA_HOME` not exported | Re-run step 2 in the same shell |
| `nvcc: command not found` during build | CUDA bin not on PATH | `export PATH=/usr/local/cuda/bin:$PATH` before maturin |
| `AttributeError: RustScheduler` during smoke | Expected — scheduler module is commented out at `lib/bindings/kvbm/src/v2/mod.rs:8` | The façade import (`kvbm.v2.vllm.connector.DynamoConnector`) should still succeed; the exception is caught inside `schedulers/dynamo.py` |
| `ImportError: cannot import name 'nixl_connector'` from pd_connector | vllm 0.19.1 renamed the module to `.nixl` | Already fixed via try/except fallback; if you see this, check `lib/bindings/kvbm/python/kvbm/v1/vllm_integration/connector/pd_connector.py:15` |

## Reference: Features

| Feature | Purpose | Default |
|---|---|---|
| `v1` | v1 bindings (dynamo-llm/block-manager) | ✓ |
| `v2` | v2 bindings (kvbm-connector) | ✓ |
| `dynamo` | Enables tokio runtime + OTEL (implied by `v1`) | ✓ (via v1) |
| `kernels` | Bundles kvbm-kernels CUDA kernels statically | no |
| `nccl` | Enables genuine NCCL collectives (inter onboard mode paths) | no |

`v1,v2` is the usual dev build. Add `nccl` if you're iterating on inter-onboard collective code in kvbm-engine.
