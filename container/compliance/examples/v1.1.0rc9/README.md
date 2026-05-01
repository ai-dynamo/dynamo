<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Example output: v1.1.0rc9

Worked attribution data produced by running this PR's pipeline against every
released image at the v1.1.0 rc9 cut on `nvcr.io/nvstaging/ai-dynamo/`. Checked
in so reviewers can see real output without rebuilding.

## Files

| File | Contents |
|------|----------|
| `all-images.csv` | Master union of every package across every image. Columns: `image, package_name, version, type, spdx_license`. |
| `<image>.csv` | Per-image union covering every ecosystem (dpkg + python from the SBOM, rust from `Cargo.lock`, go from `go.mod`, native from `native_packages.yaml`). Columns: `package_name, version, type, spdx_license`. |

The per-image CSVs here are a superset of what `process_results.py --output attribution.csv`
writes (which is dpkg + python only, byte-stable for back-compat). The extra
rows are reverse-parsed from the `ATTRIBUTIONS-Rust.md`, `ATTRIBUTIONS-Go.md`,
and `ATTRIBUTIONS-Native.md` files that the pipeline also produces.

The full `ATTRIBUTIONS-*.md` markdown set and `sbom.spdx.json` (730 KB - 29 MB
each, 99 MB total) are not checked in - CI uploads them as build artifacts per
`.github/actions/compliance-scan/action.yml`.

## Image set

| CSV | Image |
|-----|-------|
| `vllm-runtime.csv` | `nvcr.io/nvstaging/ai-dynamo/vllm-runtime:1.1.0rc9` |
| `sglang-runtime.csv` | `nvcr.io/nvstaging/ai-dynamo/sglang-runtime:1.1.0rc9` |
| `tensorrtllm-runtime.csv` | `nvcr.io/nvstaging/ai-dynamo/tensorrtllm-runtime:1.1.0rc9-cuda13` |
| `dynamo-frontend.csv` | `nvcr.io/nvstaging/ai-dynamo/dynamo-frontend:1.1.0rc9` |
| `dynamo-planner.csv` | `nvcr.io/nvstaging/ai-dynamo/dynamo-planner:1.1.0rc9` |
| `kubernetes-operator.csv` | `nvcr.io/nvstaging/ai-dynamo/kubernetes-operator:1.1.0rc9` |
| `snapshot-agent.csv` | `nvcr.io/nvstaging/ai-dynamo/snapshot-agent:1.1.0rc9` |

TRT-LLM does not publish a plain `1.1.0rc9` tag - only `-cuda13` and `-efa` variants - so the `cuda13` variant is used here.

## Per-image package counts

| Image | dpkg | python | rust | go | native | total |
|-------|-----:|-------:|-----:|---:|-------:|------:|
| `vllm-runtime` | 253 | 394 | 922 | 226 | 14 | 1809 |
| `sglang-runtime` | 419 | 290 | 922 | 226 | 14 | 1871 |
| `tensorrtllm-runtime` | 353 | 325 | 922 | 226 | 14 | 1840 |
| `dynamo-frontend` | 150 | 222 | 922 | 226 | 14 | 1534 |
| `dynamo-planner` | 32 | 131 | 922 | 226 | 14 | 1325 |
| `kubernetes-operator` | 9 | 0 | 922 | 226 | 14 | 1171 |
| `snapshot-agent` | 345 | 6 | 922 | 226 | 14 | 1513 |
| **Master** | | | | | | **11063** |

- `dpkg` and `python` come from the syft SPDX scan and reflect what is actually installed in each image.
- `rust` (922) and `go` (226) are constant because the lockfile parsers operate on `Cargo.lock` and the three `go.mod` files in the repo, not on the image filesystem - statically linked deps are invisible to syft.
- `native` (14) is the constant overlay from `container/compliance/native_packages.yaml`.
- The operator image is distroless plus the controller binary, so `dpkg` is tiny (9) and there are no python deps.

## License resolution

Generated with `--lookup-licenses --dedupe` plus the curated overrides in [`container/compliance/license_overrides.yaml`](../../license_overrides.yaml). Resolution order per row:

1. Syft's SPDX scan (Apt/Python from the image filesystem).
2. Lockfile resolvers - `crates.io` for Rust, `api.deps.dev` for Go, PyPI JSON for Python.
3. PyPI retry against the PEP 440 base version (drops `+cu129`, `+nv26.x`, etc.) so NVIDIA-rebuilt wheels resolve against the upstream release.
4. GitHub `/repos/<owner>/<repo>/license` fallback when PyPI metadata is empty but a `github.com` URL is linked.
5. Hand-curated `license_overrides.yaml` entries for NVIDIA proprietary packages (CUDA toolkit, Nsight, mlnx-dpdk) and a small set of repackaged PyPI wheels whose metadata is empty (DeepSeek MoE kernels, openai-harmony, torchao, etc).

All 11,063 rows resolve to a SPDX or `LicenseRef-*` license - **0 `UNKNOWN`**.

## How these were generated

```bash
docker buildx build \
  --platform linux/amd64 \
  --build-arg TARGET_IMAGE=nvcr.io/nvstaging/ai-dynamo/<image>:1.1.0rc9 \
  --output type=local,dest=<out> \
  --pull --no-cache-filter extractor \
  -f container/compliance/Dockerfile.extract container/compliance/

python container/compliance/process_results.py \
  --target-dir <out> \
  --cargo-lock Cargo.lock \
  --go-mod deploy/operator/go.mod \
  --go-mod deploy/snapshot/go.mod \
  --go-mod deploy/inference-gateway/epp/go.mod \
  --native-packages container/compliance/native_packages.yaml \
  --license-overrides container/compliance/license_overrides.yaml \
  --attributions-dir <out> \
  --output <out>/attribution.csv \
  --lookup-licenses --dedupe
```

The first `--lookup-licenses` run takes ~16 min (crates.io throttles to 1 req/sec) and populates an SQLite cache at `~/.cache/dynamo-compliance/license-lookup.sqlite`. Subsequent runs - any image, any release - are instant.
