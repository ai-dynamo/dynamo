# Container Python Dependencies

Python dependency files for Dynamo container images, split by component
so each image only installs what it needs.

## Files

| File | Purpose |
|------|---------|
| `requirements.common.txt` | Core deps shared by all containers |
| `requirements.aisimulate.txt` | Source-built AISim wheel staged into all container wheelhouses |
| `requirements.planner.txt` | Planner, profiler, and global_planner deps |
| `requirements.frontend.txt` | Frontend deps |
| `requirements.benchmark.txt` | Benchmark and profiling tools |
| `requirements.test.txt` | Test-only deps |
| `requirements.dev.txt` | Dev-only tools |

## AISim Source Build

Container builds compile `aisim` from the exact `aisim-core` source revision in
the root `Cargo.toml` and install that wheel explicitly. Python package metadata
uses `aisim==0.14.0`; it does not encode the source revision. Until that version is
published, local source builds must build and install the matching wheel before
installing Dynamo:

```bash
python container/deps/build_aisimulate.py --root . \
  --source /tmp/dynamo-aisim-source --output /tmp/dynamo-aisim-wheels
python -m pip install /tmp/dynamo-aisim-wheels/aisim-*.whl
```

The helper requires Git, Rust, `maturin`, and Python `packaging`. The source
directory must not already exist. Imports remain `aisimulate` and `aisimulate_core`.

## Version Pinning Strategy

- Use `==` for packages that are pure Python and well-tested.
- Use `<=` or `<` for packages that may have platform-specific builds
  (CUDA, system packages) — the max available version can differ across
  x86_64 / aarch64 and CUDA versions.
- **Never use `>=`** as it allows untested future versions that may introduce
  breaking changes, create non-reproducible builds, and cause dependency
  conflicts. Every installed version should be explicitly tested.
