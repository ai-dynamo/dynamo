# Container Python Dependencies

Python dependency files for Dynamo container images, split by component
so each image only installs what it needs.

## Files

| File | Purpose |
|------|---------|
| `requirements.common.txt` | Core deps shared by all containers |
| `requirements.planner.txt` | Planner, profiler, global_planner, deploy utils |
| `requirements.frontend.txt` | Frontend deps |
| `requirements.vllm.txt` | vLLM-specific deps |
| `requirements.trtllm.txt` | TensorRT-LLM-specific deps |
| `requirements.benchmark.txt` | Benchmark and profiling tools |
| `requirements.test.txt` | Test-only deps |
| `requirements.dev.txt` | Dev-only tools |

## Version Pinning Strategy

- Use `==` for packages that are pure Python and well-tested.
- Use `<=` or `<` for packages that may have platform-specific builds
  (CUDA, system packages) — the max available version can differ across
  x86_64 / aarch64 and CUDA versions.
- **Never use `>=`** as it allows untested future versions that may introduce
  breaking changes, create non-reproducible builds, and cause dependency
  conflicts. Every installed version should be explicitly tested.
