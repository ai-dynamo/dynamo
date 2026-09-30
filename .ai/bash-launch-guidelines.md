# Bash Launch Script Guidelines

Rules and conventions for bash scripts that launch inference engines (vLLM, SGLang,
TensorRT-LLM) in this repository. These apply to scripts under `examples/backends/*/launch/`,
`tests/serve/launch/`, and any other script that starts `dynamo.frontend`, `dynamo.vllm`,
`dynamo.sglang`, or `dynamo.trtllm`.

## Why These Guidelines Exist

Launch scripts are the interface between the test framework and the inference engines.
When they follow a shared pattern, several things become possible that are otherwise
not:

- **Parallel GPU test execution.** The test framework runs multiple GPU tests
  concurrently on the same machine. This only works if each test's launch script
  accepts VRAM budgets (via `gpu_utils.sh`) and unique ports (`DYN_HTTP_PORT`,
  `DYN_SYSTEM_PORT`) from the environment. Scripts that hardcode ports or let
  engines grab all available VRAM cannot participate in parallel runs.

- **Immediate failure detection.** Inference stacks run multiple cooperating processes
  (frontend, workers, routers). If one crashes and the script doesn't notice, the test
  hangs until a global timeout kills it -- wasting GPU-minutes and producing useless
  logs. `wait_any_exit` detects the first child failure immediately and tears everything
  down, so failures surface in seconds instead of minutes.

- **Consistent, debuggable logs.** When every script prints the same startup banner
  (model, port, GPU memory args, example curl), triaging a failed test from CI logs
  is straightforward. Without this, every script prints different things (or nothing),
  and you have to reverse-engineer what configuration was actually used.

- **Reduced duplication and drift.** Shared utilities (`gpu_utils.sh`, `launch_utils.sh`)
  are maintained in one place. Bug fixes and new features (e.g., support for a new
  engine's memory control flag) propagate to all scripts automatically. When scripts
  reimplement this logic inline, they diverge over time and silently break.

- **Lower barrier for contributors.** A new launch script is mostly boilerplate --
  source two files, set a model, background processes, call `wait_any_exit`. This
  makes it easy to add new deployment configurations without understanding the
  internals of process management, VRAM budgeting, or port allocation.

## Critical Rules

These are the conventions that matter most. Exceptions should be rare and justified
in a code comment.

### Source the shared utility libraries

Launch scripts throughout the codebase source `gpu_utils.sh` and `launch_utils.sh`
to share process management, VRAM budgeting, and banner logic from a single
maintained location. New launch scripts should follow the same convention:

```bash
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"   # build_vllm_gpu_mem_args, build_sglang_gpu_mem_args, etc.
source "$SCRIPT_DIR/../../../common/launch_utils.sh" # print_launch_banner, wait_any_exit
```

For test scripts that live outside the examples tree, use `DYNAMO_HOME`:

```bash
export DYNAMO_HOME="${DYNAMO_HOME:-/workspace}"
source "${DYNAMO_HOME}/examples/common/gpu_utils.sh"
source "${DYNAMO_HOME}/examples/common/launch_utils.sh"
```

**Always flag** a launch script that reimplements GPU memory arg construction,
`wait_any_exit`, or banner printing instead of sourcing the shared libraries.

**Always flag** a launch script that manually checks `_PROFILE_OVERRIDE_*` env vars
instead of calling the appropriate `build_*_gpu_mem_args` function.

### Use the engine-specific GPU memory functions for VRAM control

Existing launch scripts call an engine-specific function from `gpu_utils.sh` and pass
the result to the engine CLI to support GPU-parallel test execution. Without
it, engines grab all available VRAM and concurrent tests OOM each other.

Each engine has its own function because the CLI flag semantics differ:

```bash
# vLLM -- returns --kv-cache-memory-bytes N --gpu-memory-utilization 0.01
GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
python -m dynamo.vllm --model "$MODEL" $GPU_MEM_ARGS &

# SGLang -- returns --max-total-tokens N
GPU_MEM_ARGS=$(build_sglang_gpu_mem_args)
python -m dynamo.sglang --model-path "$MODEL" $GPU_MEM_ARGS &

# TensorRT-LLM -- returns JSON for --override-engine-args (merge-aware)
JSON=$(build_trtllm_override_args_with_mem)
python -m dynamo.trtllm --model-path "$MODEL" ${JSON:+--override-engine-args "$JSON"} &
```

```bash
# BAD -- manual env var check; duplicates logic, easy to get wrong
if [[ -n "${_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES:-}" ]]; then
    GPU_MEM_ARGS="--kv-cache-memory-bytes $_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES"
fi
```

### Use `wait_any_exit` instead of bare `wait` or foreground processes

Launch scripts across the codebase background all processes and call `wait_any_exit`
as the last line to detect failures immediately. If any child process crashes,
the script exits with that error code and the EXIT trap tears down the rest.

```bash
# GOOD -- all backgrounded, first failure detected immediately
python -m dynamo.frontend &
python -m dynamo.vllm --model "$MODEL" $GPU_MEM_ARGS &
wait_any_exit

# BAD -- if frontend crashes, script blocks on the foreground vllm process
python -m dynamo.frontend &
python -m dynamo.vllm --model "$MODEL"

# BAD -- `wait` blocks until ALL children exit; a crash in one doesn't surface
# until the others also finish (or hang forever)
python -m dynamo.frontend &
python -m dynamo.vllm --model "$MODEL" &
wait
```

### Make ports injectable via environment variables

Launch scripts accept `DYN_HTTP_PORT` and `DYN_SYSTEM_PORT` from the environment so
the test framework can assign unique ports for parallel execution. This convention is
used throughout the codebase to allow multiple inference stacks to run concurrently on
the same machine without port collisions.

```bash
# GOOD -- test framework can override; defaults are sane for manual use
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
python -m dynamo.frontend &

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    python -m dynamo.vllm --model "$MODEL" &

# BAD -- hardcoded ports; two concurrent tests will collide
python -m dynamo.frontend --http-port 8000 &
python -m dynamo.vllm --model "$MODEL" &
```

For scripts that launch multiple workers, use numbered port vars (`DYN_SYSTEM_PORT1`,
`DYN_SYSTEM_PORT2`, etc.) or compute offsets from a base.

## Standard Script Structure

A well-structured launch script follows this order:

```bash
#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Brief description of what this script launches.

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# ---- Default model ----
MODEL="Qwen/Qwen3-0.6B"

# ---- Parse CLI args ----
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        *)       EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ---- Tunable (override via env vars) ----
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching <description>" "$MODEL" "$HTTP_PORT"

# ---- Launch processes ----
python -m dynamo.frontend &

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    python -m dynamo.vllm --model "$MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

wait_any_exit
```

### Required elements

| Element | Why |
|---------|-----|
| `#!/bin/bash` | Consistent shebang (not `#!/bin/sh` -- we need bash features) |
| SPDX license header | Required by CI copyright check |
| `set -e` | Exit on first error |
| `trap 'echo Cleaning up...; kill 0' EXIT` | Tear down all children on exit |
| `source gpu_utils.sh` | Access to `build_vllm_gpu_mem_args`, `build_sglang_gpu_mem_args`, `build_trtllm_override_args_with_mem` |
| `source launch_utils.sh` | Access to `wait_any_exit`, `print_launch_banner` |
| `GPU_MEM_ARGS=$(build_<engine>_gpu_mem_args)` | VRAM-safe parallel execution |
| `DYN_HTTP_PORT` / `DYN_SYSTEM_PORT` injectable | Port-safe parallel execution |
| `print_launch_banner` | Consistent, debuggable startup logs |
| All processes backgrounded with `&` | Required for `wait_any_exit` |
| `wait_any_exit` as last line | Immediate failure detection |

### Tunable parameters via env vars

Launch scripts should expose key parameters as env vars with sensible defaults:

| Variable | Purpose | Typical default |
|----------|---------|-----------------|
| `MODEL` or `MODEL_PATH` | Model to serve | `Qwen/Qwen3-0.6B` or similar small model |
| `MAX_MODEL_LEN` | Max sequence length | `4096` |
| `MAX_CONCURRENT_SEQS` | Max concurrent sequences | `2` |
| `DYN_HTTP_PORT` | Frontend HTTP port | `8000` |
| `DYN_SYSTEM_PORT` | Worker system port | `8081` |
| `CUDA_VISIBLE_DEVICES` | GPU assignment | Inherited from environment |

## What Not to Do

**Always flag** these patterns in launch scripts:

- Hardcoded ports (e.g., `--http-port 8000` without env var fallback)
- Manual `_PROFILE_OVERRIDE_*` env var handling instead of `build_*_gpu_mem_args`
- Running the last process in the foreground instead of backgrounding + `wait_any_exit`
- Using bare `wait` instead of `wait_any_exit`
- Missing `set -e`
- Missing EXIT trap for cleanup
- Not sourcing `gpu_utils.sh` / `launch_utils.sh` when they provide needed functionality
- Using `sleep N` to wait for server readiness instead of proper health checks

## Shared Utility Reference

Both files live in `examples/common/` and are sourced (not executed) by launch scripts.

### `gpu_utils.sh`

Three engine-specific functions, each returning CLI flags (or JSON) for GPU memory
control. All return empty when no override env var is set, so the engine uses its
default allocation.

| Function | Env var | Output |
|----------|---------|--------|
| `build_vllm_gpu_mem_args` | `_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES` | `--kv-cache-memory-bytes N --gpu-memory-utilization 0.01` |
| `build_sglang_gpu_mem_args` | `_PROFILE_OVERRIDE_SGLANG_MAX_TOTAL_TOKENS` | `--max-total-tokens N` |
| `build_trtllm_override_args_with_mem` | `_PROFILE_OVERRIDE_TRTLLM_MAX_TOTAL_TOKENS` or `_PROFILE_OVERRIDE_TRTLLM_MAX_GPU_TOTAL_BYTES` | JSON for `--override-engine-args` |

vLLM also gets `--gpu-memory-utilization 0.01` because vLLM's startup check rejects
the launch when co-resident tests use >10% of VRAM (it checks free memory against the
fraction *before* applying the byte cap). Setting it to 0.01 bypasses that check.

The TensorRT-LLM function supports `--merge-with-json` to combine GPU memory config
with existing `--override-engine-args` JSON (e.g., perf metrics, tracing endpoints).

The `--kv-cache-memory-bytes` value is per-process: each vLLM worker gets the same
value, even in multi-worker-per-GPU setups. The profiler finds the per-worker budget
directly.

Self-test: `bash examples/common/gpu_utils.sh --self-test` runs built-in assertions.

For the rationale behind absolute caps instead of memory fractions, see
[`examples/common/gpu_utils.md`](../examples/common/gpu_utils.md).

### `launch_utils.sh`

Requires **bash 4.3+** (uses `wait -n`). The file checks at source time and exits
with an error if the bash version is too old.

**`wait_any_exit`** -- Waits for any background child to exit and propagates its exit
code. Traps TERM/INT to exit 0 (clean shutdown) so that test harnesses sending SIGTERM
to the process group don't produce spurious non-zero exit codes. Prints a diagnostic
if no background jobs are found (catches missing `&`).

**`print_launch_banner [flags] <title> <model> <port> [extra_lines...]`** -- Prints a
startup banner with model info and an example curl command. Automatically includes
`MAX_MODEL_LEN` (or `CONTEXT_LENGTH` / `MAX_SEQ_LEN`) and `GPU_MEM_ARGS` in the
banner when those variables are set, so the test log shows exactly what configuration
was used.

Flags (before positional args):
- `--multimodal` -- Use a multimodal (image_url) curl example (`max_tokens=50`)
- `--max-tokens N` -- Override `max_tokens` in the curl example (default: 32)
- `--no-curl` -- Print only the banner, skip the example curl section

**`print_curl_footer`** -- Prints a custom curl example from stdin wrapped in the
standard framing. Pair with `print_launch_banner --no-curl` for non-standard endpoints
(images, video, embeddings) that need a custom request body.

**Constants:** `EXAMPLE_PROMPT` (text LLM prompt), `EXAMPLE_PROMPT_VISUAL` (image/video
generation prompt).

## Related Documentation

- **[`examples/common/gpu_utils.md`](../examples/common/gpu_utils.md)** -- Deep-dive on
  GPU memory control: why absolute caps instead of fractions, per-engine semantics
  (vLLM bytes vs SGLang tokens vs TensorRT-LLM fractions), and how the GPU memory
  functions fit into the profiling and scheduling pipeline.
- **[`tests/README.md`](../tests/README.md)** -- VRAM profiler (`profile_pytest.py`),
  pytest markers for VRAM budgets, and how the test framework sets the
  `_PROFILE_OVERRIDE_*` env vars that the GPU memory functions read.
- **`.ai/bash-launch-guidelines.md`** -- This file
  (CodeRabbit reviews launch scripts against it).
