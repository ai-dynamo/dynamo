# Bash Launch Script Guidelines

Rules and conventions for bash scripts that launch inference engines (vLLM, SGLang,
TensorRT-LLM) in this repository. These apply to scripts under `examples/backends/*/launch/`,
`tests/serve/launch/`, and any other script that starts `dynamo.frontend`, `dynamo.vllm`,
`dynamo.sglang`, or `dynamo.trtllm`.

## Why These Guidelines Exist

Launch scripts are the interface between the test framework and the inference engines.
When they follow a shared pattern, several things become possible that are otherwise
not:

- **Parallel GPU test execution.** The GPU-parallel scheduler (`pytest_parallel_gpu.py`)
  runs multiple tests concurrently on the same GPU. This only works if each test's
  launch script accepts VRAM budgets (`build_gpu_mem_args`) and unique ports
  (`DYN_HTTP_PORT`, `DYN_SYSTEM_PORT`) from the environment. Scripts that hardcode
  ports or let engines grab all available VRAM cannot participate in parallel runs.

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
in order to share process management, VRAM budgeting, and banner logic from a single
maintained location. New launch scripts should follow the same convention:

```bash
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"   # build_gpu_mem_args
source "$SCRIPT_DIR/../../../common/launch_utils.sh" # print_launch_banner, wait_any_exit
```

For test scripts that live outside the examples tree, use `DYNAMO_HOME`:

```bash
export DYNAMO_HOME="${DYNAMO_HOME:-/workspace}"
source "${DYNAMO_HOME}/examples/common/gpu_utils.sh"
source "${DYNAMO_HOME}/examples/common/launch_utils.sh"
```

**Always flag** a launch script that reimplements `build_gpu_mem_args`, `wait_any_exit`,
or banner printing instead of sourcing the shared libraries.

**Always flag** a launch script that manually checks `_PROFILE_OVERRIDE_*` env vars
instead of calling `build_gpu_mem_args`.

### Use `build_gpu_mem_args` for VRAM control

Existing launch scripts call `build_gpu_mem_args <engine>` and pass the result to the
engine CLI in order to support GPU-parallel test execution. Without it, engines grab
all available VRAM and concurrent tests OOM each other.

```bash
# GOOD -- uses shared function; parallel-safe
GPU_MEM_ARGS=$(build_gpu_mem_args vllm)
python -m dynamo.vllm --model "$MODEL" $GPU_MEM_ARGS &

# BAD -- manual env var check; duplicates logic, easy to get wrong
if [[ -n "${_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES:-}" ]]; then
    GPU_MEM_ARGS="--kv-cache-memory-bytes $_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES"
fi
```

For scripts with multiple workers sharing a GPU, use `--workers-per-gpu`:

```bash
GPU_MEM_ARGS=$(build_gpu_mem_args vllm --workers-per-gpu 2)
```

### Use `wait_any_exit` instead of bare `wait` or foreground processes

Launch scripts across the codebase background all processes and call `wait_any_exit`
as the last line in order to detect failures immediately. If any child process crashes,
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

GPU_MEM_ARGS=$(build_gpu_mem_args vllm)

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
| `source gpu_utils.sh` | Access to `build_gpu_mem_args` |
| `source launch_utils.sh` | Access to `wait_any_exit`, `print_launch_banner` |
| `GPU_MEM_ARGS=$(build_gpu_mem_args <engine>)` | VRAM-safe parallel execution |
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
- Manual `_PROFILE_OVERRIDE_*` env var handling instead of `build_gpu_mem_args`
- Running the last process in the foreground instead of backgrounding + `wait_any_exit`
- Using bare `wait` instead of `wait_any_exit`
- Missing `set -e`
- Missing EXIT trap for cleanup
- Not sourcing `gpu_utils.sh` / `launch_utils.sh` when they provide needed functionality
- Using `sleep N` to wait for server readiness instead of proper health checks

## Shared Utility Reference

### `gpu_utils.sh`

- **`build_gpu_mem_args <engine> [--workers-per-gpu N]`** -- Returns engine-specific
  CLI flags for VRAM control. Empty string if no override is set. Engines: `vllm`, `sglang`.

### `launch_utils.sh`

- **`wait_any_exit`** -- Waits for any background child to exit, propagates its exit
  code. Traps TERM/INT for clean shutdown by test harnesses.
- **`print_launch_banner [flags] <title> <model> <port> [extra_lines...]`** -- Prints
  startup banner with model info and example curl. Flags: `--multimodal`, `--max-tokens N`,
  `--no-curl`.
- **`print_curl_footer`** -- Prints a custom curl example from stdin (pair with
  `print_launch_banner --no-curl`).
- **`EXAMPLE_PROMPT`** / **`EXAMPLE_PROMPT_VISUAL`** -- Default prompts for curl examples.
