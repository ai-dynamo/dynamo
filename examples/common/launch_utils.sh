#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared launch utilities for example scripts.
#
# Goal:
#   Unify behavior and reduce duplication across vLLM, SGLang, and TensorRT-LLM
#   example launch scripts so they share one pattern for banners, process
#   management, and example curl output.
#
# Benefits:
#   - Single place to change banner format, example prompts, and wait/cleanup logic
#   - Consistent UX: same startup output and exit behavior across all backends
#   - Less per-script boilerplate (no manual PID tracking or custom cleanup traps)
#   - wait_any_exit propagates the first failing child's exit code and lets the
#     EXIT trap tear down the rest, so failures and Ctrl+C behave predictably
#
# Usage:
#   source "$(dirname "$(readlink -f "$0")")/../common/launch_utils.sh"
#   # or with SCRIPT_DIR already set:
#   source "$SCRIPT_DIR/../common/launch_utils.sh"
#
# Constants:
#   EXAMPLE_PROMPT         Default example prompt for curl commands (LLM / embedding)
#   EXAMPLE_PROMPT_VISUAL  Default example prompt for image / video generation
#
# Requires: bash 4.3+ (wait -n)
#
# Functions:
#   print_launch_banner    Print startup banner with model info and example curl
#   print_curl_footer      Print a custom curl example with standard framing (heredoc)
#   wait_any_exit          Wait for any background process to exit, propagate its code

if [[ "${BASH_VERSINFO[0]}" -lt 4 || ( "${BASH_VERSINFO[0]}" -eq 4 && "${BASH_VERSINFO[1]}" -lt 3 ) ]]; then
    echo "launch_utils.sh requires bash 4.3+ (for wait -n), found ${BASH_VERSION}" >&2
    exit 1
fi

EXAMPLE_PROMPT="Who is the tennis GOAT: Federer, Djokovic, or Nadal?"
EXAMPLE_PROMPT_VISUAL="A golden retriever riding a skateboard through a neon-lit city"

# wait_any_exit
#
# Waits for ANY backgrounded process to exit and propagates its exit code.
# Call this as the LAST line of every launch script, after backgrounding
# all processes (including the one that would otherwise run in the foreground).
#
# Why this is better than tracking PIDs manually or running in the foreground:
#   Foreground pattern:  if the frontend crashes, the script blocks on the
#   foreground worker and never notices until that worker also exits.
#   Manual PIDs:  requires bookkeeping ($DYNAMO_PID, $PREFILL_PID, ...),
#   a custom cleanup() function, and `wait $PID` only watches one process.
#   wait -n watches ALL children and returns as soon as ANY child dies, so
#   failures are detected immediately regardless of which process it was.
#
# Signal handling:
#   SIGTERM/SIGINT exit 0 (clean shutdown).  External cleanup (e.g. a test
#   harness signalling the process group) otherwise interrupts wait -n and
#   returns 143, which under set -e looks like a test failure.  Real errors
#   from child processes (OOM, Python exceptions, etc.) still propagate.
#
# Usage:
#   python -m dynamo.frontend &
#   python -m dynamo.vllm --model "$MODEL" &
#   wait_any_exit

# Signals the pgid, reaps tracked jobs, then exits with the given code.
# Without explicit reaping, children reparent as zombies under a non-reaping
# subreaper (pytest in CI), and the harness's pgid liveness check
# (_terminate_process_group in managed_process.py) burns 8s on them.
dynamo_reap_and_exit() {
    local _rc=${1:-0}
    # Shield bash from its own pgid TERM/INT; otherwise the kill below
    # boomerangs and re-enters this trap.
    trap '' TERM INT
    # Always signal the pgid, not just when `jobs -p` is non-empty. A
    # direct child reaped by `wait -n` upstream leaves jobs -p empty,
    # but its subprocesses may still be alive in our pgid.
    kill -TERM 0 2>/dev/null || true
    wait 2>/dev/null || true
    exit "$_rc"
}

# EXIT-trap helper; captures $? before echo clobbers it.
dynamo_exit_trap() {
    local _rc=$?
    echo "Cleaning up..."
    dynamo_reap_and_exit "$_rc"
}

wait_any_exit() {
    trap 'dynamo_reap_and_exit 0' TERM INT
    if ! jobs -p | grep -q .; then
        echo "wait_any_exit: no background processes found (script bug: did you forget '&'?)" >&2
        exit 1
    fi
    # `|| _rc=$?` keeps set -e from swallowing the child's exit code.
    local _rc=0
    wait -n || _rc=$?
    echo "A background process exited with code $_rc"
    # Backstop signal trap; double quotes bake _rc in before it goes out of scope.
    # shellcheck disable=SC2064  # intentional expand-at-set-time
    trap "dynamo_reap_and_exit $_rc" TERM INT
    dynamo_reap_and_exit "$_rc"
}

# print_launch_banner [flags] <title> <model> <port> [extra_info_lines...]
#
# Prints a startup banner with model/frontend info and an example curl command.
#
# Flags (must come before positional args):
#   --multimodal       Use a multimodal (image_url) curl example (max_tokens=50)
#   --max-tokens N     Override max_tokens in the curl example (default: 32)
#   --no-curl          Print only the banner, skip the example curl section
#
# Positional args:
#   title              Banner title, e.g. "Launching Aggregated Serving (1 GPU)"
#   model              Model name, e.g. "$MODEL"
#   port               HTTP port, e.g. "$HTTP_PORT"
#   extra_info_lines   Optional extra lines printed below "Frontend:" (one per arg)
#
# Examples:
#   # Standard text serving
#   print_launch_banner "Launching Aggregated Serving (1 GPU)" "$MODEL" "$HTTP_PORT"
#
#   # With extra info
#   print_launch_banner "Launching Disagg on Same GPU" "$MODEL" "$HTTP_PORT" \
#       "GPU Mem:     0.09 per worker (4 GiB each)"
#
#   # Multimodal
#   print_launch_banner --multimodal "Launching Multimodal" "$MODEL" "$HTTP_PORT"
#
#   # Banner only (script prints its own curl or conditionally skips)
#   print_launch_banner --no-curl "Launching DSR1 (Multi-Node)" "$MODEL" "$HTTP_PORT" \
#       "Nodes:       $NUM_NODES" \
#       "Node rank:   $NODE_RANK"
print_launch_banner() {
    local _curl_type="text"
    local _max_tokens=32
    local _no_curl=false

    while [[ "${1:-}" == --* ]]; do
        case "$1" in
            --multimodal) _curl_type="multimodal"; _max_tokens=50; shift ;;
            --max-tokens) _max_tokens="$2"; shift 2 ;;
            --no-curl)    _no_curl=true; shift ;;
            *) break ;;
        esac
    done

    local _title="$1"
    local _model="$2"
    local _port="$3"
    shift 3

    echo "=========================================="
    echo "$_title"
    echo "=========================================="
    echo "Model:       $_model"
    echo "Frontend:    http://localhost:$_port"

    local _seq_len="${MAX_MODEL_LEN:-${CONTEXT_LENGTH:-${MAX_SEQ_LEN:-}}}"
    local _mem_args="${GPU_MEM_ARGS:-}"
    [[ -n "$_seq_len" ]] && echo "Max seq len: $_seq_len"
    [[ -n "$_mem_args" ]] && echo "GPU mem:     $_mem_args"

    for _line in "$@"; do
        echo "$_line"
    done
    echo "=========================================="

    if [[ "$_no_curl" == true ]]; then
        return
    fi

    echo ""
    echo "Example test command:"
    echo ""

    if [[ "$_curl_type" == "multimodal" ]]; then
        cat <<CURL_EOF
  curl http://localhost:${_port}/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{
      "model": "${_model}",
      "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png"}}
      ]}],
      "max_tokens": ${_max_tokens}
    }'
CURL_EOF
    else
        cat <<CURL_EOF
  curl http://localhost:${_port}/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{
      "model": "${_model}",
      "messages": [{"role": "user", "content": "Hello!"}],
      "max_tokens": ${_max_tokens}
    }'
CURL_EOF
    fi

    echo ""
    echo "=========================================="
}

# wait_for_ready <url> [timeout_seconds]
#
# Polls an HTTP endpoint until it returns 200 or timeout is reached.
# Useful for waiting for a worker to finish loading before starting the
# next one (e.g. disaggregated same-GPU deployments where concurrent
# model loading causes OOM).
#
# Args:
#   url              HTTP URL to poll (e.g. http://localhost:8081/health)
#   timeout_seconds  Max seconds to wait (default: 30)
#
# Returns 0 on success, 1 on timeout.
wait_for_ready() {
    local _url="$1"
    local _timeout="${2:-30}"
    local _start=$SECONDS
    echo "Polling $_url (timeout: ${_timeout}s)..."
    while (( SECONDS - _start < _timeout )); do
        if curl -sf --max-time 2 "$_url" > /dev/null 2>&1; then
            echo "Ready after $(( SECONDS - _start ))s"
            return 0
        fi
        sleep 1
    done
    echo "WARNING: $_url not ready after ${_timeout}s" >&2
    return 1
}

# pick_worker_module <legacy_module> <unified_module> "$@"
#
# Strips `--unified` from argv and selects the matching `python -m <module>`
# target. Sets two globals:
#
#   WORKER_MODULE    The chosen module (legacy by default; unified if --unified seen).
#   REMAINING_ARGS   Array of surviving argv. Caller does `set -- "${REMAINING_ARGS[@]}"`.
#
# Why this lives here:
#   Every disagg launch script needs the same switch. Inlining it three times
#   means three places to keep in sync; centralising here means the
#   "consume --unified BEFORE installing the EXIT trap" discipline is enforced
#   by construction. Other flags (--enable-otel, --model-name, ...) stay in
#   the caller's own arg loop so engine-specific options keep working.
#
# Usage:
#   pick_worker_module dynamo.vllm dynamo.vllm.unified_main "$@"
#   set -- "${REMAINING_ARGS[@]}"
#   trap 'echo Cleaning up...; kill 0' EXIT
#   # ... caller's own argument parsing on the surviving $@ ...
pick_worker_module() {
    local _legacy="$1"
    local _unified="$2"
    shift 2
    WORKER_MODULE="$_legacy"
    REMAINING_ARGS=()
    for _arg in "$@"; do
        if [[ "$_arg" == "--unified" ]]; then
            WORKER_MODULE="$_unified"
        else
            REMAINING_ARGS+=("$_arg")
        fi
    done
}

# print_curl_footer
#
# Prints a custom curl example wrapped in the standard framing (matching
# print_launch_banner's built-in curl output). Reads the curl command from
# stdin so callers can use a heredoc -- no quoting issues with embedded
# double quotes, variable interpolation, etc.
#
# Pair with print_launch_banner --no-curl for non-standard endpoints
# (images, video, embeddings, etc.) that need a custom request body.
#
# Usage:
#   print_launch_banner --no-curl "Launching Image Diffusion" "$MODEL" "$PORT"
#   print_curl_footer <<CURL
#   curl http://localhost:${PORT}/v1/images/generations \\
#     -H 'Content-Type: application/json' \\
#     -d '{
#       "model": "${MODEL}",
#       "prompt": "A cat on a skateboard"
#     }'
#   CURL
print_curl_footer() {
    echo ""
    echo "Example test command:"
    echo ""
    cat
    echo ""
    echo "=========================================="
}
