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
# Functions:
#   print_launch_banner    Print startup banner with model info and example curl
#   print_curl_footer      Print a custom curl example with standard framing (heredoc)
#   wait_any_exit          Wait for any background process to exit, propagate its code

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
#   SIGTERM/SIGINT are trapped to exit 0 (clean shutdown).  Without this,
#   external cleanup (e.g. a test harness sending SIGTERM to the process
#   group) interrupts wait -n, which returns 143 (128+15).  Combined with
#   set -e, that non-zero code looks like a test failure.  Trapping TERM/INT
#   makes external teardown exit cleanly while still propagating real errors
#   (OOM, Python exceptions, etc.) from child processes.
#
# The EXIT trap (set at the top of each script) still fires when this function
# calls exit, tearing down the remaining processes via kill 0.
#
# Usage:
#   python -m dynamo.frontend &
#   python -m dynamo.vllm --model "$MODEL" &
#   wait_any_exit
wait_any_exit() {
    trap 'exit 0' TERM INT
    if ! jobs -p | grep -q .; then
        echo "wait_any_exit: no background processes found (script bug: did you forget '&'?)" >&2
        exit 1
    fi
    wait -n
    local _rc=$?
    echo "A background process exited with code $_rc"
    exit "$_rc"
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
