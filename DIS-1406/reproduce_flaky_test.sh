#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Router E2E flaky reproducer (Docker-per-iteration).
#
# - Runs pytest in a fresh container per iteration.
# - Writes:
#   - `logs/<serial|nN>_<set>/runNNN/pytest.log` (pytest console output)
#   - `logs/<serial|nN>_<set>/runNNN/.exit_code` (exit code for that run)
#   - `logs/<serial|nN>_<set>/runNNN/test_*/...` (per-test logs from ManagedProcess)
#
# Chunking mode (recommended for 50+ runs):
#   - Pass `--chunk-size 5` and the script will fork children (each child runs one chunk),
#     wait for them all, then regenerate `summary.txt`/`results.txt` once at the end.
#
# SIGHUP behavior:
#   - Kills all forked children (and any active docker container in this process),
#     then `chown -R` the log directory so you can delete/inspect logs immediately.

# Don't use set -e - we want to continue on test failures

usage() {
echo "Usage: $0 -p <parallelism> [-t set] [-s start] [-e end] [--chunk-size N]"
echo ""
echo "Required:"
echo "  -p, --parallelism   'serial' or a number (e.g., 1, 4, 16, 32)"
echo ""
echo "Optional:"
echo "  -t, --set           Set name for organizing multiple runs (e.g., A, run2, test)"
echo "                      Creates logs/n4_A/, logs/serial_run2/, etc."
echo "  -s, --start         Start iteration (default: 1)"
echo "  -e, --end           End iteration (default: 100)"
echo "      --chunk-size    Run in forked chunk mode (e.g., 5 => 1-5, 6-10, ...)"
echo "      --chunk-parallel Max number of concurrent chunks (default: unlimited)"
echo "      --no-regenerate Internal: child mode (skip summary regeneration)"
echo "      --pytest-nodeid  Pytest nodeid to run (e.g., 'tests/router/test_router_e2e_with_mockers.py::test_router_decisions')"
echo "      --dry-run       Show what would be executed without actually running"
echo "  -h, --help          Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 -p serial              # logs/serial/"
    echo "  $0 -p serial -t A         # logs/serial_A/"
    echo "  $0 -p 4                   # logs/n4/"
    echo "  $0 -p 4 -t run2           # logs/n4_run2/"
    echo "  $0 -p 4 -s 25             # resume from iteration 25"
    echo ""
    echo "Chunking (forks + waits, then regenerates once):"
    echo "  $0 -p serial -t fix8 -s 1 -e 50 --chunk-size 5"
    exit 1
}

# Default values
PARALLELISM=""
SET_NAME=""
START=1
END=100
IMAGE="dynamo:latest-dev"
DOCKER_GPU_ARGS=()
CHUNK_SIZE=""
CHUNK_PARALLEL=0
NO_REGENERATE=0
PYTEST_NODEID=""
DRY_RUN=0
CHILD_PIDS=()

# Optional: expose GPU to the container (useful on GPU nodes).
# Example:
#   export DYNAMO_DOCKER_GPU_ARGS="--gpus all"
if [ -n "${DYNAMO_DOCKER_GPU_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    DOCKER_GPU_ARGS=(${DYNAMO_DOCKER_GPU_ARGS})
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--parallelism)
            PARALLELISM="$2"
            shift 2
            ;;
        -t|--set)
            SET_NAME="$2"
            shift 2
            ;;
        -s|--start)
            START="$2"
            shift 2
            ;;
        -e|--end)
            END="$2"
            shift 2
            ;;
        -i|--image)
            IMAGE="$2"
            shift 2
            ;;
        --pytest-nodeid)
            PYTEST_NODEID="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --chunk-parallel)
            CHUNK_PARALLEL="$2"
            shift 2
            ;;
        --no-regenerate)
            NO_REGENERATE=1
            shift 1
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "ERROR: Unknown option: $1"
            usage
            ;;
    esac
done

# Validate parallelism argument
if [ -z "$PARALLELISM" ]; then
    echo "ERROR: Missing required argument: -p <parallelism>"
    echo ""
    usage
fi

if [ "$PARALLELISM" != "serial" ] && ! [[ "$PARALLELISM" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Invalid parallelism '$PARALLELISM' - must be 'serial' or a number"
    exit 1
fi

# Validate start/end
if ! [[ "$START" =~ ^[0-9]+$ ]] || ! [[ "$END" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Start and end must be numbers"
    exit 1
fi

if [ "$START" -gt "$END" ]; then
    echo "ERROR: Start ($START) cannot be greater than end ($END)"
    exit 1
fi

if [ -n "$CHUNK_SIZE" ] && ! [[ "$CHUNK_SIZE" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --chunk-size must be a number"
    exit 1
fi

if ! [[ "$CHUNK_PARALLEL" =~ ^[0-9]+$ ]]; then
    echo "ERROR: --chunk-parallel must be a number"
    exit 1
fi

# Get HF_TOKEN
if [ -z "$HF_TOKEN" ] && [ -f ~/.cache/huggingface/token ]; then
    export HF_TOKEN=$(cat ~/.cache/huggingface/token)
fi

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set and ~/.cache/huggingface/token not found"
    exit 1
fi

# Ensure docker inherits HF_TOKEN without echoing it in xtrace output.
export HF_TOKEN

# Determine log directory and pytest args
if [ "$PARALLELISM" = "serial" ]; then
    BASE_DIR="serial"
    PYTEST_ARGS=""
    LABEL="serial"
else
    BASE_DIR="n${PARALLELISM}"
    PYTEST_ARGS="-n ${PARALLELISM}"
    LABEL="-n ${PARALLELISM}"
fi

# Add set name suffix if provided
if [ -n "$SET_NAME" ]; then
    LOG_DIR="logs/${BASE_DIR}_${SET_NAME}"
    LABEL="$LABEL (set: $SET_NAME)"
else
    LOG_DIR="logs/${BASE_DIR}"
fi

# Create log directory
mkdir -p "$LOG_DIR"

# Kill any forked children and active docker container (if any).
kill_children() {
    if [ "${#CHILD_PIDS[@]}" -eq 0 ]; then
        return 0
    fi

    for pid in "${CHILD_PIDS[@]}"; do
        kill "$pid" >/dev/null 2>&1 || true
    done

    # Give children a moment to exit, then force kill.
    sleep 1
    for pid in "${CHILD_PIDS[@]}"; do
        kill -9 "$pid" >/dev/null 2>&1 || true
    done
}

# SIGHUP handler:
# - kill all forked children
# - kill our active docker container (if any)
# - chown -R log dir (so rm works immediately)
fix_ownership() {
    echo ""
    echo ">>> SIGHUP received: killing children + fixing ownership of $LOG_DIR"
    kill_children
    # Also kill any containers spawned by children (chunked mode).
    container_prefix="router_${SET_NAME:-${BASE_DIR}}_run"
    docker ps --format "{{.Names}}" | awk -v p="$container_prefix" 'index($0, p) == 1 {print}' | xargs -r docker kill 2>/dev/null || true
    if [ -n "${CONTAINER_NAME:-}" ]; then
        docker kill "$CONTAINER_NAME" 2>/dev/null || true
    fi
    sudo chown -R "$(whoami)" "$LOG_DIR" 2>/dev/null || true
    echo ">>> Ownership fixed"
}
trap 'fix_ownership; exit 129' SIGHUP

regenerate_results() {
    echo ""
    echo "Regenerating results (with flock)..."
    (
        flock -x 200

        > "$LOG_DIR/results.txt"
        > "$LOG_DIR/summary.txt"

        for run_dir in $(ls -d "$LOG_DIR"/run[0-9][0-9][0-9] 2>/dev/null | sort); do
            run_name=$(basename "$run_dir")
            exit_file="$run_dir/.exit_code"

            if [ -f "$exit_file" ]; then
                exit_code=$(cat "$exit_file")
                if [ "$exit_code" -eq 0 ]; then
                    echo "PASSED" >> "$LOG_DIR/results.txt"
                    echo "$run_name: PASSED" >> "$LOG_DIR/summary.txt"
                else
                    echo "FAILED" >> "$LOG_DIR/results.txt"
                    echo "$run_name: FAILED (exit $exit_code)" >> "$LOG_DIR/summary.txt"
                fi
            fi
        done

    ) 200>"$LOG_DIR/.lock"
}

print_final_summary() {
    echo ""
    echo "================================================================================"
    echo "FINAL RESULTS"
    echo "================================================================================"
    PASSED=$(grep -c "^PASSED$" "$LOG_DIR/results.txt" 2>/dev/null || true)
    FAILED=$(grep -c "^FAILED$" "$LOG_DIR/results.txt" 2>/dev/null || true)
    PASSED=${PASSED:-0}
    FAILED=${FAILED:-0}
    TOTAL_RESULTS=$((PASSED + FAILED))
    echo "Range: $START-$END ($TOTAL runs this session)"
    echo "Total results in directory: $TOTAL_RESULTS"
    if [ "$TOTAL_RESULTS" -gt 0 ]; then
        echo "Passed: $PASSED ($(( PASSED * 100 / TOTAL_RESULTS ))%)"
        echo "Failed: $FAILED ($(( FAILED * 100 / TOTAL_RESULTS ))%)"
    fi
    echo ""
    echo "=== Failed runs ==="
    grep "FAILED" "$LOG_DIR/summary.txt" 2>/dev/null || echo "None"
    echo ""
    echo "=== Error breakdown ==="
    grep -rh "FAILED tests/" "$LOG_DIR"/run*/pytest.log 2>/dev/null | sort | uniq -c | sort -rn || echo "None"
    echo ""
    echo "Completed: $(date)"
    echo "Logs: $LOG_DIR/"
    echo "================================================================================"
}

run_chunked() {
    echo "================================================================================"
    echo "Chunked Stress Test: iterations $START-$END, chunk_size=$CHUNK_SIZE, $LABEL"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "*** DRY RUN MODE - No tests will be executed ***"
    fi
    echo "================================================================================"
    echo "Log directory: $LOG_DIR"
    echo "Started: $(date)"
    echo "================================================================================"
    echo ""

    local script_path="$0"
    local extra_args=()
    if [ -n "$SET_NAME" ]; then
        extra_args+=("-t" "$SET_NAME")
    fi
    if [ -n "$PYTEST_NODEID" ]; then
        extra_args+=("--pytest-nodeid" "$PYTEST_NODEID")
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        extra_args+=("--dry-run")
    fi

    for chunk_start in $(seq "$START" "$CHUNK_SIZE" "$END"); do
        local chunk_end=$((chunk_start + CHUNK_SIZE - 1))
        if [ "$chunk_end" -gt "$END" ]; then
            chunk_end="$END"
        fi

        echo ">>> Chunk ${chunk_start}-${chunk_end} would run:"
        echo "    $script_path -p $PARALLELISM ${extra_args[*]} -s $chunk_start -e $chunk_end -i $IMAGE --no-regenerate"

        if [ "$DRY_RUN" -eq 0 ]; then
            "$script_path" -p "$PARALLELISM" "${extra_args[@]}" -s "$chunk_start" -e "$chunk_end" -i "$IMAGE" --no-regenerate &
            echo ">>> Chunk ${chunk_start}-${chunk_end} pid=$!"
            CHILD_PIDS+=("$!")

            if [ "$CHUNK_PARALLEL" -gt 0 ]; then
                while [ "$(jobs -pr | wc -l)" -ge "$CHUNK_PARALLEL" ]; do
                    wait -n || true
                done
            fi
        fi
    done

    if [ "$DRY_RUN" -eq 1 ]; then
        echo ""
        echo "*** DRY RUN COMPLETE - No tests were executed ***"
        return 0
    fi

    echo ">>> Waiting for ${#CHILD_PIDS[@]} chunk(s) to finish..."

    local rc=0
    for pid in "${CHILD_PIDS[@]}"; do
        if ! wait "$pid"; then
            rc=1
        fi
    done

    regenerate_results
    print_final_summary

    if grep -q "^FAILED$" "$LOG_DIR/results.txt" 2>/dev/null; then
        return 1
    fi
    return "$rc"
}

if [ -n "$CHUNK_SIZE" ] && [ "$NO_REGENERATE" -eq 0 ]; then
    run_chunked
    exit $?
fi

TOTAL=$((END - START + 1))

echo "================================================================================"
echo "Stress Test: iterations $START-$END ($TOTAL runs), $LABEL"
if [ "$DRY_RUN" -eq 1 ]; then
    echo "*** DRY RUN MODE - No tests will be executed ***"
fi
echo "================================================================================"
echo "Log directory: $LOG_DIR"
echo "Started: $(date)"
echo "================================================================================"
echo ""

# Run iterations
for i in $(seq $START $END); do
    echo ""
    echo "================================================================================"
    echo "=== Iteration $i/$END - $(date) ==="
    echo "================================================================================"

    # Run test in fresh container with isolated bridge network
    # DYN_TEST_OUTPUT_PATH tells tests (conftest.py logger + ManagedProcess) where to write logs
    # Log structure: logs/n4/run001/pytest.log (pytest output) + logs/n4/run001/test_*/  (test outputs)
    RUN_NUM=$(printf "%03d" $i)
    RUN_DIR="$LOG_DIR/run${RUN_NUM}"
    mkdir -p "$RUN_DIR"

    # Container name: router_<set>_run<num> (e.g., router_fix3_run001)
    # Sanitize SET_NAME: replace invalid Docker name characters with underscores
    # Docker allows: [a-zA-Z0-9][a-zA-Z0-9_.-]
    SANITIZED_SET_NAME=$(echo "${SET_NAME:-${BASE_DIR}}" | sed 's/[^a-zA-Z0-9_.-]/_/g')
    CONTAINER_NAME="router_${SANITIZED_SET_NAME}_run${RUN_NUM}"
    if [ -n "$PYTEST_NODEID" ]; then
        PYTEST_CMD="pytest $PYTEST_NODEID $PYTEST_ARGS --basetemp=/tmp/pytest --durations=0 --timeout=600"
    else
        PYTEST_CMD="pytest tests/router/test_router_e2e_with_mockers.py $PYTEST_ARGS --basetemp=/tmp/pytest --durations=0 --timeout=600"
    fi
    DOCKER_TIMEOUT=660  # 11 min (pytest timeout + buffer for container startup)

    # Random delay (0-5s) to stagger parallel container starts
    RANDOM_DELAY=$((RANDOM % 6))

    if [ "$DRY_RUN" -eq 1 ]; then
        echo ">>> Would wait ${RANDOM_DELAY}s before starting container"
        echo ">>> Would run command:"
        echo "    timeout $DOCKER_TIMEOUT docker run --rm ${DOCKER_GPU_ARGS[*]} --name $CONTAINER_NAME --network bridge --shm-size=10G \\"
        echo "        -e HF_TOKEN -e HF_HUB_OFFLINE=1 -e DYN_TEST_OUTPUT_PATH=/workspace/$RUN_DIR \\"
        echo "        -v /home/keivenc/dynamo/dynamo2:/workspace -v /home/keivenc/.cache:/root/.cache:ro -w /workspace \\"
        echo "        $IMAGE bash -c 'set -x; echo \"=== Command: $PYTEST_CMD ===\" && echo \"=== Started: \$(date -Iseconds) ===\" && $PYTEST_CMD 2>&1 | tee /workspace/$RUN_DIR/pytest.log; exit \${PIPESTATUS[0]}'"
        continue
    fi

    echo ">>> Waiting ${RANDOM_DELAY}s before starting container..."
    sleep $RANDOM_DELAY

    # Print the full expanded docker command for debugging.
    # Keep secrets out of xtrace: pass HF_TOKEN by name only (docker inherits it).
    set -x
    timeout $DOCKER_TIMEOUT docker run --rm "${DOCKER_GPU_ARGS[@]}" --name "$CONTAINER_NAME" --network bridge --shm-size=10G \
        --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=65536:65536 \
        -e HF_TOKEN \
        -e HF_HUB_OFFLINE=1 \
        -e DYN_ROUTER_ENABLE_PREFIX_CACHE \
        -e DYN_ROUTER_PREFIX_CACHE_SIZE \
        -e DYN_TEST_OUTPUT_PATH="/workspace/$RUN_DIR" \
        -v "$(pwd):/workspace" \
        -v "$HOME/.cache:/root/.cache:ro" \
        -w /workspace \
        "$IMAGE" \
        bash -c "set -x; echo '=== Command: $PYTEST_CMD ===' && echo '=== Started: '\$(date -Iseconds)' ===' && $PYTEST_CMD 2>&1 | tee /workspace/$RUN_DIR/pytest.log; exit \${PIPESTATUS[0]}"
    # IMPORTANT: capture the docker/timeout exit code BEFORE any other command runs.
    # (If we run `set +x` first, `$?` becomes the status of `set +x`, which is always 0.)
    EXIT_CODE=$?
    set +x

    # Handle timeout (exit code 124)
    if [ $EXIT_CODE -eq 124 ]; then
        echo ">>> TIMEOUT: Run $i exceeded ${DOCKER_TIMEOUT}s - container killed"
        docker kill "$CONTAINER_NAME" 2>/dev/null || true
    fi

    # Fix ownership of files created by Docker (runs as root)
    sudo chown -R "$(whoami)" "$RUN_DIR" 2>/dev/null || true

    # Store exit code in run directory (for later regeneration)
    echo "$EXIT_CODE" > "$RUN_DIR/.exit_code"

    # Progress update every 10 iterations
    if [ $((i % 10)) -eq 0 ]; then
        # Quick count from .exit_code files
        PASSED=$(find "$LOG_DIR" -name ".exit_code" -exec cat {} \; 2>/dev/null | grep -c "^0$" || echo 0)
        FAILED=$(find "$LOG_DIR" -name ".exit_code" -exec cat {} \; 2>/dev/null | grep -cv "^0$" || echo 0)
        echo ""
        echo ">>> Progress: $i/$END | Passed: $PASSED | Failed: $FAILED"
        echo ""
    fi
done

if [ "$NO_REGENERATE" -eq 0 ]; then
    regenerate_results
    print_final_summary
fi
