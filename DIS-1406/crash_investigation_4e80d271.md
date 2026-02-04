# Flaky Test Investigation: Router E2E Tests

**Test Suite:** `tests/router/test_router_e2e_with_mockers.py`
**Date:** 2026-02-03

---

## 1. Problem Statement

### CI Failures to Reproduce

- https://github.com/ai-dynamo/dynamo/actions/runs/21553857421/job/62106847745
- https://github.com/ai-dynamo/dynamo/actions/runs/21549672070/job/62096453750

### Observed Errors in CI

The CI shows intermittent failures with these error patterns:

```
Fatal Python error: Aborted
[gw6] node down: Not properly terminated
worker 'gw6' crashed while running 'tests/router/test_router_e2e_with_mockers.py::test_kv_push_router_bindings[True-tcp]'
```

### Error Types Observed

#### 1. Network Timeout
- **Cause:** Transient HuggingFace API failure
- **Severity:** Low (infrastructure, not test bug)
- **Example:**
```
Failed to fetch model 'Qwen/Qwen3-0.6B' from HuggingFace
error sending request for url (https://huggingface.co/api/models/Qwen/Qwen3-0.6B/revision/main)
```

#### 2. Routing Assertion
- **Cause:** Race condition in KV router under parallel load
- **Severity:** Medium (test logic issue under contention)
- **Example:**
```python
AssertionError: Expected all responses to have the same decode_worker_id
(due to prefix reuse), but found 2 unique values: {id1, id2}
# Location: tests/router/common.py:133
```

#### 3. Fatal Crash (SIGABRT)
- **Cause:** PyO3/Tokio runtime interaction issue
- **Severity:** High (process crash)
- **Example:**
```
Fatal Python error: Aborted
[gw6] node down: Not properly terminated
worker 'gw6' crashed while running 'tests/router/test_router_e2e_with_mockers.py::test_kv_push_router_bindings[True-tcp]'
```

---

## 2. How to Reproduce

### Prerequisites: Build Docker Image

Build the test image using `container/build.sh` (same as CI):

```bash
cd /path/to/dynamo2

# Build the dev image (same command as CI job 62106847745)
./container/build.sh --tag dynamo:latest-dev --target dev --framework none --enable-kvbm --enable-media-ffmpeg

# This creates: dynamo:latest-dev
```

### Run Tests at Different Parallelism Levels

Use `reproduce_flaky_test.sh` which runs N iterations with a fresh container per iteration:

```bash
cd /path/to/dynamo2

# Usage: ./reproduce_flaky_test.sh [iterations] [parallelism]

# Serial (no xdist) - ~8 hours for 100 iterations
./reproduce_flaky_test.sh 100

# -n 16 parallelism - ~1 hour for 100 iterations
./reproduce_flaky_test.sh 100 16

# -n 32 parallelism - ~30 min for 100 iterations
./reproduce_flaky_test.sh 100 32
```

The script:
1. Launches a fresh Docker container for each iteration
2. Runs pytest with specified parallelism
3. Saves individual logs to `logs/{serial,n16,n32}/iter_N.log`
4. Tracks pass/fail in `results.txt` and `summary.txt`
5. Prints final summary with error breakdown

### Analyze Results

```bash
# Count pass/fail for a given parallelism level (e.g., n16)
grep -l "passed" logs/n16/iter_*.log | wc -l          # passed iterations
grep -l "failed" logs/n16/iter_*.log | wc -l          # failed iterations

# Error details
grep -h "FAILED" logs/n16/iter_*.log | sort | uniq -c | sort -rn
grep -h "AssertionError" logs/n16/iter_*.log | head -5
grep -h "Fatal Python error" logs/n16/iter_*.log | head -5
```

### Files

| File | Purpose |
|------|---------|
| `reproduce_flaky_test.sh` | Stress test script |
| `logs/serial/` | Serial results (100 iterations) |
| `logs/n16/` | `-n 16` results (100 iterations) |
| `logs/n32/` | `-n 32` results (100 iterations) |
| `logs/*/iter_N.log` | Individual iteration output |
| `logs/*/full_output.log` | Combined output with timestamps |

---

## 3. Root Cause Analysis

### Why Serial Works but xdist Fails

The "Fatal Python error: Aborted" crashes occur because **pytest-xdist workers share global static Tokio runtime state across sequential tests within the same worker process**.

| Factor | Serial | xdist |
|--------|--------|-------|
| Process per test | Fresh container each iteration | Same process per worker (shares N tests) |
| Global RT state | Fresh initialization | Reused from previous test |
| Cancellation tokens | Fresh | May be cancelled from previous test |
| TCP server port | Released between tests | Bound for process lifetime |
| Event loop | Fresh | May be from wrong test |

### Global Static State (Root Cause)

#### 1. Global Tokio Runtime (`lib/runtime/src/worker.rs:31-33`)

```rust
static RT: OnceCell<tokio::runtime::Runtime> = OnceCell::new();
static RTHANDLE: OnceCell<tokio::runtime::Handle> = OnceCell::new();
static INIT: OnceCell<Mutex<Option<JoinHandle<...>>>> = OnceCell::new();
```

**Problem:** `OnceCell` is initialized once per process. In xdist:
- Test A creates `DistributedRuntime` -> initializes `RT`
- Test A shutdown cancels tokens but `RT` persists
- Test B reuses `RT` with stale cancellation tokens

#### 2. Global TCP Server (`lib/runtime/src/pipeline/network/manager.rs:38-39`)

```rust
static GLOBAL_TCP_SERVER: OnceCell<Arc<SharedTcpServer>> = OnceCell::new();
static ACTUAL_TCP_RPC_PORT: OnceLock<u16> = OnceLock::new();
```

**Problem:** TCP server port is bound once, never released.

#### 3. PyO3 Async Runtime (`lib/bindings/python/rust/lib.rs:563-569`)

```rust
pyo3_async_runtimes::tokio::init_with_runtime(primary)
```

**Problem:** Python event loop association is initialized once. Subsequent tests may use wrong event loop.

### Crash Sequence

1. Test A creates `DistributedRuntime` -> initializes global `OnceCell` (first time)
2. Test A runs, spawns async tasks on the global Tokio runtime
3. Test A calls `shutdown()` -> cancels cancellation tokens, but `RT` persists
4. Test B creates `DistributedRuntime` -> reuses existing `RT` (OnceCell already set)
5. Test B calls `block_on()` from constructor on a runtime with stale state
6. Rust code hits inconsistent state -> `SIGABRT`

### Evidence from Logs

The `-n 16` test shows:
```
Fatal Python error: Aborted
[gw5] node down: Not properly terminated
```

The crash pattern:
1. `Fatal Python error: Aborted` - Rust code called `abort()` or panicked
2. `node down: Not properly terminated` - xdist detected worker crash
3. Worker number varies (`gw0`, `gw3`, `gw5`, etc.) - any worker can crash

---

## 4. Results

### Results Summary

| Parallelism | Iterations | Passed | Failed | Pass Rate |
|-------------|------------|--------|--------|-----------|
| Serial | 100 | 99 | 1 | **99%** |
| `-n 16` | 100 | - | - | TBD |
| `-n 32` | 100 | - | - | TBD |

### Error Breakdown by Parallelism

#### Serial (100 iterations)

| Error Type | Count | % | Sample Message |
|------------|-------|---|----------------|
| Network timeout | 1 | 1% | `Failed to fetch model 'Qwen/Qwen3-0.6B' from HuggingFace: error sending request` |
| Routing assertion | 0 | 0% | - |
| Fatal crash (SIGABRT) | 0 | 0% | - |
| **Total failures** | **1** | **1%** | |

#### `-n 16` (100 iterations)

| Error Type | Count | % | Sample Message |
|------------|-------|---|----------------|
| Network timeout | - | - | - |
| Routing assertion | - | - | - |
| Fatal crash (SIGABRT) | - | - | - |
| **Total failures** | **-** | **-** | |

#### `-n 32` (100 iterations)

| Error Type | Count | % | Sample Message |
|------------|-------|---|----------------|
| Network timeout | - | - | - |
| Routing assertion | - | - | - |
| Fatal crash (SIGABRT) | - | - | - |
| **Total failures** | **-** | **-** | |

---

## 5. Conclusion and Potential Fixes

### Conclusion

- **Serial execution is stable (99%)** - Fresh process per iteration, no global state issues
- **Parallel execution crashes** - Global static Tokio runtime state causes SIGABRT
- **Root cause:** `OnceCell` statics in `worker.rs` are not reset between tests
- **Recommended fix:** Use `--forked` for CI, refactor global state long-term

### Potential Fixes

#### Option A: Process Isolation (Short-term)

Use `pytest-forked` to fork a new process per test:
```bash
pytest --forked tests/router/test_router_e2e_with_mockers.py
```
- No code changes required
- Slower but reliable

#### Option B: Add Reset Method (Medium-term)

Add `DistributedRuntime::reset()` for test cleanup that properly resets global state.

#### Option C: Refactor to Instance State (Long-term)

Replace global statics with instance-scoped state:
```rust
// Instead of:
static RT: OnceCell<Runtime> = OnceCell::new();

// Use:
struct Worker {
    runtime: Arc<Runtime>,  // Instance-scoped
}
```

---

## 6. TODO Tasks

### TODO 1: Add timeout to `ManagedProcess.__exit__()` wait call

**File:** `tests/utils/managed_process.py`
**Line:** 221
**CI log:** `~/.cache/dynamo-utils/raw-log-text/62377203601.log`

**Problem:** In `__exit__()`, after calling `terminate_process_tree()`, there's an unbounded `process.wait()` call:

```python
# Line 220-221
terminate_process_tree(process.pid, self._logger)
process.wait()  # <-- NO TIMEOUT - CAN HANG FOREVER!
```

Even though `terminate_process_tree()` sends SIGKILL, if the process is in **D-state** (uninterruptible sleep - e.g., stuck in GPU/CUDA operations, NCCL collectives, or kernel I/O), the process cannot be killed until the blocking kernel operation completes. The `wait()` call blocks indefinitely.

**Evidence from CI:** In job `62377203601`, `test_sglang_kv_router_basic[tcp]` hit its 150s pytest-timeout. The test was marked FAILED, but the cleanup hung for **~1 hour** (from `17:49:18` to `18:47:36`) with zero output until the GitHub Actions job was canceled. Remaining tests (72%-100%) never ran.

**Fix:**
```python
try:
    process.wait(timeout=10)
except subprocess.TimeoutExpired:
    self._logger.warning(
        "Process %s did not exit after SIGKILL (likely D-state), giving up",
        process.pid
    )
```

**Why SGLang workers might be in D-state:**
- Waiting for CUDA synchronization to complete
- GPU memory allocation/deallocation stuck
- NCCL collective operations hung (multi-GPU)
- Model loading from disk/network
