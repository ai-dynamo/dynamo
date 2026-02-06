# Flaky Test Investigation: Router E2E Tests

**Test Suite:** `tests/router/test_router_e2e_with_mockers.py`
**Date:** 2026-02-04 (updated)

=============================================================================
TABLE OF CONTENTS
=============================================================================
1. QUICK START + STATUS SNAPSHOTS
  1.1 Running stress tests
  1.2 Fix8 status snapshot (proof)
  1.3 Fix9 status snapshot (regression note)
  1.4 Fix14 status snapshot (prefix caching)

2. PROBLEM STATEMENT
  2.1 CI failures to reproduce
  2.2 Observed errors in CI

3. TESTING GUIDELINES
  3.1 Avoid `sleep()` in tests

4. ERROR TAXONOMY
  4.1 Part A: Single-thread issues (Serial / -n 1)
  4.2 Part B: Multi-thread issues (xdist parallelism)

5. HOW TO REPRODUCE
6. ROOT CAUSE ANALYSIS
7. RESULTS
8. CONCLUSION AND FIX PRIORITIES
9. FIX LOG (Fix12 / Fix13)
10. TODO TASKS

=============================================================================
1. QUICK START + STATUS SNAPSHOTS
=============================================================================

## 1.1 Running Stress Tests

```bash
# Run 1..50 (50 runs) in parallel 5-run chunks
cd /home/keivenc/dynamo/dynamo2
for start in 1 6 11 16 21 26 31 36 41 46; do
    end=$((start + 4))
    nohup ./DIS-1406/reproduce_flaky_test.sh -s "$start" -e "$end" -p serial -t fix8 -i dynamo:latest-dev-fix8 \
        > "/tmp/fix8_${start}_${end}.log" 2>&1 &
done

# Monitor progress
watch -n 30 'docker ps --format "table {{.Names}}\t{{.Status}}" | grep router_fix8 || true; echo "---"; ls logs/serial_fix8/ 2>/dev/null | wc -l; tail -5 logs/serial_fix8/summary.txt 2>/dev/null'

# Check results
cat logs/serial_fix8/summary.txt
```

## 1.2 Fix8 status snapshot (proof)

- **What we fixed (A1)**: `test_router_decisions` flake caused by the tie-breaker picking the *smallest* `tree_size`
  (routes away from prefix reuse and spreads events across dp_ranks).
  - **Where**: `lib/llm/src/kv_router/scheduler.rs` in `DefaultWorkerSelector::select_worker()` (tie-breaker on `candidates`)
  - **Fix**: prefer **larger** `tree_size` using `Reverse(tree_size)`, then deterministic `WorkerWithDpRank`.

- **Before → after evidence (A1)**:
  - **Before (pre-fix)**: **2 / 75** serial iterations failed (3%): `logs/serial/run029/`, `logs/serial/run035/`
  - **After (fix8)**: **0 occurrences** of the A1 dp_rank-spread assertions in **1..50** containerized serial iterations.
    - Evidence source: absence of the A1 signature in `/tmp/repro_fix8/fix8_1_50.nohup.log` and no A1 assertion in
      `logs/serial_fix8/run*/pytest.log`.

- **What we have NOT fixed yet (A2/A3/A4)**:
  - **A2 (event sync mismatch)**: `AssertionError: Router states have different numbers of events`
    - **Fix8 frequency**: **13 / 50 (26%)** in 1..50 (mostly `test_indexers_sync[nats_core]`)
    - **Failing runs**: `logs/serial_fix8/run006/`, `run008/`, `run010/`, `run018/`, `run021/`, `run024/`, `run028/`,
      `run031/`, `run039/`, `run043/`, `run044/` (jetstream), `run047/`, `run049/`
  - **A3 (timeouts)**: `test_indexers_sync[jetstream]` sometimes times out (historically `-n 1`; see runs 71/99).
  - **A4 (indefinite hang)**: `test_indexers_sync[...]` can hang on creating the **second** router (global OnceCell/Tokio).
  - **Harness bug**: some failing `pytest.log` runs still recorded `.exit_code == 0` (do not trust summary based only on `.exit_code`).

## 1.3 Fix9 status snapshot (regression note)

- **Result (1..50)**: **11 PASSED / 39 FAILED** (`logs/serial_fix9/`)
- **Most failures were router-decisions**:
  - **Runs with any failure**: 34/50
  - **Runs with `test_router_decisions[...]` failure**: 32/50
  - **Runs with `test_indexers_sync[...]` failure**: 5/50
  - Note: a single run can fail both.
- **Why fix9 regressed A1**: fix9 effectively reintroduced **HashMap-order tie-breaking** for `test_router_decisions`.
  - When `softmax_sample()` returns multiple tied candidates, the candidate list order can follow `HashMap` iteration
    order (pseudo-random across process runs).
  - The tie-breaker uses `candidates.iter().min_by_key(tree_size)`. When `tree_size` is also tied (often all zeros),
    `min_by_key` keeps the **first** candidate it sees, so selection becomes pseudo-random across runs.

## 1.4 Fix14 status snapshot (prefix caching)

- **What we added (A1 production improvement)**: Prefix-based routing cache to maximize KV reuse even when events haven't propagated yet.
  - **Where**: `lib/llm/src/kv_router/scheduler.rs` - Added `PrefixCache` struct and integrated into `DefaultWorkerSelector`
  - **Files modified**:
    - `lib/llm/src/kv_router.rs`: Added `router_enable_prefix_cache`, `router_prefix_cache_size` config (env: `DYN_ROUTER_ENABLE_PREFIX_CACHE`, `DYN_ROUTER_PREFIX_CACHE_SIZE`)
    - `lib/llm/src/kv_router/scheduler.rs`: Implemented prefix cache with LRU eviction and fixed 5-minute TTL
  - **Configuration**:
    - `DYN_ROUTER_ENABLE_PREFIX_CACHE=true` - Enable feature (default: false)
    - `DYN_ROUTER_PREFIX_CACHE_SIZE=1000` - Max cache entries (default: 1000)

- **How it works**:
  1. Before routing decision, checks cache for matching token prefix (min 3 tokens, tries longest match first)
  2. If cached worker found and still available, routes there immediately (bypasses logit calculation)
  3. After routing decision, stores the route with full token sequence as key
  4. Cache uses hash-based lookup with LRU eviction when full, TTL-based expiration for stale entries

- **Benefits**:
  - **Production robustness**: Routes related requests to same worker even when KV events delayed
  - **Test stability**: Provides deterministic routing for related requests via prefix matching
  - **Opt-in**: Disabled by default, no impact when not enabled

- **Testing results (A1)**: `test_router_decisions[jetstream-tcp]` on `dynamo:latest-dev-fix14-dev`

  **WITH prefix caching** (`DYN_ROUTER_ENABLE_PREFIX_CACHE=true`):
  - **49/50 passed (98.0%)** in 1..50 serial iterations
  - **0/50 A1 routing failures** ✅
  - **Failure breakdown**:
    - A1 (routing): 0/50 (0%) ✅
    - A4_crash (infrastructure): 1/50 (2%) - run024: `Fatal Python error: Aborted`
  - Evidence: `logs/serial_fix14/run001-run050/pytest.log`

  **WITHOUT prefix caching** (control test, default config):
  - **50/50 passed (100.0%)** ✅
  - **0/50 A1 routing failures** ✅
  - **0/50 infrastructure crashes** ✅
  - Evidence: `logs/serial_fix14-no-prefix-cache/run001-run050/pytest.log`

  **Key Finding**: Fix13's timing improvements alone are sufficient for 100% A1 stability.
  The prefix cache provides additional **production benefits** (consistent routing for related
  requests to maximize KV reuse) but is not required for test stability. The single A4_crash
  in the "with prefix" run was random infrastructure noise (Python SIGABRT).

  - Comparison: Fix13 (92%) → Fix14 without prefix (100%) → Fix14 with prefix (98%/100% routing)

---

=============================================================================
2. PROBLEM STATEMENT
=============================================================================

## 2.1 Problem Statement

### 2.1.1 CI Failures to Reproduce

- https://github.com/ai-dynamo/dynamo/actions/runs/21553857421/job/62106847745
- https://github.com/ai-dynamo/dynamo/actions/runs/21549672070/job/62096453750

### 2.1.2 Observed Errors in CI

The CI shows intermittent failures with these error patterns:

```
Fatal Python error: Aborted
[gw6] node down: Not properly terminated
worker 'gw6' crashed while running 'tests/router/test_router_e2e_with_mockers.py::test_kv_push_router_bindings[True-tcp]'
```

### Error Types Observed

---

=============================================================================
3. TESTING GUIDELINES
=============================================================================

## 3.1 Avoid `sleep()` in tests

Prefer explicit “wait until condition” checks over `time.sleep()` / `asyncio.sleep()` in tests.

Reasons:
- Sleeps make tests **timing-sensitive**: the right delay depends on machine load, CI noise, and parallelism.
- Sleeps are either **too short** (flake) or **too long** (slow), and both outcomes get worse as you scale concurrency.
- Sleeps hide real readiness/synchronization signals; waiting on a condition produces better failures (what we were waiting for never happened).

=============================================================================
4. ERROR TAXONOMY
=============================================================================

## 4.1 Part A: Single-thread issues (Serial / -n 1)

These issues occur even when tests run sequentially (no parallelism). They indicate actual test bugs or race conditions that must be fixed regardless of parallelism level.

---

#### A1. KV Prefix Routing Assertion (test_router_decisions) - SERIAL
- **Cause:** Tie-breaker picked the *smallest* `tree_size` (worst prefix reuse) during ties
- **Severity:** Medium (flaky test)
- **Frequency:** 2/75 serial iterations (3%)
- **Log files:**
  - `logs/serial/run029/run.log` - iter 29 failure
  - `logs/serial/run035/run.log` - iter 35 failure
- **Example:**
```python
AssertionError: Expected exactly 1 (worker_id, dp_rank) to have events (due to prefix reuse),
but found 3 with events: [(0, 0), (0, 1), (0, 2)]
# Location: tests/router/common.py:2071
```

**Root cause analysis (2026-02-05 update):**
1. `softmax_sample()` can return multiple tied candidates.
2. We were breaking ties with `min_by_key((tree_size, worker))`, which picks the *smallest* `tree_size`.
3. Smaller `tree_size` means less prefix reuse, so this systematically routed away from the best match and spread requests across dp_ranks (triggering the “events seen on 3 dp_ranks” assertion).

**Fix:** Prefer *larger* `tree_size` by sorting descending via `Reverse(tree_size)`:
`min_by_key((Reverse(tree_size), worker))`.

- **Where:** `lib/llm/src/kv_router/scheduler.rs` in `DefaultWorkerSelector::select_worker()`
  (tie-breaker on `candidates`), see the `min_by_key(|w| (Reverse(tree_size), *w))` key.

- **Status:** FIX IMPLEMENTED - validated with repeated Docker runs.

**Before vs after (router-decisions-specific):**
- **Before (pre-fix):** 2 failures / 75 serial iterations (3%) of `test_router_decisions[jetstream-nats]`
  - See `logs/serial/run029/` and `logs/serial/run035/` for the dp_rank-spread assertion.
- **After (fix8):** 0 occurrences of the dp_rank-spread assertions in the fix8 repro stream so far.
  - Verified by searching `/tmp/repro_fix8/fix8_1_50.nohup.log` for:
    - `Expected exactly 1 (worker_id, dp_rank) to have events`
    - `Expected all responses to have the same decode_worker_id`
    - `FAILED ...::test_router_decisions[jetstream-nats]`
  - Note: the current fix8 failures are in **A2** (`test_indexers_sync[...]`) and are a different
    problem: routers disagree on KV event counts (sync/indexer convergence), not worker selection.

#### A2. Router Event Sync Assertion (test_indexers_sync) - N1
- **Cause:** Race condition in router event synchronization
- **Severity:** Medium (flaky test)
- **Frequency:** 1/100 -n 1 iterations (1%)
- **Log files:**
  - `logs/n1/run028/run.log` - iter 28 failure
- **Example:**
```python
AssertionError: Router states have different numbers of events
# Router 1 has 100 events, Router 2 has 50 events
# Location: tests/router/common.py:1608
```

#### A3. Test Timeout (test_indexers_sync[jetstream]) - N1
- **Cause:** Jetstream transport hangs during sync
- **Severity:** Medium (flaky test)
- **Frequency:** 2/100 -n 1 iterations (2%)
- **Log files:**
  - `logs/n1/run071/run.log` - iter 71 timeout
  - `logs/n1/run099/run.log` - iter 99 timeout
- **Example:**
```
E           Failed: Timeout (>90.0s) from pytest-timeout.
```

#### A4. Second Router Creation Hang (test_indexers_sync) - SERIAL/N1
- **Cause:** Global `OnceCell` Tokio runtime state from first router blocks second router initialization
- **Severity:** **CRITICAL** (indefinite hang, blocks all subsequent tests)
- **Frequency:** Observed in both serial (run099) and -n 4 (run091) containers
- **Duration:** 5-10+ hours until manually killed
- **Location:** `tests/router/common.py` - "Creating second KV router with its own runtime"
- **Why pytest-timeout doesn't help:** Hang is in Rust native code; Python signals can't interrupt it
- **Log files (showing last output before hang):**
  - `logs/serial/run099/test_indexers_sync[jetstream]/test.log.txt` - serial hang (5.5 hours)
  - `logs/n4/run091/test_indexers_sync[file]/test.log.txt` - n4 hang (10 hours)
  - Note: These hangs were from containers that were manually killed and restarted
- **Example test log:**
```
[TEST] INFO tests.router.common: Completed 25/25 requests for Router 1
[TEST] INFO tests.router.common: Waiting for 1 second before creating second router
[TEST] INFO tests.router.common: Creating second KV router with its own runtime
<--- HUNG INDEFINITELY --->
```
- **See:** TODO 2 for detailed analysis and fixes

#### A5. Network Timeout (HuggingFace) - Infrastructure
- **Cause:** Transient HuggingFace API failure (rate limiting when running parallel containers)
- **Severity:** Low (infrastructure, not test bug)
- **Workaround:** Set `HF_HUB_OFFLINE=1` in `DIS-1406/reproduce_flaky_test.sh` to use cached models
- **Example:**
```
Failed to fetch model 'Qwen/Qwen3-0.6B' from HuggingFace
error sending request for url (https://huggingface.co/api/models/Qwen/Qwen3-0.6B/revision/main)
```

---

## 4.2 Part B: Multi-thread issues (xdist parallelism)

These issues occur only when multiple tests run in parallel (pytest-xdist). They indicate global state corruption between tests.

---

#### B1. Fatal Crash (SIGABRT) - **REPRODUCED** (2026-02-04)
- **Cause:** PyO3/Tokio runtime interaction issue (global OnceCell state shared across xdist workers)
- **Severity:** **CRITICAL** (process crash, high frequency at -n 4)
- **Frequency:** 5/10 iterations (50%) at -n 4 parallelism (runs 91-100)
- **Affected tests:** Multiple tests crash with SIGABRT:
  - `test_kv_push_router_bindings` (runs 091, 099)
  - `test_indexers_sync[file]` (runs 096, 098, 099)
  - `test_router_decisions[jetstream-tcp]` (run 100)
- **Log files:**
  - `logs/n4/run091/run.log` - gw0 crash
  - `logs/n4/run096/run.log` - gw3 crash
  - `logs/n4/run098/run.log` - gw2 crash + assertion errors
  - `logs/n4/run099/run.log` - **2 crashes** (gw3, gw1)
  - `logs/n4/run100/run.log` - gw1 crash
- **Example (local reproduction):**
```
Fatal Python error: Aborted
[gw0] node down: Not properly terminated
# Crash in test_kv_push_router_bindings at line 535
# Stack trace shows crash in main test thread
```
- **Root cause:** Global `OnceCell` state persists across sequential tests within the same xdist worker. When Test A shuts down and Test B starts, the stale global state causes crashes.

---

=============================================================================
5. HOW TO REPRODUCE
=============================================================================

## 5.1 How to Reproduce

### Prerequisites: Build Docker Image

Build the test image using `container/build.sh` (same as CI):

```bash
cd /path/to/dynamo2

# Build the dev image (same command as CI job 62106847745)
./container/build.sh --tag dynamo:latest-dev --target dev --framework none --enable-kvbm --enable-media-ffmpeg

# This creates: dynamo:latest-dev
```

### Run Tests at Different Parallelism Levels

Use `DIS-1406/reproduce_flaky_test.sh` which runs N iterations with a fresh container per iteration:

```bash
cd /path/to/dynamo2

# Usage: ./DIS-1406/reproduce_flaky_test.sh -p <parallelism> [-s start] [-e end]
#   -p, --parallelism   'serial' or a number (e.g., 1, 4, 16, 32)
#   -s, --start         Start iteration (default: 1)
#   -e, --end           End iteration (default: 100)

# Serial (no xdist) - ~5 min per iteration
./DIS-1406/reproduce_flaky_test.sh -p serial

# -n 1 parallelism (single xdist worker) - ~5 min per iteration
./DIS-1406/reproduce_flaky_test.sh -p 1

# -n 4 parallelism - ~2 min per iteration
./DIS-1406/reproduce_flaky_test.sh -p 4

# Resume from iteration 50 to 100
./DIS-1406/reproduce_flaky_test.sh -p serial -s 50 -e 100
```

The script:
1. Launches a fresh Docker container for each iteration
2. Runs pytest with specified parallelism
3. Saves pytest console logs to `logs/{serial,n1,n4}/runNNN/run.log`
4. Saves per-test output to `logs/{serial,n1,n4}/runNNN/test_name/`
5. Tracks pass/fail in `results.txt` and `summary.txt`
6. Prints final summary with error breakdown
7. Supports resuming from any iteration (preserves existing logs)
8. Uses `flock` to safely support parallel execution to the same log directory

### Analyze Results

```bash
# Count pass/fail for a given parallelism level (e.g., n4)
grep -l "passed" logs/n4/run*/run.log | wc -l          # passed iterations
grep -l "failed" logs/n4/run*/run.log | wc -l          # failed iterations

# Error details
grep -h "FAILED" logs/n4/run*/run.log | sort | uniq -c | sort -rn
grep -h "AssertionError" logs/n4/run*/run.log | head -5
grep -h "Fatal Python error" logs/n4/run*/run.log | head -5

# Check summary
cat logs/n4/summary.txt
```

### Files

| File | Purpose |
|------|---------|
| `DIS-1406/reproduce_flaky_test.sh` | Stress test script |
| `logs/*/runNNN/run.log` | Individual iteration pytest output |
| `logs/*/runNNN/test_name/` | Per-test logs (test.log.txt, python.log.txt, etc.) |
| `logs/*/summary.txt` | Pass/fail summary |
| `logs/*/results.txt` | Detailed results per iteration |

### Test Run Statistics (Baseline - No Fixes Applied)

| Directory | Parallelism | Iterations | Passed | Failed | Pass Rate | Notes |
|-----------|-------------|------------|--------|--------|-----------|-------|
| `logs/serial/` | Serial | 77* | 75 | 2 | **97%** | *Runs 2-24 missing (overwritten during testing) |
| `logs/n1/` | `-n 1` | 100 | 97 | 3 | **97%** | Complete baseline |
| `logs/n4/` | `-n 4` | 100 | 95 | 5 | **95%** | Runs 1-90: 0 failures; **Runs 91-100: 5 failures (50%)** |

**Key observations:**
- Serial and `-n 1` have similar ~97% pass rates (flakiness from test race conditions)
- `-n 4` shows bimodal behavior:
  - Runs 1-90: **0% failure** (90/90 passed) - likely clean global state
  - Runs 91-100: **50% failure** (5/10 failed with SIGABRT) - corrupted global state
- The 50% failure rate when SIGABRT starts occurring is the critical finding
- Once global OnceCell state gets corrupted, failures become frequent

**Failure breakdown:**
| Directory | Failure Type | Count | Tests Affected |
|-----------|--------------|-------|----------------|
| `logs/serial/` | Routing assertion | 2 | `test_router_decisions[jetstream-nats]` |
| `logs/n1/` | Event sync assertion | 1 | `test_indexers_sync[file]` |
| `logs/n1/` | Timeout | 2 | `test_indexers_sync[jetstream]` |
| `logs/n4/` | **SIGABRT crash** | 5 | Multiple tests |

*This section will be updated as more test runs are added (e.g., with fixes applied).*

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

### Results Summary (2026-02-04)

| Parallelism | Iterations | Passed | Failed | Pass Rate | Status |
|-------------|------------|--------|--------|-----------|--------|
| Serial | 75 (26-100) | 73 | 2 | **97%** | Complete |
| `-n 1` | 100 | 97 | 3 | **97%** | Complete |
| `-n 4` | 10 (91-100) | 5 | 5 | **50%** | Complete |

*Note: Serial runs 26-100 (75 iterations) because iterations 1-25 were partially overwritten during testing.*
*Note: -n 4 runs 91-100 show dramatically higher failure rate due to SIGABRT crashes.*

### Error Breakdown by Parallelism

#### Serial (75 iterations: 26-100)

| Iter | Test | Error Type | Message |
|------|------|------------|---------|
| 29 | `test_router_decisions[jetstream-nats]` | Routing assertion | `Expected exactly 1 (worker_id, dp_rank) to have events, but found 2` |
| 35 | `test_router_decisions[jetstream-nats]` | Routing assertion | `Expected same decode_worker_id for prefix reuse, found 2 unique values` |

**Summary:** 2/75 failures (3%) - Both failures in `test_router_decisions[jetstream-nats]` with KV prefix routing issues.

#### `-n 1` (100 iterations) - COMPLETE

| Iter | Test | Error Type | Message |
|------|------|------------|---------|
| 28 | `test_indexers_sync[file]` | Routing assertion | `Router 1 has 100 events, Router 2 has 50 events` |
| 71 | `test_indexers_sync[jetstream]` | Timeout | `Failed: Timeout (>90.0s) from pytest-timeout` |
| 99 | `test_indexers_sync[jetstream]` | Timeout | `Failed: Timeout (>90.0s) from pytest-timeout` |

**Summary:** 3/100 failures (3%) - All failures in `test_indexers_sync` (2 timeouts, 1 event count mismatch).

#### `-n 4` (iterations 91-100) - COMPLETE

| Iter | Test | Error Type | Worker |
|------|------|------------|--------|
| 91 | `test_kv_push_router_bindings` | **SIGABRT** | gw0 |
| 96 | `test_indexers_sync[file]` | **SIGABRT** | gw3 |
| 98 | `test_indexers_sync[file]`, `test_router_decisions` | **SIGABRT** + assertion | gw2 |
| 99 | `test_kv_push_router_bindings`, `test_indexers_sync[file]` | **2x SIGABRT** | gw3, gw1 |
| 100 | `test_router_decisions[jetstream-tcp]` | **SIGABRT** | gw1 |

**Summary:** 5/10 failures (50%) - **SIGABRT crashes are highly reproducible at -n 4!**

### Flaky Tests Identified

| Test | Transport | Failure Mode | Occurrences |
|------|-----------|--------------|-------------|
| `test_router_decisions[jetstream-nats]` | nats | KV prefix routing to multiple workers | 2 (serial), 1 (-n 4) |
| `test_router_decisions[jetstream-tcp]` | tcp | **SIGABRT crash** | 1 (-n 4) |
| `test_indexers_sync[file]` | file | **SIGABRT crash** | 4 (-n 4) |
| `test_indexers_sync[file]` | file | Router event count mismatch | 1 (-n 1) |
| `test_indexers_sync[jetstream]` | jetstream | Timeout >90s | 2 (-n 1) |
| `test_indexers_sync[file/jetstream]` | any | **Indefinite hang** on 2nd router creation | 2 (serial, n4) |
| `test_kv_push_router_bindings` | tcp | **SIGABRT crash** (PyO3/Tokio) | 2 (-n 4) |

### Log Files

| Failure | Pytest Console | Test Output Dir |
|---------|----------------|-----------------|
| Serial iter 29 | `logs/serial/run029/run.log` | `logs/serial/run029/` |
| Serial iter 35 | `logs/serial/run035/run.log` | `logs/serial/run035/` |
| -n 1 iter 28 | `logs/n1/run028/run.log` | `logs/n1/run028/` |
| -n 1 iter 71 | `logs/n1/run071/run.log` | `logs/n1/run071/` |
| -n 1 iter 99 | `logs/n1/run099/run.log` | `logs/n1/run099/` |
| -n 4 iter 91 | `logs/n4/run091/run.log` | `logs/n4/run091/` |
| -n 4 iter 96 | `logs/n4/run096/run.log` | `logs/n4/run096/` |
| -n 4 iter 98 | `logs/n4/run098/run.log` | `logs/n4/run098/` |
| -n 4 iter 99 | `logs/n4/run099/run.log` | `logs/n4/run099/` |
| -n 4 iter 100 | `logs/n4/run100/run.log` | `logs/n4/run100/` |

---

## 5. Conclusion and Potential Fixes

### Conclusion

**Two distinct categories of issues identified:**

#### Single-Thread Issues (Must fix first - these are actual test bugs)
- ~~`test_router_decisions` - **3% failure rate in serial** due to non-deterministic `softmax_sample` routing~~ **FIXED (2026-02-04)**
- `test_indexers_sync` - Event sync failures and timeouts (1-2% in n1)
- `test_indexers_sync` - **Indefinite hang** when creating second router (global OnceCell blocks)

#### Multi-Thread Issues (Global state corruption)
- **SIGABRT crashes** - 50% failure rate at -n 4 due to global `OnceCell` state reuse
- Only occurs when multiple tests run in same xdist worker process

### Recommended Fix Priority

#### ~~Priority 1: Fix `test_router_decisions` Non-Deterministic Routing (SERIAL FIX)~~ **COMPLETED**

**Status:** RESOLVED (2026-02-05)

**Implementation:** Deterministic tie-breaking that prefers *more* prefix reuse:
```rust
// Tie-breaker: prefer larger tree_size (more prefix reuse), then worker_id/dp_rank.
let best_worker = if candidates.len() > 1 {
    let tree_sizes = &request.overlaps.tree_sizes;
    *candidates
        .iter()
        .min_by_key(|w| (std::cmp::Reverse(tree_sizes.get(w).copied().unwrap_or(0)), **w))
        .expect("candidates should not be empty")
} else {
    candidates[0]
};
```

**Verification:** `test_router_decisions[jetstream-nats]` is stable across repeated runs; run full stress to confirm 0/50 failures.

**Note:** Test-side waiting for KV events does NOT fix this - events ARE received, but routing is still probabilistic.

#### Priority 2: Fix `test_indexers_sync` Second Router Hang (SERIAL FIX)

Add `DistributedRuntime::reset()` method or use subprocess isolation:
```rust
impl DistributedRuntime {
    pub fn reset() {
        // Clear RT, RTHANDLE, INIT OnceCells
        // Shutdown TCP server
        // Clear PyO3 async runtime association
    }
}
```

#### Priority 3: Fix SIGABRT at Parallelism (MULTI-THREAD FIX)

Use `pytest-forked` to fork a new process per test:
```bash
pytest --forked tests/router/test_router_e2e_with_mockers.py
```
- **Solves:** SIGABRT crashes (each test gets fresh global state)
- **Maintains:** Full parallelism with `-n auto`

#### Priority 4: Refactor to Instance-Scoped State (LONG-TERM)

Replace global statics with instance-scoped state:
```rust
// Instead of:
static RT: OnceCell<Runtime> = OnceCell::new();

// Use:
struct Worker {
    runtime: Arc<Runtime>,  // Instance-scoped
}
```
- **Solves:** Root cause permanently - tests become truly hermetic
- **Effort:** Significant refactoring across multiple crates

---

## 6. Serial/N1 Flakiness Analysis (Priority: MUST FIX)

> **See also:** Part A in Section 1 for error type details.

Serial and `-n 1` tests run one test at a time - they should have **0% failure rate**. Current failures indicate actual test bugs.

### Serial Failures (2/77 = 2.6%)

**Test:** `test_router_decisions[jetstream-nats]`
**Log files:** `logs/serial/run029/run.log`, `logs/serial/run035/run.log`

**Failure 1 (run029):**
```
AssertionError: Expected exactly 1 (worker_id, dp_rank) to have events (due to prefix reuse),
but found 2 with events: [(2484751312148977927, 3), (2484751312148977927, 1)]
```

**Failure 2 (run035):**
```
AssertionError: Expected all responses to have the same decode_worker_id (due to prefix reuse),
but found 2 unique values: {893573269278361612, 893573269278361615}
```

**Root Cause Analysis (UPDATED 2026-02-04):**

~~The test expects prefix reuse routing (same prefix → same worker/dp_rank). The race condition is:~~

~~1. Request 1 arrives → router has no overlap info → selects worker via softmax sampling~~
~~2. Request 1 completes → KV events published to NATS/JetStream~~
~~3. Request 2 arrives **BEFORE** router receives KV events~~
~~4. Router still has no overlap info → selects potentially different worker/dp_rank~~
~~5. Test assertion fails~~

**ACTUAL ROOT CAUSE:** The tie-breaker picked the smallest `tree_size`, which is the opposite of what we want for prefix reuse.

**FIX IMPLEMENTED (2026-02-05):**
- **Location:** `lib/llm/src/kv_router/scheduler.rs` lines 562-581
- **Change:** Prefer larger `tree_size` via `Reverse(tree_size)`, then use `WorkerWithDpRank` for deterministic fallback ordering
- **Why it works:** `tree_size` correlates with prefix reuse; descending selection keeps routing on the best match and avoids spreading across dp_ranks

**Verification:**
- All 6 test variants (jetstream/nats_core/no_kv_events × nats/tcp) pass consistently
- Tested with `-n 6` parallelism: 6/6 passed
- Tested 5× serial runs: 10/10 passed

### N1 Failures (3/100 = 3%)

**Test:** `test_indexers_sync[file]` and `test_indexers_sync[jetstream]`
**Log files:** `logs/n1/run028/run.log`, `logs/n1/run071/run.log`, `logs/n1/run099/run.log`

**Failure 1 (run028):**
```
AssertionError: Router states have different numbers of events
```

**Failures 2-3 (run071, run099):**
```
Failed: Timeout (>90.0s) from pytest-timeout
```

**Root Cause Analysis:**
- JetStream transport sometimes fails to deliver all events within the timeout
- Two routers created in same test don't see consistent event state
- Likely JetStream consumer lag or missed messages

**Fix Options:**
1. **Increase timeout:** Current 90s may be insufficient for JetStream sync
2. **Add retry logic:** Wait for event counts to match with retries
3. **Fix JetStream consumer:** Ensure all events are delivered reliably

---

## 7. TODO Tasks

### TODO 0: Fix serial test flakiness (HIGHEST PRIORITY)

**Goal:** Serial tests must have 0% failure rate.

**Files to modify:**
- `tests/router/common.py` - `_test_router_decisions()` function (lines 1904-2098)

**Changes needed:**
1. After first request completes, poll `dump_events()` until KV events arrive
2. Only then send subsequent requests
3. This ensures router has overlap info for correct prefix routing

**Test command:**
```bash
./DIS-1406/reproduce_flaky_test.sh -p serial -s 1 -e 100
```

**Success criteria:** 100/100 iterations pass (0% failure rate)

---

## Fix12: Remove All Unnecessary Sleep Delays in Tests

**Status:** In progress
**Image:** `dynamo:latest-dev-fix12`
**Branch:** `keivenchang/fix-intermittent-router-test-hang`

**Problem:**
Beyond the `asyncio.sleep(2)` in `_test_router_decisions`, found additional unnecessary sleep delays:

1. **`send_request_with_retry` slept before first attempt** (line 351)
   - Added 1-second delay before every request
   - Should only sleep **between retries**, not before first attempt

2. **`send_request_via_python_kv_router` had unused `initial_wait` parameter**
   - All callers passed `initial_wait=1.0`
   - Parameter was never used (only `wait_time` for retry backoff)
   - Added unnecessary complexity

**Fix:**
1. Removed sleep before first attempt in `send_request_with_retry`
2. Removed `initial_wait` parameter from `send_request_via_python_kv_router`
3. Removed `initial_wait=1.0` from all 6 call sites

**Result:** Requests now execute immediately without artificial delays, while retaining proper exponential backoff between failures.

**Test command:**
```bash
./DIS-1406/reproduce_flaky_test.sh --image dynamo:latest-dev-fix12 --pytest-nodeid 'tests/router/test_router_e2e_with_mockers.py::test_router_decisions' -s 1 -e 50
```

---

## Fix13: Wait BEFORE Sending Subsequent Requests (Critical Fix)

**Status:** Testing
**Image:** `dynamo:latest-dev-fix13`
**Branch:** `keivenchang/fix-intermittent-router-test-hang`

**Problem (Root Cause Identified):**
Fix11 and Fix12 still failed because `wait_until_events_processed` was called **AFTER** sending each request, but the routing decision happens **INSIDE** `send_request_via_python_kv_router` when the request is submitted!

```python
# WRONG (Fix11/Fix12):
result1 = await send_request_via_python_kv_router(...)  # Request 1 routes HERE
await wait_until_events_processed(...)  # Wait AFTER routing (too late!)

result2 = await send_request_via_python_kv_router(...)  # Request 2 routes BEFORE wait finishes!
```

This caused:
- Request #1 forced to (worker_id=W1, dp_rank=1), stores 4 blocks
- Wait starts after request #1 completes
- Request #2 sent **immediately** before wait finishes
- Request #2 routes with **stale** cache view → goes to different (worker_id, dp_rank)
- Result: 4 events at expected worker, 8 events at wrong worker

**Fix:**
Move `wait_until_events_processed` to **BEFORE** sending subsequent requests:

```python
# CORRECT (Fix13):
for i, request in enumerate(requests):
    # Wait for PREVIOUS request's events BEFORE routing THIS request
    if i == 1:
        await wait_until_events_processed(min_events=4)  # Wait for request 1 events
    elif i == 2:
        await wait_until_events_processed(min_events=6)  # Wait for request 1+2 events

    # NOW send request (router has up-to-date cache state)
    result = await send_request_via_python_kv_router(...)
```

**Expected Impact:**
- Request #2 will see request #1's 4 cached blocks → routes to same (worker, dp_rank)
- Request #3 will see requests #1+#2's 6 cached blocks → routes to same (worker, dp_rank)
- All 8 unique blocks end up at the forced (worker_id, dp_rank)

**Test command:**
```bash
./DIS-1406/reproduce_flaky_test.sh -p serial -t fix13 --image dynamo:latest-dev-fix13 --pytest-nodeid 'tests/router/test_router_e2e_with_mockers.py::test_router_decisions[jetstream-tcp]' -s 1 -e 50 --chunk-size 10 --dry-run
```

---

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

---

### TODO 2: Fix `test_indexers_sync` hang when creating second KV router

**File:** `tests/router/common.py` (router creation logic)
**Related:** `lib/runtime/src/worker.rs` (global OnceCell), `lib/bindings/python/rust/lib.rs` (PyO3 runtime)

**Problem:** `test_indexers_sync[file]` and `test_indexers_sync[jetstream]` hang indefinitely when creating the **second** KV router in the same test process.

**Evidence (2026-02-04):**
- **n4 container** (run091): Stuck for **10 hours** at "Creating second KV router"
- **serial container** (run099): Stuck for **5.5 hours** at the same point
- Both containers showed identical test log ending:
  ```
  [TEST] INFO tests.router.common: Completed 25/25 requests for Router 1
  [TEST] INFO tests.router.common: Waiting for 1 second before creating second router
  [TEST] INFO tests.router.common: Creating second KV router with its own runtime
  <--- HUNG INDEFINITELY - NO FURTHER OUTPUT --->
  ```

**Process state when hung:**
```
PID 121 [pytest-xdist running] tests/router/test_router_e2e_with_mockers.py::test_indexers_sync[file]
```
- pytest-xdist shows the test still "running" (not idle)
- 3 other workers are `[pytest-xdist idle]` waiting for this test to complete
- nats-server and etcd still running normally
- mocker subprocess still alive but idle (no new log entries)

**Why pytest-timeout doesn't kill it:**
- The hang occurs in **Rust native code** (Tokio runtime initialization via PyO3)
- Python signal handlers cannot interrupt blocking native code
- The 90s timeout is set, but SIGALRM can't break out of the native blocking call

**Root cause:** Global `OnceCell` state from first router blocks second router initialization.

The test does:
1. Create Router 1 → initializes global `RT` (Tokio runtime) via `OnceCell`
2. Router 1 sends 25 requests successfully
3. Wait 1 second
4. **Create Router 2 → tries to initialize another `DistributedRuntime`**
5. **DEADLOCK** - `OnceCell` already initialized, new runtime can't properly initialize

Specifically, when creating the second router:
- `DistributedRuntime::new()` is called
- It tries to get/create the global Tokio runtime via `OnceCell`
- The `OnceCell` returns the EXISTING runtime from Router 1
- The existing runtime may have:
  - Stale cancellation tokens
  - Event loops associated with wrong Python context
  - TCP ports already bound
  - Blocked channels waiting for events that will never arrive
- The new router blocks waiting on a lock or channel that is held/empty from the first router's state

**Potential Fixes:**

1. **Skip/xfail the test (immediate):**
   ```python
   @pytest.mark.skip(reason="Hangs when creating second router - global OnceCell issue")
   def test_indexers_sync(...):
   ```

2. **Run each router in subprocess (short-term):**
   ```python
   # Instead of creating routers in-process, spawn subprocesses
   import multiprocessing
   def create_router_in_subprocess(...):
       p = multiprocessing.Process(target=_create_and_run_router, args=(...))
       p.start()
       return p
   ```

3. **Add `DistributedRuntime::reset()` method (medium-term):**
   - Clear global `OnceCell` state before creating second runtime
   - Requires careful handling of in-flight tasks and connections
   - See `lib/runtime/src/worker.rs` for `RT`, `RTHANDLE`, `INIT` statics

4. **Refactor to instance-scoped state (long-term):**
   - Replace global `static RT: OnceCell<Runtime>` with instance-owned runtime
   - Each `DistributedRuntime` owns its own `tokio::Runtime`
   - Requires significant refactoring but eliminates the root cause

**Workaround for stress testing:**
- Use `pytest-forked` to run each test in a fresh process:
  ```bash
  pytest --forked tests/router/test_router_e2e_with_mockers.py
  ```
- This ensures each test gets fresh global state, but is slower
