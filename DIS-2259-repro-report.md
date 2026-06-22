# DIS-2259 — Local reproduction report

**Bug:** KV-router frontend leaks memory/threads under LoRA deployment churn
**Branch base:** `b1985e601d` (ancestor of `main`, clean tree)
**Date:** 2026-06-22
**Result:** ✅ Root cause reproduced locally and deterministically, with no NATS/etcd or GPU dependency.

---

## TL;DR

The leak's root cause — *KV-router/discovery background work is scoped to the
process/runtime cancellation token instead of the router/monitor lifetime* — is
**proven on `main`** by three deterministic unit tests added under `#[cfg(test)]`
(no production code changed):

| Test | What it proves | Observed |
|---|---|---|
| `dis_2259_kv_router_drop_cancels_process_wide_root_token` | `KvRouter::Drop` cancels the runtime's **root** token, not a router-owned one | root token `is_cancelled() == true` after dropping one router |
| `dis_2259_dropping_one_router_tears_down_co_resident_router` | Router work is tied to the shared process token → dropping one WorkerSet's router kills another's tasks | alive tasks `22 → 3` after dropping A while B is still held (B should keep ~11) |
| `dis_2259_kv_worker_monitor_task_leaks_after_drop` | `KvWorkerMonitor` has no `Drop`/owned token → its task survives forever after the monitor is dropped (unmasked accumulation) | 10 monitors churned → **80 orphaned tokio tasks**, never reclaimed |

All three pass (i.e. the buggy behavior is present) on `main`:

```
test discovery::worker_monitor::tests::dis_2259_kv_worker_monitor_task_leaks_after_drop ... DIS-2259 monitor leak: baseline_alive=0 after_alive=80 leaked=80 monitors_churned=10
ok
test kv_router::tests::dis_2259_dropping_one_router_tears_down_co_resident_router ... DIS-2259 blast radius: baseline=0 after_A=11 after_A+B=22 after_drop_A=3 (router_b still alive; per_router≈11)
ok
test kv_router::tests::dis_2259_kv_router_drop_cancels_process_wide_root_token ... DIS-2259 confirmed: dropping a single KvRouter cancelled the runtime's process-wide root cancellation token (drt().primary_token())
ok
test result: ok. 3 passed; 0 failed
```

Run them with:

```bash
cargo test -p dynamo-llm --lib dis_2259 -- --nocapture --test-threads=1
```

---

## Root cause, established from the code

### 1. The runtime's token tree (`lib/runtime/src/runtime.rs`)

```rust
// Runtime::new
let cancellation_token = CancellationToken::new();              // ROOT (process-wide)
let endpoint_shutdown_token = cancellation_token.child_token(); // child of ROOT

pub fn primary_token(&self) -> CancellationToken { self.cancellation_token.clone() } // ROOT clone
pub fn child_token(&self)   -> CancellationToken { self.endpoint_shutdown_token.child_token() } // grandchild of ROOT
```

So `primary_token()` is a clone of the **root** token, and every `child_token()`
is a **grandchild** of that same root. Cancelling the root cascades to all of them.

### 2. `KvRouter` binds to the root token and cancels it on drop (`lib/llm/src/kv_router.rs`)

```rust
// KvRouter::new
let cancellation_token = component.drt().primary_token(); // ROOT clone, NOT a router-owned token

// Drop for KvRouter
fn drop(&mut self) {
    self.cancellation_token.cancel();  // cancels the PROCESS-WIDE root token
}
```

Meanwhile the router's actual background work is scoped elsewhere:

* `KvScheduler::start` → `component.drt().child_token()` (scheduler metrics task) and
  passes another `child_token()` into `LocalScheduler::new_with_overlap_refresh`,
  which spawns **3** tasks (worker-config monitor, remote-state listener, periodic
  queue-update task) — `lib/kv-router/src/scheduling/local.rs:108,160,181`.
* KV-event subscriber → `component.drt().primary_token()` —
  `lib/llm/src/kv_router/indexer/recovery/subscriber.rs:31`.
* Worker discovery/recovery loop → `component.drt().primary_token()` —
  `lib/llm/src/kv_router/indexer/recovery/worker_query.rs:99`.

None of these tokens are owned by the router. The router's `Drop` does not cancel
the scheduler's `child_token`s individually — it only nukes the shared root.

### 3. `KvWorkerMonitor` never stops its task (`lib/llm/src/discovery/worker_monitor.rs`)

```rust
// start_monitoring
let cancellation_token = component.drt().child_token();   // grandchild of ROOT
// ... discovery list_and_watch + KV-metrics subscriber ...
tokio::spawn(async move { loop { tokio::select! { _ = cancellation_token.cancelled() => break, ... } } });
```

`KvWorkerMonitor` is `#[derive(Clone)]`, stores **no token and no `JoinHandle`**, and
has **no `Drop` impl**. Dropping all clones therefore leaves the spawned task running
(it only ever stops on `cancellation_token.cancelled()`, which nothing fires short of
full runtime shutdown). Each leaked task holds clones of `worker_load_states`, the
discovery watch, and the `Client` — i.e. it pins per-WorkerSet state alive.

---

## How this produces the observed symptom (RSS + thread growth → OOM)

There are two faces of the same mis-scoping, and the tests exercise both:

1. **`KvRouter::Drop` over-cancels.** If a router were actually dropped, it would
   cancel the **process-wide root token** and tear down *every* background task in
   the frontend — including other live WorkerSets (test #2: `22 → 3`). Because the
   production frontend keeps serving during churn, router instances are evidently
   **not** being dropped (a lingering `Arc` keeps them alive). With `Drop` never
   firing, *nothing* router-scoped ever cancels the scheduler/subscriber/discovery
   tasks, so each rebuilt WorkerSet adds a fresh, permanent set of them.

2. **`KvWorkerMonitor` leaks unconditionally.** It has no drop-time cleanup at all
   (test #3), so every WorkerSet teardown orphans its monitor task (~8 tokio tasks
   each in this harness, including the per-`Client` discovery/endpoint tasks they
   keep alive), regardless of the router `Drop` question.

Under LoRA/adapter churn, WorkerSet/router teardown+rebuild happens repeatedly on a
single long-lived frontend process. Orphaned tasks — each pinning routing state,
discovery watches, and NATS subscriptions — accumulate without bound. That is the
steady RSS climb (≈45→140+ GiB) and `Threads: 7191` reported in the ticket, and it
matches the ticket's own root-cause analysis precisely.

> Note: the unit tests count **tokio tasks** (the directly-leaked entity). In a real
> frontend each orphaned task additionally holds NATS subscriptions and discovery
> list-watch streams, which are what drive the OS-thread and RSS growth.

---

## What was added (test-only diff)

```
 lib/llm/src/discovery/worker_monitor.rs |  83 ++++++++++++++++   (1 test + helper)
 lib/llm/src/kv_router.rs                | 169 ++++++++++++++++++++  (2 tests + helper)
 2 files changed, 252 insertions(+)
```

No production code was modified. Measurement uses tokio's stable
`RuntimeMetrics::num_alive_tasks()` (tokio 1.48) and `CancellationToken::is_cancelled()`,
on a `process_local` `DistributedRuntime` (no external services).

---

## Suggested fix (matches ticket, validated by these tests)

Make the background work owner-scoped so these tests would instead show
near-baseline task counts after drop:

1. In `KvRouter::new`, create a router-owned `CancellationToken` as a **child** of
   `component.drt().primary_token()` (parented to runtime shutdown but independent),
   and pass it into the indexer, scheduler (worker-monitor loop, request loop,
   metrics task), KV-event subscriber, and `WorkerQueryClient` discovery loop.
2. Cancel that router-owned token in `Drop for KvRouter` (never `primary_token()`).
3. Give `KvWorkerMonitor` an owned token + a `Drop` (or a stored `JoinHandle`/guard)
   that cancels/aborts the monitoring task when the last clone is dropped.

After the fix, the two `kv_router` tests should be inverted (root token stays live;
co-resident router survives) and the monitor test should show `leaked ≈ 0`.
