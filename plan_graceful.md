# Graceful Shutdown Plan

## Problem

Runtime graceful shutdown currently drains inflight work too late in two places:

- Endpoint discovery can stay published until the runtime tears down the main lease, so routers can still select a worker that has already entered drain.
- `PushEndpoint` stops the NATS service endpoint as soon as endpoint shutdown starts, before inflight streaming requests finish. Active streams can receive `ControlMessage::Stop` and surface as cancelled or 500 responses.

During rollout restart or pod termination, a worker should remove itself from routing quickly while still finishing work it has already accepted.

For the KV router specifically, the worker liveness signal is discovery. The KV router can select a draining decode worker while its discovery entry is still present. Dispatch then goes through runtime `direct(...)`; depending on timing, the request either reaches a worker that is no longer accepting work, fails during response-stream setup, or fails earlier with `instance_id not found` after the runtime client has already observed removal. Temporary local quarantine is not enough because reconciliation can restore the worker while discovery still says it exists.

## Desired Contract

1. Endpoint shutdown starts.
2. The endpoint is unpublished from discovery immediately, so routers stop choosing it for new work.
3. The request-plane endpoint stops accepting new work but remains alive for already accepted requests.
4. Inflight requests drain.
5. The NATS endpoint is stopped after drain completes.
6. Runtime shutdown continues to Phase 3 and tears down shared transports such as NATS and etcd.
7. A global graceful shutdown budget bounds the whole process. After the budget expires, shutdown proceeds even if some work is still inflight.

Default global budget: 840 seconds.

The key invariant is bounded graceful shutdown: remove the worker from routing immediately, drain accepted work for at most the configured budget, then allow shutdown to continue even if some requests are still stuck.

In Kubernetes, the external hard cap is the Pod's `terminationGracePeriodSeconds`. The `preStop` hook and the process shutdown after TERM share that same budget; `preStop` is not a separate unlimited phase. Set the Dynamo runtime budget lower than the Pod grace period so there is time left for cleanup and process exit.

## Review Notes

- Discovery removal is the routing gate. The NATS service endpoint may remain alive until drain completes, but once discovery is removed, healthy routers should stop choosing the worker. Stale routers may still race and fail; that is acceptable as long as the window is bounded.
- Do not use local quarantine as the primary drain signal. `report_instance_down()` is temporary and can be undone by reconciliation while discovery still contains the worker.
- Keep graceful-tracker ownership tied to request-plane drain, not discovery cleanup. Discovery should be removed immediately on endpoint shutdown, but the tracker should only be released after the request-plane endpoint has drained or the global budget expires.
- The cleanup path must be idempotent. It should be safe for the shutdown-token cleanup and the final endpoint-exit cleanup to both attempt discovery unregister.
- Prefer a runtime-scoped timeout name, for example `DYN_RUNTIME_GRACEFUL_SHUTDOWN_TIMEOUT_SECS`, unless this is intentionally aligned with the existing worker/system-server timeout. The current default should be 840 seconds for production.
- If Kubernetes `preStop` is used, keep it as a quick drain trigger rather than the whole drain wait. A long `preStop` delays TERM delivery and consumes the same Pod grace budget the process needs for shutdown.

## Implementation Shape

### Unpublish discovery at drain start

In endpoint startup, after discovery registration succeeds, spawn a cleanup task tied to the endpoint shutdown token.

When the endpoint shutdown token is cancelled:

- unregister the `DiscoveryInstance::Endpoint` through the discovery abstraction;
- log warning on failure;
- do not wait for request-plane endpoint exit before removing discovery.

Keep a final best-effort discovery unregister after endpoint task exit. This handles non-shutdown exits and makes cleanup idempotent.

Avoid direct `etcd_client.kv_delete(...)` in upstream code unless no abstraction exists for the target branch. Prefer the existing discovery API so the behavior works for etcd, Kubernetes, file, and memory discovery backends.

This should be the only signal the KV router needs for graceful drain. Once discovery removal is observed, the worker leaves the runtime client's routable set and the KV router should stop selecting it for new decode or prefill requests.

Keep the existing request-plane cleanup ordering separate:

- discovery unregister happens immediately after endpoint shutdown token cancellation;
- request-plane `server.unregister_endpoint(...)` may wait for drain;
- graceful tracker unregister happens after request-plane unregister completes.

### Drain before stopping PushEndpoint

Change `PushEndpoint` shutdown ordering:

- On cancellation, break the accept loop and remember that this is a shutdown-triggered exit.
- If `graceful_shutdown` is true, wait for inflight requests to reach zero.
- Only after the inflight wait completes, call `endpoint.stop().await`.
- If `graceful_shutdown` is false, keep the current immediate stop behavior.

This prevents active streaming contexts from receiving service stop while they are still healthy.

Stopping `endpoint.next()` polling is not a full cluster-wide accept gate. Discovery removal is what prevents normal routing to this endpoint; delayed `endpoint.stop()` protects streams that are already active.

### Add a global runtime shutdown budget

Wrap Runtime Phase 2 in a timeout:

- `Runtime::shutdown()` cancels the endpoint shutdown token.
- Phase 2 waits for the graceful shutdown tracker.
- If the global timeout expires, log the active count and proceed to Phase 3 by cancelling the main token.

Expose the timeout through configuration or environment. Suggested default:

```text
DYN_RUNTIME_GRACEFUL_SHUTDOWN_TIMEOUT_SECS=840
```

The existing per-endpoint inflight timeout can remain, but the runtime-wide cap is the safety net that prevents one stuck stream, leaked guard, or broken endpoint from blocking termination indefinitely.

The global budget should apply to the whole graceful shutdown window, not to each endpoint independently. A deployment termination should have one upper bound, so multiple endpoints cannot multiply the effective drain time.

For testability, factor the shutdown coordinator into a helper that accepts the endpoint token, main token, tracker, and timeout. `Runtime::shutdown()` can spawn that helper, while unit tests call it directly with short durations.

Suggested Kubernetes alignment:

```yaml
terminationGracePeriodSeconds: 900
env:
  - name: DYN_RUNTIME_GRACEFUL_SHUTDOWN_TIMEOUT_SECS
    value: "840"
```

That leaves roughly one minute for final cleanup, transport teardown, and process exit after the runtime stops waiting gracefully.

## Test Strategy

Prefer fast deterministic tests first. A full NATS/etcd rollout test is useful later, but the core regressions can be covered without external services.

### Unit seams

Add small helper functions and test them directly:

- `wait_for_endpoint_shutdown_and_cleanup(token, cleanup)` for the discovery cleanup trigger;
- `finish_shutdown_after_inflight_drain(inflight, notify, timeout, stop_service)` for the PushEndpoint stop ordering;
- `run_shutdown_phases(endpoint_token, main_token, tracker, timeout)` for bounded Runtime Phase 2.

These tests should use milliseconds, not seconds, and avoid real NATS or etcd.

### Discovery/client behavior

Use `MockDiscovery` or in-memory KV discovery to prove routing state updates from discovery removal:

- register an endpoint instance;
- create or observe a runtime `Client` for that endpoint;
- verify the instance is routable;
- unregister the instance through the discovery abstraction;
- wait for the watcher to observe removal;
- verify `instance_ids_avail()` no longer includes the worker.

This test covers the signal the KV router depends on without needing the full KV router stack.

### Integration coverage

One higher-level test should cover the shutdown ordering if a reliable harness exists:

- register/start an endpoint with a handler that blocks one request;
- trigger runtime shutdown;
- assert discovery removal is visible before the blocked handler is released;
- assert request-plane stop/tracker completion happens only after release, or after the global timeout.

Keep this as a runtime integration test rather than an e2e deployment test unless CI already provides NATS/etcd.

## Regression Tests

### Discovery cleanup waits for shutdown

Unit-test the helper that waits on the endpoint shutdown token:

- create a token and cleanup callback;
- wait briefly and assert cleanup has not run;
- cancel the token;
- assert cleanup runs immediately after cancellation.

### Discovery cleanup uses abstraction

Use a mock or in-memory discovery backend:

- register an endpoint;
- verify it is listed;
- cancel the endpoint shutdown token;
- verify the endpoint disappears before the request-plane task finishes draining.

### KV router stops selecting draining workers

Use a runtime-client level test as the primary regression:

- register two workers and confirm both are routable;
- start shutdown for one worker and keep its request-plane task artificially draining;
- verify the worker is removed from discovery/routable IDs before drain completes;
- verify a new KV-routed request selects only the remaining worker, or returns no-workers/backpressure if no replacement exists.

A full KV-router test can be added later if there is already a lightweight fixture for `KvPushRouter`; otherwise the runtime-client test is the correct low-cost coverage because discovery removal is the KV router's liveness input.

### PushEndpoint stop waits for inflight drain

Unit-test a helper around the drain-then-stop ordering:

- set inflight to 1;
- start shutdown helper with a stop callback that records when it runs;
- assert stop has not run while inflight is nonzero;
- decrement inflight and notify;
- assert stop runs only after inflight reaches zero.

### Non-graceful remains immediate

Add or preserve a test that `graceful_shutdown = false` calls service stop without waiting for inflight drain.

### Global timeout proceeds to Phase 3

Unit-test the runtime shutdown coordination:

- register a graceful task or endpoint that never completes;
- call shutdown with a short global timeout;
- assert the main token is cancelled after timeout.
- assert shutdown does not wait once per endpoint beyond the global budget.

## Upstreaming Strategy

Prefer small PRs:

1. Drain inflight requests before stopping `PushEndpoint`.
2. Unpublish endpoint discovery on endpoint shutdown start.
3. Add the runtime-wide graceful shutdown timeout.

This keeps review focused and makes backports easier.
