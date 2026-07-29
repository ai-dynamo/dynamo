# Online Replay Runtime Contract

Online replay owns a fresh Tokio runtime for each invocation. Failure and
cancellation guarantee propagation, not restoration or settlement of replay
state.

## Internal failures

- Return the first observed `Err` immediately.
- Let task panics surface as `JoinError`; do not catch or translate them inside
  online replay orchestration.
- Produce no partial report after an internal failure.
- Dropping the replay-owned runtime is the cleanup mechanism for outstanding
  work. Do not add per-request settlement, router restoration, panic hooks,
  `catch_unwind`, or generalized failure-shutdown machinery.
- Propagate router bookkeeping failures with `?`; do not warn and continue with
  potentially invalid routing state.
- Do not use the external cancellation token as internal-error cleanup.
- Do not change shared `LiveEngine` lifecycle behavior to make replay failure
  recovery more elaborate.

## Explicit cancellation

The injected `CancellationToken` is an external control seam. It must stop
outer arrival and workload waits, prevent new dispatch, terminate outstanding
replay work, and return `online replay cancelled`. It does not promise settled
requests or restored router state.

## Successful shutdown

Only successful completion has an ordered settlement contract:

1. Request and workload tasks reach terminal completion.
2. Engines shut down, closing admission producers.
3. The admission forwarder drains.
4. The replay router shuts down.
5. The recorder finalizes and returns the complete report.

Preserve this order. `InFlightGuard` remains part of the normal request
lifecycle and is not failure-recovery machinery.
