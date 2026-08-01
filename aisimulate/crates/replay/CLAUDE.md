# Offline Replay Liveness

Offline replay is a discrete-event simulation. Real runtime scheduling must
not be mistaken for virtual-time progress or quiescence.

## Balanced liveness contract

- **Tight-spin/livelock extreme:** do not repeatedly report an effect-free,
  queued-only zero-duration pass as progress at the same virtual timestamp.
  Preserve the `made_progress` filtering introduced by PR #10919. Do not emit
  an unconditional immediate completion, poll/spin, sleep, or synthesize time.
- **Dead-end/lost-wakeup extreme:** an empty effects/event queue is not proof of
  quiescence while workers own unfinished requests. Every unfinished request
  must have a concrete future wakeup: a scheduled worker completion, modeled
  deadline, or dependency notification.
- Stop same-time iteration when no observable state changed, but only after the
  owning subsystem can account for how unfinished work will wake.

For now, preserve the current replay behavior and the balanced checks above.
Changes to async settlement belong to DEP #11018; do not approximate them here
with replay-level timing tricks or new hard assertions.

## Eager forward-pass execution

Starting an epoch commits one non-preemptive batch. Here, non-preemptive means
later arrivals or commands cannot interrupt that batch or change its outcome;
they may affect only queued or post-pass state, or be deferred.

Offline replay executes the committed batch eagerly at epoch start, finalizing
its participating `EngineCore` state changes and scheduled completion payload
before virtual time reaches the completion timestamp. Visibility is split:

- Admission observations and `PassStart` events are visible at epoch start.
- Request outputs, lifecycle events, FPM publications, and `PassEnd` events
  wait for the shared completion boundary.
- All ranks in an attention-DP group, including ranks with no work in the
  current epoch, share the slowest-rank completion boundary.
- Do not generalize this guarantee to the whole `EngineComponent` or to other
  replay components. KVBM transport, disaggregated handoffs, worker startup,
  planner actions, and router queues may have independent deadlines or
  explicitly revocable events.
- Preserve eager pass execution unless the modeled engine gains genuinely
  preemptive behavior. Do not add engine-core snapshotting, transactional
  rollback, or speculative-state recovery without a demonstrated semantic
  need.
