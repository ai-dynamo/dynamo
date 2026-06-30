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
