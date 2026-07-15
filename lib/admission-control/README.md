# dynamo-admission-control

`dynamo-admission-control` contains Dynamo's built-in policy-class queue admission strategies. The KV router owns request storage, lifecycle delivery, and final worker selection; this crate supplies the strategy-specific admission decisions.

The currently implemented strategy is [Session-Aware Admission Control](src/session_aware/), configured as `queue_admission.type: session_aware`. Its policy is inspired by [ThunderAgent](https://arxiv.org/abs/2602.13692), adapted to Dynamo's native queue-admission lifecycle.

Session-Aware Admission Control retains quiescent completed sessions for `session_retention_seconds` (1,800 seconds by default) and then forgets their placement and accounting state. The inactivity lease resets after every successful turn, so clients do not need to emit a terminal signal. A later request for an expired session is admitted as a new program.

## Known follow-ups

- A resumed request released with `WorkerPlacement::Exact(worker)` can fail if that worker disappears before final selection. A future fix should retry normal selection for worker removal without weakening sticky placement during temporary overload.
- A paused program whose assigned worker leaves the structurally eligible set cannot be greedily placed elsewhere until its resume timeout. A future fix should clear stale assignments before resume while preserving assignments during temporary overload.
