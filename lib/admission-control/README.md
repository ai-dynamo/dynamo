# dynamo-admission-control

`dynamo-admission-control` contains Dynamo's built-in policy-class admission policies. The KV router owns request storage, lifecycle delivery, and final worker selection; this crate supplies policy-specific admission decisions.

The currently implemented policy is [Session-Aware Admission Control](src/session_aware/), configured as `admission: {type: session_aware}`. Policy-family and cache-bucket classification, including exact-placement reclassification, remain existing KV-router behavior outside this concrete policy.

To apply `session_aware` by default, configure it on the default policy family's only cache bucket. To opt in selected traffic instead, configure an explicit class by omitting `policy_family` and `cache_bucket`, then send `x-dynamo-meta-policy-class: <class-name>` on every HTTP request using that policy. Startup rejects Session-Aware attachment to one class in a multi-bucket family because cache-state changes could split a session across class-local program tables; cross-bucket ownership is intentionally deferred until that model is agreed.

Session-Aware Admission Control immediately forgets a session's placement and accounting state when a request carries Dynamo's session-final signal, while forwarding that request normally. When a client cannot emit the signal, quiescent completed sessions are retained for `session_retention_seconds` (1,800 seconds by default) and then forgotten. The inactivity lease resets after every successful turn. A later request for a released or expired session is admitted as a new program.

## Inspiration and comparison

Session-Aware Admission Control is inspired by the [ThunderAgent paper](https://arxiv.org/abs/2602.13692) and adapted to Dynamo's native policy-class admission lifecycle.

## Known follow-ups

- A resumed request released with `WorkerPlacement::Exact(worker)` can fail if that worker disappears before final selection. A future fix should retry normal selection for worker removal without weakening sticky placement during temporary overload.
