# dynamo-admission-control

`dynamo-admission-control` contains Dynamo's built-in policy-class queue admission strategies. The KV router owns request storage, lifecycle delivery, and final worker selection; this crate supplies the strategy-specific admission decisions.

The currently implemented strategy is [ThunderAgent](src/thunderagent/), configured as `queue_admission.type: session_aware`.
