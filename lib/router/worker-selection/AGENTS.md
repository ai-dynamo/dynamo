# lib/router/worker-selection

- This crate is the worker selection stage: the candidate rows and views, the `WorkerFilter`, `WorkerScorer`, and `WorkerPicker` plugin API, the composed `WorkerSelectionPolicy`, the candidate preparer, kv-transfer source choice, the shipped default policy, the LoRA filter, conditional disaggregation, `SelectionConfig`, and `WorkerSelectionPolicyError`. The kernel builds the rows and calls the policy; the plugin registry and the prelude plugins name live in `dynamo-kv-router`.
- Depend only on `dynamo-router-protocol` and `dynamo-slot-tracker` among router crates; the slot-tracker edge exists for the fullness rule (`RankLoad`, `BusyThresholds`), which is that stage's output. Never add `dynamo-runtime`, `dynamo-kv-router`, `dynamo-kv-indexer`, axum, reqwest, zmq, or prometheus; the gate rejects them.
- Filters and scorers are functions of (context, row): a policy must give the same answer for the same inputs. The picker's randomness is the one exception and is the subject of plan step 9e.
- In-tree hosts import this crate through `host`; plugins import `dynamo_kv_router::plugins::prelude`. Items either module names are `pub` with a doc comment, never `#[doc(hidden)] pub`.
- Run `scripts/router_layering_check.sh` before committing.
