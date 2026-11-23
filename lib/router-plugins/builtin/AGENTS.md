# lib/router-plugins/builtin

`dynamo-router-policies` holds the router plugins Dynamo ships: the default worker-selection policy (`default_policy`, `default_policy_with_rng`, `default_factory`, `startup_policy_parameters`), the sticky min-load, two-tier cost-function, and ThunderAgent policies, the output-length prediction request classifier, and `register`. A plugin crate depends on `dynamo-kv-router` alone and imports through `dynamo_kv_router::plugins::prelude`; a type a policy needs and the prelude lacks is added to the prelude by name.

- Hosts link this crate directly (`default_registry`, `register`).
- Design: `lib/router/README.md`; `docs/fern/pages/developer-guide/knowledge-base/modular-components/router/custom-worker-selection.mdx`.

## Rules

- **Adding a shipped policy, classifier, or hint policy.** Give it a module with a `register` function and call that from `register` in `src/lib.rs`. The type name `default` is reserved (`ReservedDefault`), and a name that is already registered fails (`Duplicate`). Add a row to `README.md` and to `docs/fern/pages/developer-guide/knowledge-base/modular-components/router/configuration-and-tuning.md`.
- **Importing a router type.** Import it from `dynamo_kv_router::plugins::prelude` only. The other router crates appear only under `[dev-dependencies]`, for tests and benches.
- **Adding parameters.** Deserialize them into a struct with `#[serde(deny_unknown_fields)]`, so a mistyped key fails at startup (`rejects_an_unknown_parameter_key`). A routing mode takes none (`routing_modes_reject_parameters`).
- **Crediting shared-cache blocks in a scorer.** Credit only what the engine will load: `SharedCacheHits::loadable_beyond`, not every hit (#18176).
- **Writing a request classifier.** `classify` is synchronous and its worker view is valid only for that call, so copy out what you keep. To hold a queued request, return a `DispatchGate`; that contract is experimental, and `tests/thunderagent_gate.rs` shows it in use (#18443, #18940).
- **Testing.** Registration cases go in `tests/plugin_registry.rs`; a policy's own behavior goes in its own file under `tests/`.

## Test and bench

```bash
cargo test -p dynamo-router-policies --lib
cargo test -p dynamo-router-policies --test default_policy   # or another file under tests/
cargo bench -p dynamo-router-policies --bench default_selection
cargo bench -p dynamo-router-policies --bench default_lifecycle
cargo bench -p dynamo-router-policies --bench thunderagent_ledger
```

## Performance

PR numbers point to worked examples; open one only when you need to see how a rule was applied.

- **Adding or changing a shipped policy.** `required_worker_inputs` decides how much the scheduler computes for every admission, so declare the narrowest set. Gate an optional group on the setting that reads it, as the default scorer does for `CACHE`, so a config that leaves it off doesn't pay. For the default policy, run `cargo bench -p dynamo-router-policies --bench default_selection` and `--bench default_lifecycle`. Those benches build only the default policy, so for any other shipped policy add a case for it or measure it another way, and say how in the PR. The row and input rules are in `lib/router/worker-selection/AGENTS.md` (#18528).
