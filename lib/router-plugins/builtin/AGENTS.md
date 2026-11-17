# lib/router-plugins/builtin

`dynamo-router-policies` holds the router plugins Dynamo ships: the default worker-selection policy (`default_policy`, `default_policy_with_rng`, `default_factory`, `startup_policy_parameters`), the sticky min-load, two-tier cost-function, and ThunderAgent policies, and `register`. A plugin crate depends on `dynamo-kv-router` alone and imports through `dynamo_kv_router::plugins::prelude`; a type a policy needs and the prelude lacks is added to the prelude by name.

- Hosts link this crate directly (`default_registry`, `register`).
- Design: `lib/router/README.md`; `docs/fern/pages/developer-guide/knowledge-base/modular-components/router/custom-worker-selection.mdx`.

## Performance

- **Adding or changing a shipped policy.** `required_worker_inputs` decides how much the scheduler computes for every admission, so declare the narrowest set. Gate an optional group on the setting that reads it, as the default scorer does for `CACHE`, so a config that leaves it off doesn't pay. Then run `cargo bench -p dynamo-router-policies --bench default_selection` and `--bench default_lifecycle`. The row and input rules are in `lib/router/worker-selection/AGENTS.md` (#18528).
