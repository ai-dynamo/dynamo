# lib/router-plugins/builtin

`dynamo-router-policies` holds the router plugins Dynamo ships: the default worker-selection policy (`default_policy`, `default_policy_with_rng`, `default_factory`, `startup_policy_parameters`), the sticky min-load, two-tier cost-function, and ThunderAgent policies, and `register`. A plugin crate depends on `dynamo-kv-router` alone and imports through `dynamo_kv_router::plugins::prelude`; a type a policy needs and the prelude lacks is added to the prelude by name.

- Depends on (Dynamo crates): `dynamo-kv-router`.
- Must not depend on: any other Dynamo crate in `[dependencies]`; the kernel is a dev-dependency for tests and benches only.
- Hosts link this crate directly (`default_registry`, `register`).
- Check: `bash scripts/router_layering_check.sh` (and `--host`).
- Design: `lib/router/README.md`; `docs/fern/pages/developer-guide/knowledge-base/modular-components/router/custom-worker-selection.mdx`.
