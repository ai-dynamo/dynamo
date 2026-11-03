# lib/router-plugins/catalog

`dynamo-worker-selection-policy-catalog` is the catalog slot a custom image replaces: its `register` is intentionally empty so `default` always selects Dynamo's built-in worker selector, and the policies Dynamo ships register separately from `dynamo-custom-policy-builtin`. A plugin crate depends on `dynamo-kv-router` alone and imports through `dynamo_kv_router::plugins::prelude`.

- Depends on (Dynamo crates): `dynamo-kv-router`.
- Must not depend on: any other Dynamo crate.
- Hosts link this crate directly.
- Check: `bash scripts/router_layering_check.sh` (and `--host`).
- Design: `lib/router/README.md`; `docs/fern/pages/developer-guide/knowledge-base/modular-components/router/custom-worker-selection.mdx`.
