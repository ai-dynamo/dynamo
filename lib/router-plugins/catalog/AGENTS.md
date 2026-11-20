# lib/router-plugins/catalog

`dynamo-worker-selection-policy-catalog` is the catalog slot a custom image replaces: its `register` is intentionally empty so `default` always selects Dynamo's built-in worker selector, and the policies Dynamo ships register separately from `dynamo-router-policies`. A plugin crate depends on `dynamo-kv-router` alone and imports through `dynamo_kv_router::plugins::prelude`.

- Hosts link this crate directly.
- Design: `lib/router/README.md`; `docs/fern/pages/developer-guide/knowledge-base/modular-components/router/custom-worker-selection.mdx`.
