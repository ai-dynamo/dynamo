# lib/router/protocol

`dynamo-router-protocol` is data: the identities and worker-facing wire vocabulary every router crate names. Read `README.md` for what lives here and where everything else went.

## Rules

- Add a type only if it is serialized across a process boundary to an engine, a worker, or a sidecar, or is an identity every crate names. Otherwise add it to the crate that produces it and let consumers depend on that crate.
- Keep `[dependencies]` exactly `serde`, `serde_json`, `thiserror`. Put test-only crates under `[dev-dependencies]`. Never add a `dynamo-*` dependency or any other router crate.
- Do not write `dyn`, `async`, `std::fs`, `anyhow`, `tracing`, or `derive_builder` in `src/`, and do not add `#[doc(hidden)] pub` items. Behavior beyond `serde` derives and conversions belongs to a consumer crate. If a line must carry one of these tokens as prose, end it with `// layering: allow`; the gate prints every exempt line for the reviewer.
- Do not give `AttemptId` a public constructor, a `From<u64>`, or a hidden escape. `AttemptIdMinter::mint` yields fresh ids only; the slot tracker owns the one instance. The chosen-value constructors stay behind the `testing` and `bench` features; never enable either from a normal dependency, and do not add new features to this crate.
- Do not re-export other crates: no `pub use other_crate::...` and no `pub use ...::*`.
- Run `scripts/router_layering_check.sh` before committing. CI runs the same script with `--host`.
- Keep this file and its `CLAUDE.md` twin (exactly `@AGENTS.md`) together.
