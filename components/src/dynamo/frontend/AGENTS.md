# Frontend Configuration Boundary

`dynamo.frontend` parses CLI arguments and environment defaults into
`FrontendConfig`. After that point, treat `FrontendConfig` as the source of
truth for frontend-owned settings.

- Do not write parsed `FrontendConfig` values back to `os.environ` for Rust to
  re-read. That creates a cyclic Python -> env -> Rust contract where CLI
  overrides can diverge from Rust behavior.
- Pass frontend-owned values through explicit PyO3 binding parameters or
  config structs.
- Env fallbacks may remain for standalone/direct Rust entrypoints, but an
  explicit Python binding value must win once the frontend has parsed config.
- If a setting must cross Python serialization, PyO3, RPC, or process
  boundaries, add an explicit transport path at that boundary rather than
  relying on shared process environment state.
