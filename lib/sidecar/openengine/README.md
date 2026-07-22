# Dynamo OpenEngine sidecar

This crate consumes generated `openengine-proto` 0.2.0 bindings from immutable
OpenEngine commit `df3a9be24a2a36a4ff7a6d4fef9f1d7480ae210d`.

`Cargo.toml` uses the pinned Git dependency so clean CI and release builds do
not depend on a sibling checkout. To develop against a local OpenEngine
worktree, add an uncommitted Cargo patch with an absolute path:

```toml
[patch."https://github.com/ai-dynamo/openengine.git"]
openengine-proto = { path = "/absolute/path/to/openengine/packages/rust/openengine-proto" }
```

Remove the override before generating `Cargo.lock` or publishing changes.
