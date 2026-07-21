# Dynamo OpenEngine sidecar

This crate consumes generated `openengine-proto` 0.2.0 bindings from immutable
OpenEngine commit `f1a7189311770f8aa1f0dd787561df809847595d`.

`Cargo.toml` uses the pinned Git dependency so clean CI and release builds do
not depend on a sibling checkout. To develop against a local OpenEngine
worktree, add an uncommitted Cargo patch with an absolute path:

```toml
[patch."https://github.com/ai-dynamo/openengine.git"]
openengine-proto = { path = "/absolute/path/to/openengine/packages/rust/openengine-proto" }
```

Remove the override before generating `Cargo.lock` or publishing changes.
