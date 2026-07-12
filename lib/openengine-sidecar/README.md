# Dynamo OpenEngine sidecar

This crate is developed against the generated `openengine-proto` 0.2.0 crate
from the sibling OpenEngine worktree. The local checkout layout is fixed:

```text
sidecar/
├── dynamo-trtllm-sidecar/
└── openengine-trtllm/
    └── packages/rust/openengine-proto/
```

`build.rs` verifies that the sibling OpenEngine worktree is exactly commit
`cea19cb06acf03c911b84d5c147e519b60dd92a6`; a different HEAD fails the build.
The Cargo dependency also requires package version 0.2.0. Keep the path
dependency until that exact OpenEngine commit is published at an immutable Git
or registry location, then replace it with the corresponding immutable pin.
