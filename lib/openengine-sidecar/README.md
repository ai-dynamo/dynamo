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
`7d6ac3848bba9c9a2b1aabcd4fab5f1a4ea04558`; a different HEAD fails the build.
The Cargo dependency also requires package version 0.2.0. Keep the path
dependency until that exact OpenEngine commit is published at an immutable Git
or registry location, then replace it with the corresponding immutable pin.
