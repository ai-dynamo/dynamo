# Sidecar common

Shared infrastructure for Rust sidecars:

- gRPC transport arguments and defaults
- plaintext endpoint validation
- connection pooling and startup retries
- gRPC-to-Dynamo error mapping

Engine protocols and request conversion stay in each sidecar crate.
