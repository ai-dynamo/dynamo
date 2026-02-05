# Rust Development Guide for Dynamo

This guide covers setting up the Rust toolchain and building, testing, and developing the Dynamo runtime.

## Prerequisites

### System Dependencies

**Ubuntu:**

```bash
sudo apt install -y \
  build-essential \
  libhwloc-dev \
  libudev-dev \
  pkg-config \
  libclang-dev \
  protobuf-compiler \
  python3-dev \
  cmake
```

**macOS:**

```bash
brew install cmake protobuf
```

### Rust Toolchain

The project pins Rust **1.90.0** via `rust-toolchain.toml`. Installing via `rustup` will automatically pick up the correct version.

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

After installation, verify by running from the repo root:

```bash
rustc --version   # should print 1.90.0
cargo --version
```

`rustup` reads `rust-toolchain.toml` automatically, so no manual version selection is needed.

### Optional: cargo-deny (license/dependency checks)

CI uses `cargo-deny` for license compliance. To run locally:

```bash
cargo install cargo-deny
```

## Workspace Structure

The workspace is defined in the root `Cargo.toml` with Rust edition **2024** and resolver **3**.

### Default Members

Running `cargo build`, `cargo check`, or `cargo test` without `-p` or `--workspace` operates on these crates:

| Crate | Path | Description |
|-------|------|-------------|
| `dynamo-llm` | `lib/llm` | LLM engine, routing, discovery, preprocessing |
| `dynamo-runtime` | `lib/runtime` | Distributed runtime, pipelines, service mesh |
| `dynamo-config` | `lib/config` | Configuration utilities |
| `dynamo-tokens` | `lib/tokens` | Tokenization |
| `dynamo-async-openai` | `lib/async-openai` | Async OpenAI client |
| `dynamo-parsers` | `lib/parsers` | Parser utilities |
| `libdynamo_llm` | `lib/bindings/c` | C API bindings |

### Non-default Members

These are excluded from defaults because they are slow to build or require extra native dependencies:

| Crate | Path | Description |
|-------|------|-------------|
| `dynamo-run` | `launch/dynamo-run` | CLI application |
| `dynamo-kv-router` | `lib/kv-router` | KV cache router |
| `dynamo-memory` | `lib/memory` | GPU memory management |
| `dynamo-engine-mistralrs` | `lib/engines/mistralrs` | Mistral.rs engine |
| `dynamo-codegen` | `lib/bindings/python/codegen` | Python binding codegen |

To build or test everything:

```bash
cargo check --workspace
cargo test --workspace
```

## Building

### Quick Build (default members)

```bash
cargo build
```

### Full Workspace Build

```bash
cargo build --workspace
```

### Release Build

```bash
cargo build --release
```

The release profile uses `codegen-units = 1` and `lto = "thin"` for optimized output.

### Building Specific Crates

```bash
cargo build -p dynamo-runtime
cargo build -p dynamo-llm
cargo build -p dynamo-kv-router
```

### Feature Flags

Several crates expose feature flags for optional functionality:

**dynamo-llm:**

| Feature | Description |
|---------|-------------|
| `cuda` | Enable CUDA support |
| `block-manager` | KV cache block manager (requires CUDA + NIXL) |
| `integration` | Enable integration tests |
| `testing-cuda` | Test helpers requiring CUDA |
| `testing-nixl` | Test helpers requiring NIXL |
| `testing-etcd` | Tests requiring a live etcd instance |
| `media-nixl` | Media processing with NIXL transfers |
| `media-ffmpeg` | Media processing with FFmpeg |

**dynamo-runtime:**

| Feature | Description |
|---------|-------------|
| `integration` | Enable integration tests |
| `testing-etcd` | Tests requiring a live etcd instance |
| `tokio-console` | Enable tokio-console debugging |
| `tcp-low-latency` | Low-latency TCP optimizations |

**dynamo-kv-router:**

| Feature | Description |
|---------|-------------|
| `metrics` | Prometheus metrics via dynamo-runtime |
| `bench` | Enable benchmark binaries |

Example:

```bash
cargo build -p dynamo-llm --features cuda
cargo test -p dynamo-llm --features integration
```

## Checking and Linting

### Type Check (fast, no codegen)

```bash
cargo check                # default members
cargo check --workspace    # everything
```

### Formatting

```bash
cargo fmt             # auto-format
cargo fmt -- --check  # check only (CI mode)
```

### Clippy

```bash
cargo clippy --no-deps --all-targets -- -D warnings
```

### License Compliance

```bash
cargo-deny -L error check --hide-inclusion-graph licenses bans
```

## Testing

### Run All Tests (default members)

```bash
cargo test --locked --all-targets
```

### Run All Tests (full workspace)

```bash
cargo test --locked --workspace --all-targets
```

### Doc Tests

```bash
cargo doc --no-deps && cargo test --locked --doc
```

### Run Tests for a Specific Crate

```bash
cargo test -p dynamo-runtime
cargo test -p dynamo-llm
cargo test -p dynamo-kv-router
```

### Run a Specific Test

```bash
cargo test -p dynamo-llm -- test_name_substring
```

### Integration Tests

Integration tests are gated behind feature flags and may require external services (etcd, CUDA GPUs):

```bash
# Runtime integration tests
cargo test -p dynamo-runtime --features integration

# LLM integration tests
cargo test -p dynamo-llm --features integration

# Tests requiring a live etcd
cargo test -p dynamo-llm --features testing-etcd
```

### Key Test Files

| File | Coverage |
|------|----------|
| `lib/llm/tests/http-service.rs` | HTTP/OpenAI API service |
| `lib/llm/tests/kserve_service.rs` | gRPC/KServe service |
| `lib/llm/tests/kv_manager.rs` | KV cache management |
| `lib/llm/tests/preprocessor.rs` | Prompt preprocessing and templates |
| `lib/llm/tests/backend.rs` | Backend sequence factory |
| `lib/runtime/tests/pipeline.rs` | Pipeline and worker pools |
| `lib/runtime/tests/lifecycle.rs` | Component lifecycle |
| `lib/runtime/tests/soak.rs` | Stress / soak tests |

Test data (model configs, mock responses, media files) lives in `lib/llm/tests/data/`.

## Benchmarks

Benchmarks use the [Criterion](https://docs.rs/criterion) framework:

```bash
# Tokenizer performance
cargo bench -p dynamo-llm --bench tokenizer

# Runtime compute pool overhead
cargo bench -p dynamo-runtime --bench compute_pool_overhead

# TCP codec performance
cargo bench -p dynamo-runtime --bench tcp_codec_perf

# KV router radix tree (requires bench feature)
cargo bench -p dynamo-kv-router --bench radix_tree_microbench --features bench
```

## Build Configuration Notes

### Tokio Unstable

The project enables `tokio_unstable` via `.cargo/config.toml`:

```toml
[build]
rustflags = ["--cfg", "tokio_unstable"]
```

This is required for tokio-console support and is set globally for the workspace.

### Build Scripts

Three crates have `build.rs` files that run at compile time:

- **`lib/llm/build.rs`** — Compiles `kserve.proto` via `tonic-build`. Requires `protoc` on `PATH`.
- **`lib/bindings/c/build.rs`** — Generates C headers via `cbindgen`.
- **`launch/dynamo-run/build.rs`** — Detects CUDA/Metal availability and embeds git metadata.

### CUDA Builds

For GPU-enabled builds, ensure the CUDA toolkit is installed and `nvcc` is on your `PATH`, then:

```bash
cargo build -p dynamo-run --features cuda
cargo build -p dynamo-llm --features cuda
```

## Troubleshooting

### `protoc` not found

```
error: Failed to compile proto files: Could not find `protoc`
```

Install the protobuf compiler:

```bash
# Ubuntu
sudo apt install -y protobuf-compiler

# macOS
brew install protobuf
```

### `libclang` not found

```
error: Unable to find libclang
```

Install the clang development libraries:

```bash
# Ubuntu
sudo apt install -y libclang-dev

# macOS (usually included with Xcode CLI tools)
xcode-select --install
```

### NIXL header not found

```
warning: nixl.h: No such file or directory
```

This is expected when building without NVIDIA NIXL installed. The build falls back to stub APIs. This only affects `lib/memory` and related GPU transfer functionality.

### Wrong Rust version

```
error: package `some-crate` requires rustc 1.90.0
```

Ensure `rustup` is managing your toolchain and run from the repo root so `rust-toolchain.toml` is detected:

```bash
rustup show   # should show 1.90.0 as active
```

## CI Workflow Summary

The pre-merge CI (`pre-merge.yml`) runs these Rust checks on every PR that touches Rust files:

1. `cargo fmt -- --check`
2. `cargo-deny check licenses bans`
3. `cargo test --locked --no-run` (compile tests)
4. `cargo doc --no-deps && cargo test --locked --doc`
5. `cargo test --locked --all-targets`
6. `cargo clippy --no-deps --all-targets -- -D warnings`
