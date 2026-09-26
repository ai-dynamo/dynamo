#!/usr/bin/env bash
set -euo pipefail
ROOT=/lustre/fsw/coreai_comparch_trtllm/jothomson/dynamo-velo-response-20260923
SHARED=/lustre/fsw/coreai_comparch_trtllm/jothomson/dynamo-numa
export CARGO_HOME="$SHARED/cache/cargo-home" RUSTUP_HOME="$SHARED/cache/rustup"
export RUSTUP_TOOLCHAIN=1.96.1-aarch64-unknown-linux-gnu
export PROTOC="$SHARED/env/protoc-27.3/bin/protoc"
export PATH="$CARGO_HOME/bin:$(dirname "$PROTOC"):$PATH"
export CARGO_TARGET_DIR="$ROOT/target-runtime" CARGO_BUILD_JOBS=64
export BINDGEN_EXTRA_CLANG_ARGS="-isystem $(gcc -print-file-name=include)"
export PCRE2_SYS_STATIC=1
cd "$ROOT/src/dynamo"
cargo test -p dynamo-runtime --lib "$@"
