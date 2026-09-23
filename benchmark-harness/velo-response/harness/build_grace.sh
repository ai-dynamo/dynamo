#!/usr/bin/env bash
# Controller step only; failures cannot exit the allocation keeper.
set -euo pipefail
ROOT=/lustre/fsw/coreai_comparch_trtllm/jothomson/dynamo-velo-response-20260923
SHARED=/lustre/fsw/coreai_comparch_trtllm/jothomson/dynamo-numa
REPO="$ROOT/src/dynamo"
export CARGO_HOME="$SHARED/cache/cargo-home" RUSTUP_HOME="$SHARED/cache/rustup" RUSTUP_TOOLCHAIN=1.96.1-aarch64-unknown-linux-gnu
export PROTOC="$SHARED/env/protoc-27.3/bin/protoc"
export PATH="$CARGO_HOME/bin:$(dirname "$PROTOC"):$PATH"
export CARGO_BUILD_JOBS=64 CARGO_TARGET_DIR="$ROOT/target"
export RUSTFLAGS='-C target-cpu=native -C force-frame-pointers=yes --cfg tokio_unstable'
export CARGO_PROFILE_RELEASE_DEBUG=2 PCRE2_SYS_STATIC=1
export BINDGEN_EXTRA_CLANG_ARGS="-isystem $(gcc -print-file-name=include)"
export SWAGGER_UI_DOWNLOAD_URL="file:$ROOT/harness/swagger-ui-v5.17.14.zip"
export PYO3_PYTHON=/lustre/fsw/coreai_comparch_trtllm/jothomson/dynamo-tyche-sidecar-agentx-c1024-dp1-20260910/src/dynamo/.venv/bin/python
unset RAYON_NUM_THREADS RAYON_RS_NUM_THREADS TOKENIZERS_PARALLELISM FASTOKENS_BPE_THREADS
test "$(uname -m)" = aarch64
rustc -Vv > "$ROOT/manifests/rustc.txt"
printf '%s\n' "$RUSTFLAGS" > "$REPO/.dynamo-native-rustflags"
cd "$REPO/lib/bindings/python"
cargo metadata --format-version 1 --features velo-ucx > "$ROOT/manifests/prebuild-bindings-metadata.json"
cargo build --release --locked --features tracing/release_max_level_warn,velo-ucx > "$ROOT/logs/frontend-build.log" 2>&1
cp "$CARGO_TARGET_DIR/release/lib_core.so" src/dynamo/_core.so
sha256sum src/dynamo/_core.so > "$ROOT/manifests/core.sha256"
cargo metadata --locked --format-version 1 > "$ROOT/manifests/grace-bindings-metadata.json"
"$REPO/.venv/bin/python" -c 'import dynamo._core, dynamo.frontend, dynamo.mocker; print(dynamo._core.__file__); print(dynamo.frontend.__file__)' > "$ROOT/manifests/python-imports.txt"
date -Is > "$ROOT/control/BUILD_COMPLETE"
