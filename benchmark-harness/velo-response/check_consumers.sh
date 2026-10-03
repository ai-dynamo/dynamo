#!/usr/bin/env bash
set -euo pipefail
ROOT=/lustre/fsw/coreai_comparch_trtllm/jothomson/dynamo-velo-response-20260923
SHARED=/lustre/fsw/coreai_comparch_trtllm/jothomson/dynamo-numa
export CARGO_HOME="$SHARED/cache/cargo-home" RUSTUP_HOME="$SHARED/cache/rustup"
export RUSTUP_TOOLCHAIN=1.96.1-aarch64-unknown-linux-gnu
export PROTOC="$SHARED/env/protoc-27.3/bin/protoc"
export PATH="$CARGO_HOME/bin:$(dirname "$PROTOC"):$PATH"
export CARGO_TARGET_DIR="$ROOT/target-runtime" CARGO_BUILD_JOBS=64 PCRE2_SYS_STATIC=1
export BINDGEN_EXTRA_CLANG_ARGS="-isystem $(gcc -print-file-name=include)"
VALIDATION="$ROOT/src/dynamo-validation"
if [[ ! -d "$VALIDATION" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git -c core.hooksPath=/dev/null -C "$ROOT/src/dynamo" worktree add --detach "$VALIDATION" HEAD
fi
cd "$VALIDATION"
cargo metadata --format-version 1 --features dynamo-runtime/velo-ucx > "$ROOT/manifests/workspace-metadata.json"
cargo check -p kvbm-config -p kvbm-engine -p kvbm-physical --no-default-features
cargo metadata --manifest-path lib/bindings/kvbm/Cargo.toml --format-version 1 > "$ROOT/manifests/kvbm-metadata.json"
UCX_TLS=tcp cargo test -p dynamo-runtime --features velo-ucx --lib pipeline::network -- --test-threads=4
export PYTHONPATH="$ROOT/python-test-deps:$ROOT/src/dynamo/lib/bindings/python/src:$VALIDATION/components/src"
/lustre/fsw/coreai_comparch_trtllm/jothomson/dynamo-tyche-sidecar-agentx-c1024-dp1-20260910/src/dynamo/.venv/bin/python -m pytest -q -o addopts= -o filterwarnings=default components/src/dynamo/common/tests/configuration/test_runtime_args.py components/src/dynamo/common/tests/configuration/test_kv_router_args.py components/src/dynamo/mocker/tests/unit/test_config.py -k response_plane
