#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build the campaign's Dynamo for the vLLM 0.24.0 image. Never touches the workstation worktree's
# .venv; every artifact lives under LR_LIVE_ROOT keyed by the commit.
#
# Usage: build_env.sh wheel OUT_DIR   (allocated node; host or container, see below)
#        build_env.sh venv  OUT_DIR   (inside the vLLM container)
#
# wheel: release build of lib/bindings/python (ai-dynamo-runtime; default features = custom-policy,
#        which links the builtin router-plugin catalog incl. learned-choice and sticky-session; no
#        ais-forward-pass, so the live router cannot reach AISimulate). The GPU hosts and the image
#        are both Ubuntu 22.04 / glibc 2.35, and the wheel is abi3 (cp310), so building on the node
#        host with its python3.10 and libclang is equivalent; verify_env.py then imports the
#        binding inside the image, which proves the linkage. Cargo target is node-local.
# venv:  --system-site-packages over the image's python3 (keeps vLLM 0.24.0 and torch), plus the
#        wheel, ai-dynamo's runtime dependencies (minus aisimulate and transformers), and ai-dynamo
#        editable from LR_SRC (the commit + live engine shims applied by stage_source.sh); then
#        verify_env.py.
# Artifacts are reused while their manifest names the same source manifest; a rebuild writes a
# temporary directory and renames it into place. Concurrent jobs at one commit build once: the
# first takes <artifact>.building (mkdir is atomic on the shared filesystem) and the others wait for its manifest.
# The lock directory is never removed; a stale one (builder died) makes waiters fail after
# LR_BUILD_WAIT_S with its owner named.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ $# -eq 2 ]] || die "usage: build_env.sh wheel|venv OUT_DIR"
mode=$1 out=$2
lr_require_commit
mkdir -p "$out"
source_manifest="$LR_SRC.source-manifest.json"
[[ -s "$source_manifest" ]] || die "missing $source_manifest; run stage_source.sh first"
source_sha="$(sha256sum "$source_manifest" | awk '{print $1}')"
python_bin="$(command -v python3)"
source "$LR_TOOLCHAIN_ENV"
export UV_CACHE_DIR="$LR_CACHE_ROOT/uv"
mkdir -p "$UV_CACHE_DIR"

manifest_matches() {
  [[ -s "$1" ]] && "$python_bin" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])).get("source_manifest_sha256") != sys.argv[2])' "$1" "$source_sha"
}

# Returns 0 when this job must build TARGET, 1 when a concurrent job's build of it appeared.
claim_build() {
  local target=$1 manifest=$2 lock="$1.building" waited=0
  mkdir -p "$(dirname "$target")"
  until mkdir "$lock" 2> /dev/null; do
    if manifest_matches "$manifest"; then
      log "reusing $target built by $(cat "$lock/owner" 2> /dev/null || echo 'another job')"
      return 1
    fi
    (( waited < ${LR_BUILD_WAIT_S:-2400} )) \
      || die "no manifest at $manifest after ${waited}s; $lock is held by $(cat "$lock/owner" 2> /dev/null)"
    # A builder that is no longer a live Slurm job left a stale lock: fail now, not at the timeout.
    local owner_job
    owner_job="$(awk '{print $2}' "$lock/owner" 2> /dev/null || true)"
    if [[ "$owner_job" =~ ^[0-9]+$ ]] && command -v squeue > /dev/null \
      && [[ -z "$(squeue -h -j "$owner_job" -o %T 2> /dev/null)" ]]; then
      die "stale $lock: owner job $owner_job is gone and $manifest is missing; rename the lock to rebuild"
    fi
    sleep 15
    waited=$((waited + 15))
  done
  echo "job ${SLURM_JOB_ID:-none} on $(hostname) at $(date -u +%FT%TZ)" > "$lock/owner"
}

build_wheel() {
  if manifest_matches "$LR_WHEEL_DIR/build-manifest.json"; then
    log "reusing wheel in $LR_WHEEL_DIR"
    return
  fi
  claim_build "$LR_WHEEL_DIR" "$LR_WHEEL_DIR/build-manifest.json" || return 0
  [[ ! -e "$LR_WHEEL_DIR" ]] || die "$LR_WHEEL_DIR exists with a stale manifest; set a new LR_WHEEL_DIR"
  export CARGO_TARGET_DIR="$LR_NODE_ROOT/cargo-target"
  export CARGO_BUILD_JOBS="${LR_CARGO_JOBS:-96}"
  local buildenv="$LR_NODE_ROOT/buildenv-$(uname -n)"
  uv venv --quiet --python "$python_bin" "$buildenv"
  uv pip install --quiet --python "$buildenv/bin/python" 'maturin[patchelf]==1.15.0'
  # Some -sys crates can fall back to CMake; give them one if the host has none.
  if ! command -v cmake > /dev/null; then
    uv pip install --quiet --python "$buildenv/bin/python" 'cmake>=3.28,<4'
    export PATH="$buildenv/bin:$PATH"
  fi
  # nixl-sys runs bindgen, which needs a loadable libclang. Prefer the system one; otherwise use
  # the PyPI libclang wheel inside the throwaway build venv (no root, no apt).
  if ! ldconfig -p 2>/dev/null | grep -qE 'libclang(-[0-9]+)?\.so'; then
    uv pip install --quiet --python "$buildenv/bin/python" 'libclang==18.1.1'
    LIBCLANG_PATH="$("$buildenv/bin/python" -c 'import clang, os; print(os.path.join(os.path.dirname(clang.__file__), "native"))')"
    export LIBCLANG_PATH
    # The PyPI libclang ships no clang resource headers (stdbool.h, stddef.h); use the host
    # compiler's builtin include directory instead.
    local cc_include
    cc_include="$(cc -print-file-name=include)"
    [[ -f "$cc_include/stdbool.h" ]] || die "no stdbool.h in $cc_include for bindgen"
    export BINDGEN_EXTRA_CLANG_ARGS="-I$cc_include"
  fi
  local tmp started wheel
  mkdir -p "$(dirname "$LR_WHEEL_DIR")"
  tmp="$(mktemp -d "$LR_WHEEL_DIR.tmp.XXXXXX")"
  started=$(date +%s)
  (cd "$LR_SRC/lib/bindings/python" && "$buildenv/bin/maturin" build --release \
    --interpreter "$python_bin" --out "$tmp") > "$out/wheel-build.log" 2>&1 \
    || die "wheel build failed; see $out/wheel-build.log"
  wheel="$(ls "$tmp"/ai_dynamo_runtime-*.whl)"
  "$python_bin" - "$tmp" "$wheel" "$source_manifest" "$source_sha" "$(($(date +%s) - started))" <<'PY'
import hashlib, json, platform, re, subprocess, sys, zipfile
tmp, wheel, source_manifest, source_sha, seconds = sys.argv[1:6]
archive = zipfile.ZipFile(wheel)
core = next(n for n in archive.namelist() if n.endswith("_core.abi3.so"))
data = archive.read(core)
versions = sorted({tuple(map(int, m.group(1).split(b"."))) for m in re.finditer(rb"GLIBC_(\d+(?:\.\d+)+)", data)})
json.dump({
    "source_manifest": source_manifest,
    "source_manifest_sha256": source_sha,
    "commit": json.load(open(source_manifest))["commit"],
    "wheel": wheel.rsplit("/", 1)[1],
    "wheel_sha256": hashlib.sha256(open(wheel, "rb").read()).hexdigest(),
    "core_so_sha256": hashlib.sha256(data).hexdigest(),
    "features": ["default (custom-policy)"],
    "profile": "release",
    "rustc": subprocess.run(["rustc", "--version"], capture_output=True, text=True).stdout.strip(),
    "build_host": platform.node(),
    "build_python": platform.python_version(),
    "build_glibc": platform.libc_ver()[1],
    "max_required_glibc": ".".join(map(str, versions[-1])) if versions else None,
    "seconds": int(seconds),
}, open(f"{tmp}/build-manifest.json", "w"), indent=1, sort_keys=True)
PY
  mv -T "$tmp" "$LR_WHEEL_DIR"
  log "built wheel into $LR_WHEEL_DIR"
}

build_venv() {
  manifest_matches "$LR_WHEEL_DIR/build-manifest.json" || die "no wheel for this source in $LR_WHEEL_DIR"
  if manifest_matches "$LR_VENV/venv-manifest.json"; then
    log "reusing venv $LR_VENV"
  elif ! claim_build "$LR_VENV" "$LR_VENV/venv-manifest.json"; then
    :
  else
    [[ ! -e "$LR_VENV" ]] || die "$LR_VENV exists with a stale manifest; set a new LR_VENV"
    local wheel
    wheel="$(ls "$LR_WHEEL_DIR"/ai_dynamo_runtime-*.whl)"
    mkdir -p "$(dirname "$LR_VENV")"
    uv venv --quiet --system-site-packages --python "$python_bin" "$LR_VENV"
    uv pip install --quiet --python "$LR_VENV/bin/python" "$wheel" \
      'aiohttp>=3.14.3,<4.0' 'kubernetes>=32.0.1,<33.0.0' 'prometheus_client>=0.23.1,<1.0' \
      'msgspec>=0.19.0' 'redis>=6.2.0,<9.0.0' 'zstandard>=0.23.0,<1.0' 'pyzmq>=26.0.0' \
      'typing_extensions>=4.10.0' 'tqdm>=4.0.0'
    uv pip install --quiet --python "$LR_VENV/bin/python" --no-deps -e "$LR_SRC"
    "$python_bin" - "$LR_VENV/venv-manifest.json" "$source_sha" "$LR_WHEEL_DIR/build-manifest.json" <<'PY'
import json, sys
json.dump({"source_manifest_sha256": sys.argv[2], "wheel_manifest": json.load(open(sys.argv[3]))},
          open(sys.argv[1], "w"), indent=1, sort_keys=True)
PY
  fi
  uv pip freeze --python "$LR_VENV/bin/python" > "$out/venv-freeze.txt"
  "$LR_VENV/bin/python" "$LR_DEPLOY_DIR/verify_env.py" --venv "$LR_VENV" --src "$LR_SRC" \
    --wheel-manifest "$LR_WHEEL_DIR/build-manifest.json" --vllm-version "$LR_VLLM_VERSION" \
    --out "$out/env-verify.json"
}

case "$mode" in
  wheel) build_wheel; cp "$LR_WHEEL_DIR/build-manifest.json" "$out/wheel-build-manifest.json" ;;
  venv) build_venv ;;
  *) die "unknown mode $mode" ;;
esac
