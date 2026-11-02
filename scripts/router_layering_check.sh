#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Router crate layering gate. Checks, from `cargo tree` and `cargo metadata`,
# that every router crate under lib/router has exactly the dependency edges
# the split design allows, and that the protocol crate carries no executable
# behavior.
#
# Usage: scripts/router_layering_check.sh [--host]
#
#   (no flag)  per-crate checks from TABLE below (exact router-dependency set,
#              forbidden transport crates, exact direct dependencies, the
#              protocol behavior grep), the composition crate's no-features
#              tree and direct dependencies, the workspace feature check, and
#              the source rules (no #[doc(hidden)], no cross-crate glob
#              re-export, README/AGENTS/CLAUDE per crate); pre-commit runs this
#              in about ten seconds.
#   --host     additionally resolve the host packages in HOSTS and
#              HOSTS_EXTERNAL and fail if any table crate reaches a heavy
#              external crate in the host's own graph (feature unification
#              included); CI runs this.
#
# Exit status: 0 all checks passed, 1 at least one layering check failed,
# 2 the gate could not run (missing crate, empty cargo output, bad usage).
# The gate fails closed: an empty or failed `cargo tree` is an error, never a
# pass. Runs from any directory inside the repository.
set -Eeuo pipefail
export LC_ALL=C

# ---------------------------------------------------------------------------
# Table. One row per router crate; add a crate by adding one line. Fields are
# separated by '|' (surrounding spaces ignored); '-' means "none" or "skip".
#
#   1 crate         cargo package name
#   2 router deps   the EXACT set of ROUTER_FAMILY crates that may appear
#                   anywhere in its normal dependency tree with default
#                   features (compared by set equality, not upper bound)
#   3 forbidden     external crates that must not appear anywhere in its
#                   normal dependency tree with --all-features
#   4 direct deps   the EXACT set of direct [dependencies] from Cargo.toml
#                   (normal and optional, not dev or build), from
#                   `cargo metadata`; '-' skips the check
#   5 behavior dir  source directory grepped for executable behavior with
#                   BEHAVIOR_RE (a line carrying ALLOW_MARK is exempt and
#                   printed); '-' skips the check
# ---------------------------------------------------------------------------
TABLE=(
  "dynamo-multimodal | - | dynamo-runtime dynamo-kv-router dynamo-router-protocol dynamo-kv-events dynamo-kv-hashing dynamo-kv-hints dynamo-kv-indexer dynamo-scheduler dynamo-tokens tokio anyhow tracing async-trait derive_builder thiserror prometheus axum hyper reqwest zmq | serde xxhash-rust | -"
  "dynamo-router-protocol | - | dynamo-runtime dynamo-kv-router dynamo-kv-hashing dynamo-multimodal dynamo-tokens tokio anyhow tracing async-trait derive_builder prometheus axum hyper reqwest zmq | serde serde_json thiserror | lib/router/protocol/src"
  "dynamo-kv-events | dynamo-router-protocol dynamo-kv-hashing dynamo-multimodal | dynamo-runtime dynamo-kv-router tokio anyhow async-trait prometheus axum hyper reqwest zmq | dynamo-kv-hashing dynamo-multimodal dynamo-router-protocol rmp-serde rustc-hash serde serde_json thiserror tracing | -"
  "dynamo-kv-hints | dynamo-router-protocol dynamo-kv-events dynamo-kv-hashing dynamo-kv-indexer dynamo-multimodal | dynamo-runtime dynamo-kv-router dynamo-slot-tracker dynamo-session-affinity dynamo-worker-selection dynamo-admission dynamo-scheduler prometheus axum hyper reqwest zmq | dynamo-kv-hashing dynamo-kv-indexer dynamo-router-protocol rustc-hash serde serde_json | -"
  "dynamo-zmq | - | dynamo-runtime dynamo-kv-router dynamo-router-protocol dynamo-kv-events dynamo-kv-hashing dynamo-multimodal prometheus axum hyper reqwest | anyhow tokio tracing zmq | -"
  "dynamo-kv-indexer | dynamo-router-protocol dynamo-kv-events dynamo-kv-hashing dynamo-multimodal | dynamo-runtime dynamo-kv-router dynamo-kv-hints dynamo-zmq axum hyper reqwest zmq | anyhow arc-swap async-trait blake3 dashmap dynamo-kv-events dynamo-kv-hashing dynamo-router-protocol dynamo-tokens fastrand flume parking_lot prometheus rustc-hash seqlock serde serde_json slotmap thiserror tokio tokio-util tracing xxhash-rust | -"
  "dynamo-slot-tracker | dynamo-router-protocol | dynamo-runtime dynamo-kv-router dynamo-kv-indexer dynamo-kv-events dynamo-kv-hashing dynamo-multimodal dynamo-zmq axum hyper reqwest zmq prometheus | anyhow crossbeam-utils dashmap dynamo-router-protocol dynamo-tokens fastrand indexmap parking_lot rustc-hash seqlock serde slotmap thiserror tokio tokio-util tracing | -"
  "dynamo-session-affinity | dynamo-router-protocol | dynamo-runtime dynamo-kv-router dynamo-kv-indexer dynamo-slot-tracker dynamo-kv-events dynamo-kv-hashing dynamo-multimodal dynamo-zmq axum hyper reqwest zmq prometheus | blake3 dashmap dynamo-router-protocol serde thiserror tokio tokio-util tracing | -"
  "dynamo-worker-selection | dynamo-router-protocol dynamo-slot-tracker | dynamo-runtime dynamo-kv-router dynamo-kv-indexer dynamo-session-affinity dynamo-kv-events dynamo-kv-hashing dynamo-multimodal dynamo-zmq axum hyper reqwest zmq prometheus | async-trait crossbeam-queue dynamo-router-protocol dynamo-slot-tracker fastrand parking_lot rustc-hash serde serde_json thiserror tracing | -"
  "dynamo-admission | dynamo-router-protocol dynamo-slot-tracker | dynamo-runtime dynamo-kv-router dynamo-kv-indexer dynamo-session-affinity dynamo-worker-selection dynamo-kv-events dynamo-kv-hashing dynamo-multimodal dynamo-zmq axum hyper reqwest zmq prometheus | async-trait dynamo-router-protocol dynamo-slot-tracker dynamo-tokens futures-util ordered-float parking_lot rustc-hash serde serde_json thiserror tokio tokio-util tracing | -"
  "dynamo-scheduler | dynamo-router-protocol dynamo-kv-hashing dynamo-kv-events dynamo-kv-hints dynamo-kv-indexer dynamo-slot-tracker dynamo-session-affinity dynamo-worker-selection dynamo-admission dynamo-multimodal | dynamo-runtime dynamo-kv-router dynamo-zmq axum hyper reqwest zmq prometheus | anyhow arc-swap async-trait blake3 crossbeam-queue dashmap derive_builder dynamo-admission dynamo-kv-events dynamo-kv-hashing dynamo-kv-hints dynamo-kv-indexer dynamo-multimodal dynamo-router-protocol dynamo-session-affinity dynamo-slot-tracker dynamo-tokens dynamo-worker-selection fastrand futures-util http once_cell ordered-float parking_lot rmp-serde rustc-hash serde serde_json thiserror tokio tokio-util tracing uuid zeroize | -"
  "dynamo-error | - | dynamo-runtime dynamo-kv-router dynamo-router-protocol dynamo-kv-events dynamo-kv-hashing dynamo-multimodal dynamo-tokens tokio anyhow thiserror tracing async-trait derive_builder prometheus axum hyper reqwest zmq | serde serde_json | -"
  "dynamo-kv-hashing | dynamo-multimodal | dynamo-runtime dynamo-kv-router dynamo-router-protocol tokio anyhow tracing async-trait prometheus axum hyper reqwest zmq | derive_builder dynamo-multimodal dynamo-tokens serde serde_json thiserror xxhash-rust | -"
)

# Every crate the split produces (SPLIT_V2 3a-final). Column 2 counts any of
# these found in a tree as a router dependency, so a crate that reaches for the
# monolith or a sibling fails before that sibling has a row of its own.
ROUTER_FAMILY="dynamo-multimodal dynamo-error dynamo-kv-hashing dynamo-router-protocol dynamo-kv-events dynamo-kv-hints dynamo-zmq dynamo-kv-indexer dynamo-slot-tracker dynamo-session-affinity dynamo-worker-selection dynamo-admission dynamo-scheduler dynamo-metrics dynamo-kv-router"

# Features no normal (non-dev) dependency may enable on a table crate, checked
# on the workspace graph (`cargo tree --workspace -e normal,features`).
ROUTER_TEST_FEATURES="testing bench profile"

# --host: packages whose resolved graph must not route a table crate to one of
# HOST_FORBIDDEN. Hosts missing from the workspace are skipped with a notice.
HOSTS="dynamo-llm dynamo-ext-proc dynamo-mocker"
# Hosts outside the workspace: "package|manifest|features" (features may be empty). The Python
# bindings run the standalone services, so they resolve with the service features on.
HOSTS_EXTERNAL="dynamo-py3|lib/bindings/python/Cargo.toml|kv-indexer,slot-tracker,select-service"
# prometheus joins this list in step 9c, when exposition moves to dynamo-metrics; today the
# kv-indexer `export` feature (kv-router `metrics`) reaches it by design.
HOST_FORBIDDEN="axum hyper reqwest zmq"
# The ZMQ socket helper is the one router crate allowed to reach zmq.
HOST_ADAPTER_EXEMPT="dynamo-zmq:zmq"

# Executable behavior the protocol crate must not contain (SPLIT_V2 G9).
BEHAVIOR_RE='\bdyn |\basync |std::fs|anyhow|tracing|derive_builder|#\[doc\(hidden\)\][[:space:]]*pub'
ALLOW_MARK='// layering: allow'

# ---------------------------------------------------------------------------
usage() { echo "usage: $0 [--host]" >&2; exit 2; }
host_mode=0
for arg in "$@"; do
  case "$arg" in
    --host) host_mode=1 ;;
    *) usage ;;
  esac
done

cd "$(git rev-parse --show-toplevel)"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
trap 'rc=$?; echo "router layering: ERROR exit $rc at line $LINENO" >&2; exit $rc' ERR

checks=0
failures=0
ok()  { checks=$((checks + 1)); echo "ok    $*"; }
bad() { checks=$((checks + 1)); failures=$((failures + 1)); echo "FAIL  $*"; }
die() { echo "ERROR $*" >&2; exit 2; }

trim()  { sed 's/^ *//; s/ *$//' <<<"$1"; }
# Space-separated words -> sorted unique lines, dropping '-' and empties.
words() { tr ' ' '\n' <<<"$1" | sed '/^$/d; /^-$/d' | sort -u; }
# stdin lines -> sorted unique lines, dropping empties.
lines() { sed '/^$/d' | sort -u; }
# stdin lines -> "{a b c}" for messages.
show()  { local s; s="$(tr '\n' ' ' | sed 's/ $//')"; echo "{${s}}"; }

# tree PKG EXPECT_FIRST [cargo tree args...]
# Prints "name version" lines of the normal dependency tree rooted at PKG.
# Dies unless cargo succeeds, prints something, and the first line is
# EXPECT_FIRST (the root for a forward tree, the focus for an inverted one).
tree() {
  local pkg="$1" expect="$2"; shift 2
  local out first
  if ! out="$(cargo tree --locked -p "$pkg" -e normal --prefix none "$@" 2>"$tmp/tree.err")"; then
    die "cargo tree -p $pkg $*: $(tr '\n' ' ' <"$tmp/tree.err")"
  fi
  [ -n "$out" ] || die "cargo tree -p $pkg $* printed nothing"
  first="$(head -n1 <<<"$out" | awk '{print $1}')"
  [ "$first" = "$expect" ] || die "cargo tree -p $pkg $*: first line is '$first', expected '$expect'"
  awk '{print $1" "$2}' <<<"$out" | sort -u
}

# Workspace manifest data, read once.
if ! cargo metadata --no-deps --locked --format-version 1 >"$tmp/meta.json" 2>"$tmp/meta.err"; then
  die "cargo metadata failed: $(tr '\n' ' ' <"$tmp/meta.err")"
fi
[ -s "$tmp/meta.json" ] || die "cargo metadata printed nothing"

python3 -I -c 'import json, sys; print("\n".join(sorted(p["name"] for p in json.load(sys.stdin)["packages"])))' \
  <"$tmp/meta.json" >"$tmp/packages.txt"
[ -s "$tmp/packages.txt" ] || die "cargo metadata lists no workspace packages"
in_workspace() { grep -qx -- "$1" "$tmp/packages.txt"; }

# direct_deps PKG -> one direct normal (incl. optional) dependency name per line
direct_deps() {
  python3 -I -c '
import json, sys
name = sys.argv[1]
pkgs = [p for p in json.load(sys.stdin)["packages"] if p["name"] == name]
if len(pkgs) != 1:
    sys.exit(f"{len(pkgs)} workspace packages named {name}, expected 1")
print("\n".join(sorted({d["name"] for d in pkgs[0]["dependencies"] if d["kind"] is None})))
' "$1" <"$tmp/meta.json"
}

table_crates=""

# ---------------------------------------------------------------------------
# Per-crate checks
# ---------------------------------------------------------------------------
for row in "${TABLE[@]}"; do
  IFS='|' read -r crate allowed forbidden direct behavior_dir <<<"$row"
  crate="$(trim "$crate")"
  allowed="$(trim "$allowed")"
  forbidden="$(trim "$forbidden")"
  direct="$(trim "$direct")"; : "${direct:=-}"
  behavior_dir="$(trim "$behavior_dir")"; : "${behavior_dir:=-}"
  table_crates="$table_crates $crate"

  in_workspace "$crate" || die "$crate is not a workspace package (root Cargo.toml members)"

  # Column 2: router dependencies, default features, exact set.
  tree "$crate" "$crate" >"$tmp/tree.txt"
  awk -v c="$crate" '$1 != c { print $1 }' "$tmp/tree.txt" | lines >"$tmp/names.txt"
  comm -12 "$tmp/names.txt" <(words "$ROUTER_FAMILY") >"$tmp/found.txt"
  words "$allowed" >"$tmp/want.txt"
  if cmp -s "$tmp/found.txt" "$tmp/want.txt"; then
    ok "$crate: router deps == $(show <"$tmp/want.txt") (default features)"
  else
    bad "$crate: router deps $(show <"$tmp/found.txt") != allowed $(show <"$tmp/want.txt") (default features)"
  fi

  # Column 3: forbidden external crates anywhere in the --all-features closure.
  tree "$crate" "$crate" --all-features >"$tmp/tree_all.txt"
  awk '{ print $1 }' "$tmp/tree_all.txt" | lines >"$tmp/names_all.txt"
  comm -12 "$tmp/names_all.txt" <(words "$forbidden") >"$tmp/hits.txt"
  if [ ! -s "$tmp/hits.txt" ]; then
    ok "$crate: none of $(words "$forbidden" | show) in its --all-features tree ($(wc -l <"$tmp/names_all.txt") crates)"
  else
    bad "$crate: forbidden crates in its --all-features tree: $(show <"$tmp/hits.txt")"
  fi

  # Column 4: direct dependency set, exact.
  if [ "$direct" != "-" ]; then
    direct_deps "$crate" | lines >"$tmp/direct.txt"
    words "$direct" >"$tmp/direct_want.txt"
    if cmp -s "$tmp/direct.txt" "$tmp/direct_want.txt"; then
      ok "$crate: direct deps == $(show <"$tmp/direct_want.txt")"
    else
      bad "$crate: direct deps $(show <"$tmp/direct.txt") != required $(show <"$tmp/direct_want.txt")"
    fi
  fi

  # Column 5: no executable behavior in the source tree.
  if [ "$behavior_dir" != "-" ]; then
    [ -d "$behavior_dir" ] || die "$crate: behavior directory $behavior_dir does not exist"
    mapfile -t rs_files < <(find "$behavior_dir" -type f -name '*.rs' | sort)
    [ "${#rs_files[@]}" -gt 0 ] || die "$crate: no .rs files under $behavior_dir"

    rc=0
    grep -nE -- "$BEHAVIOR_RE" "${rs_files[@]}" >"$tmp/grep.txt" || rc=$?
    case $rc in 0|1) ;; *) die "grep failed ($rc) under $behavior_dir" ;; esac
    # Attribute on its own line followed by a pub item: the single-line regex
    # above cannot see it.
    awk -v mark="$ALLOW_MARK" '
      FNR == 1 { prev = "" }
      prev ~ /^[[:space:]]*#\[doc\(hidden\)\][[:space:]]*$/ && $0 ~ /^[[:space:]]*pub/ \
        && index(prev, mark) == 0 { print FILENAME ":" FNR ":" $0 }
      { prev = $0 }
    ' "${rs_files[@]}" >>"$tmp/grep.txt"

    grep -F -- "$ALLOW_MARK" "$tmp/grep.txt" >"$tmp/grep_allowed.txt" || true
    grep -vF -- "$ALLOW_MARK" "$tmp/grep.txt" >"$tmp/grep_bad.txt" || true
    if [ -s "$tmp/grep_allowed.txt" ]; then
      echo "note  $crate: lines exempted by '$ALLOW_MARK':"
      sed 's/^/        /' "$tmp/grep_allowed.txt"
    fi
    if [ ! -s "$tmp/grep_bad.txt" ]; then
      ok "$crate: no executable behavior under $behavior_dir (${#rs_files[@]} files, $(wc -l <"$tmp/grep_allowed.txt") exempt lines)"
    else
      bad "$crate: executable behavior under $behavior_dir ($(wc -l <"$tmp/grep_bad.txt") lines):"
      sed 's/^/        /' "$tmp/grep_bad.txt"
    fi
  fi
done

# ---------------------------------------------------------------------------
# Composition crate: its adapters live behind features, so with no features it
# carries the kernel and the stages and none of the transport crates.
# ---------------------------------------------------------------------------
COMPOSITION="dynamo-kv-router"
COMPOSITION_ROUTER_DEPS="dynamo-admission dynamo-kv-events dynamo-kv-hashing dynamo-kv-hints dynamo-kv-indexer dynamo-multimodal dynamo-router-protocol dynamo-scheduler dynamo-session-affinity dynamo-slot-tracker dynamo-worker-selection"
in_workspace "$COMPOSITION" || die "$COMPOSITION is not a workspace package"
tree "$COMPOSITION" "$COMPOSITION" --no-default-features >"$tmp/comp_tree.txt"
awk -v c="$COMPOSITION" '$1 != c { print $1 }' "$tmp/comp_tree.txt" | lines >"$tmp/comp_names.txt"
comm -12 "$tmp/comp_names.txt" <(words "$ROUTER_FAMILY") >"$tmp/comp_found.txt"
words "$COMPOSITION_ROUTER_DEPS" >"$tmp/comp_want.txt"
if cmp -s "$tmp/comp_found.txt" "$tmp/comp_want.txt"; then
  ok "$COMPOSITION: router deps == $(show <"$tmp/comp_want.txt") (no features)"
else
  bad "$COMPOSITION: router deps $(show <"$tmp/comp_found.txt") != allowed $(show <"$tmp/comp_want.txt") (no features)"
fi
comm -12 "$tmp/comp_names.txt" <(words "$HOST_FORBIDDEN") >"$tmp/comp_hits.txt"
if [ ! -s "$tmp/comp_hits.txt" ]; then
  ok "$COMPOSITION: none of $(words "$HOST_FORBIDDEN" | show) in its no-features tree ($(wc -l <"$tmp/comp_names.txt") crates)"
else
  bad "$COMPOSITION: transport crates in its no-features tree: $(show <"$tmp/comp_hits.txt")"
fi
COMPOSITION_DIRECT_DEPS="anyhow arc-swap async-trait axum blake3 chrono crossbeam-queue dashmap derive_builder dynamo-admission dynamo-kv-events dynamo-kv-hashing dynamo-kv-hints dynamo-kv-indexer dynamo-multimodal dynamo-router-protocol dynamo-scheduler dynamo-session-affinity dynamo-slot-tracker dynamo-tokens dynamo-truthy dynamo-worker-selection dynamo-zmq fastrand flume futures-util http parking_lot prometheus reqwest rmp-serde rustc-hash serde serde_json serde_yaml thiserror tokio tokio-util tracing tracing-appender uuid xxhash-rust zeroize zmq"
direct_deps "$COMPOSITION" | lines >"$tmp/comp_direct.txt"
words "$COMPOSITION_DIRECT_DEPS" >"$tmp/comp_direct_want.txt"
if cmp -s "$tmp/comp_direct.txt" "$tmp/comp_direct_want.txt"; then
  ok "$COMPOSITION: direct deps == the $(wc -l <"$tmp/comp_direct_want.txt") listed (normal and optional)"
else
  bad "$COMPOSITION: direct deps $(show <"$tmp/comp_direct.txt") != required $(show <"$tmp/comp_direct_want.txt")"
fi

words "$table_crates" >"$tmp/table_crates.txt"

# ---------------------------------------------------------------------------
# Workspace: no normal dependency enables a test or bench feature on a table
# crate. Dev-dependencies are outside this graph, so a crate's own tests may
# turn them on; a production edge may not.
# ---------------------------------------------------------------------------
if ! cargo tree --workspace --locked -e normal,features --prefix none >"$tmp/ws.txt" 2>"$tmp/ws.err"; then
  die "cargo tree --workspace -e normal,features: $(tr '\n' ' ' <"$tmp/ws.err")"
fi
[ -s "$tmp/ws.txt" ] || die "cargo tree --workspace printed nothing"
while read -r c; do
  grep -qE "^$c v" "$tmp/ws.txt" || die "$c missing from the workspace graph"
done <"$tmp/table_crates.txt"
crate_alt="$(paste -sd'|' "$tmp/table_crates.txt")"
feature_alt="$(words "$ROUTER_TEST_FEATURES" | paste -sd'|')"
grep -E "^($crate_alt) feature \"($feature_alt)\"" "$tmp/ws.txt" | sort -u >"$tmp/ws_hits.txt" || true
if [ ! -s "$tmp/ws_hits.txt" ]; then
  ok "workspace: no normal dependency enables $(words "$ROUTER_TEST_FEATURES" | show) on a router crate"
else
  bad "workspace: test or bench features enabled by a normal dependency: $(show <"$tmp/ws_hits.txt")"
fi

# ---------------------------------------------------------------------------
# Source rules over every router crate (SPLIT_V2 G10): no #[doc(hidden)] (an
# item is documented or absent), no cross-crate glob re-export, and the
# README.md / AGENTS.md / CLAUDE.md trio per crate. The composition crate is
# exempt for its one remaining glob: `test_utils` over the kernel's test
# helpers, compiled for tests and the `bench` feature only.
# ---------------------------------------------------------------------------
ROUTER_CRATE_DIRS="lib/multimodal lib/error lib/router/protocol lib/router/kv-indexer lib/router/slot-tracker lib/router/session-affinity lib/router/worker-selection lib/router/admission lib/router/scheduler lib/kv-router lib/kv-events lib/kv-hashing lib/kv-hints lib/zmq"
GLOB_EXEMPT_DIRS="lib/kv-router"
for d in $ROUTER_CRATE_DIRS; do [ -d "$d/src" ] || die "$d/src does not exist"; done
mapfile -t src_files < <(for d in $ROUTER_CRATE_DIRS; do find "$d/src" -type f -name '*.rs'; done | sort)
[ "${#src_files[@]}" -gt 0 ] || die "no .rs files under the router crates"
rc=0
grep -nF -- '#[doc(hidden)]' "${src_files[@]}" >"$tmp/hidden.txt" || rc=$?
case $rc in 0|1) ;; *) die "grep failed ($rc) scanning for doc(hidden)" ;; esac
grep -vF -- "$ALLOW_MARK" "$tmp/hidden.txt" >"$tmp/hidden_bad.txt" || true
if [ ! -s "$tmp/hidden_bad.txt" ]; then
  ok "router crates: no #[doc(hidden)] (${#src_files[@]} files)"
else
  bad "router crates: #[doc(hidden)] items ($(wc -l <"$tmp/hidden_bad.txt") lines):"
  sed 's/^/        /' "$tmp/hidden_bad.txt"
fi
exempt_re="^($(words "$GLOB_EXEMPT_DIRS" | paste -sd'|'))/"
mapfile -t glob_files < <(printf '%s\n' "${src_files[@]}" | grep -vE -- "$exempt_re" || true)
[ "${#glob_files[@]}" -gt 0 ] || die "no .rs files left after exempting $GLOB_EXEMPT_DIRS"
rc=0
grep -nE -- '^[[:space:]]*pub(\([a-z]+\))? use dynamo_[a-z_]+::[A-Za-z0-9_:]*\*;' "${glob_files[@]}" >"$tmp/globs.txt" || rc=$?
case $rc in 0|1) ;; *) die "grep failed ($rc) scanning for glob re-exports" ;; esac
if [ ! -s "$tmp/globs.txt" ]; then
  ok "router crates: no cross-crate glob re-export (${#glob_files[@]} files outside $(words "$GLOB_EXEMPT_DIRS" | show))"
else
  bad "router crates: cross-crate glob re-exports ($(wc -l <"$tmp/globs.txt") lines):"
  sed 's/^/        /' "$tmp/globs.txt"
fi
: >"$tmp/guides_bad.txt"
for d in $ROUTER_CRATE_DIRS; do
  [ -f "$d/README.md" ] || echo "$d/README.md missing" >>"$tmp/guides_bad.txt"
  [ -f "$d/AGENTS.md" ] || echo "$d/AGENTS.md missing" >>"$tmp/guides_bad.txt"
  if [ ! -f "$d/CLAUDE.md" ]; then
    echo "$d/CLAUDE.md missing" >>"$tmp/guides_bad.txt"
  elif [ "$(tr -d '[:space:]' <"$d/CLAUDE.md")" != "@AGENTS.md" ]; then
    echo "$d/CLAUDE.md must contain exactly @AGENTS.md" >>"$tmp/guides_bad.txt"
  fi
done
if [ ! -s "$tmp/guides_bad.txt" ]; then
  ok "router crates: README.md, AGENTS.md, CLAUDE.md (@AGENTS.md) present in all $(words "$ROUTER_CRATE_DIRS" | wc -l) crates"
else
  bad "router crates: agent guides:"
  sed 's/^/        /' "$tmp/guides_bad.txt"
fi

# ---------------------------------------------------------------------------
# --host: no table crate reaches a heavy external crate in a host's own graph
# (feature unification included). HOSTS are workspace packages; HOSTS_EXTERNAL
# have their own manifest and lockfile and resolve with the features listed.
# ---------------------------------------------------------------------------
check_host() {
  local host="$1"; shift
  tree "$host" "$host" "$@" >"$tmp/host_tree.txt"
  for x in $HOST_FORBIDDEN; do
    mapfile -t versions < <(awk -v n="$x" '$1 == n { print $2 }' "$tmp/host_tree.txt")
    if [ "${#versions[@]}" -eq 0 ]; then
      ok "$host: $x absent from its normal tree"
      continue
    fi
    # One inverted tree per resolved version: a bare name is ambiguous to
    # cargo when two versions coexist (reqwest 0.12 and 0.13 in dynamo-llm).
    for v in "${versions[@]}"; do
      spec="$x@${v#v}"
      tree "$host" "$x" "$@" -i "$spec" >"$tmp/inv_tree.txt"
      awk '{ print $1 }' "$tmp/inv_tree.txt" | lines >"$tmp/inv.txt"
      comm -12 "$tmp/inv.txt" "$tmp/table_crates.txt" >"$tmp/inv_hits_all.txt"
      # drop the adapter crate that exists to reach this dependency, if one is declared for it
      exempt=""
      for e in $HOST_ADAPTER_EXEMPT; do [ "${e#*:}" = "$x" ] && exempt="${e%%:*}"; done
      if [ -n "$exempt" ]; then
        grep -vxF -- "$exempt" "$tmp/inv_hits_all.txt" >"$tmp/inv_hits.txt" || true
      else
        cp "$tmp/inv_hits_all.txt" "$tmp/inv_hits.txt"
      fi
      if [ ! -s "$tmp/inv_hits.txt" ]; then
        ok "$host: no router crate reaches $spec"
      else
        bad "$host: router crates reach $spec: $(show <"$tmp/inv_hits.txt")"
      fi
    done
  done
}
if [ "$host_mode" -eq 1 ]; then
  for host in $HOSTS; do
    if ! in_workspace "$host"; then
      echo "skip  $host: not a workspace package"
      continue
    fi
    check_host "$host"
  done
  for spec in $HOSTS_EXTERNAL; do
    IFS='|' read -r host manifest features <<<"$spec"
    [ -f "$manifest" ] || die "$host: manifest $manifest does not exist"
    if [ -n "$features" ]; then
      check_host "$host" --manifest-path "$manifest" --features "$features"
    else
      check_host "$host" --manifest-path "$manifest"
    fi
  done
fi

echo "router layering: $checks checks, $failures failures"
if [ "$failures" -eq 0 ]; then exit 0; fi
exit 1
