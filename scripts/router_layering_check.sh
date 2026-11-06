#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Router crate layering gate. Checks, from `cargo metadata` and `cargo tree`,
# that every router crate has exactly the Dynamo dependency edges the split
# design allows and reaches none of the crates its layer forbids, that the
# plugin crates depend on dynamo-kv-router alone, and that the protocol crate
# carries no executable behavior.
#
# Usage: scripts/router_layering_check.sh [--host]
#
#   (no flag)  per-crate checks from TABLE below (exact direct Dynamo
#              dependencies, forbidden crates anywhere in the closure), the
#              protocol behavior grep, the composition crate's no-features
#              tree, the plugin table, the workspace test-feature check, and
#              the source rules (no #[doc(hidden)], no cross-crate glob
#              re-export); run it locally on demand, about ten seconds.
#   --host     additionally resolve the host packages in HOSTS and
#              HOSTS_EXTERNAL and fail if any table crate reaches a heavy
#              external crate in the host's own graph (feature unification
#              included); the rust-clippy job runs this.
#
# Exit status: 0 all checks passed, 1 at least one layering check failed,
# 2 the gate could not run (missing crate, empty cargo output, bad usage).
# The gate fails closed: an empty or failed `cargo tree` is an error, never a
# pass. Runs from any directory inside the repository.
set -Eeuo pipefail
export LC_ALL=C

# ---------------------------------------------------------------------------
# Table. One row per router crate; add a crate by adding one line. Fields are
# separated by '|' (surrounding spaces ignored); '-' means "none".
#
#   1 crate       cargo package name
#   2 Dynamo deps the EXACT set of dynamo-* crates in its direct normal
#                 [dependencies] (optional included, dev and build excluded),
#                 from `cargo metadata`; the crate's AGENTS.md states the same
#                 set
#   3 forbidden   crates that must not appear anywhere in its normal
#                 dependency tree with --all-features
# ---------------------------------------------------------------------------
TABLE=(
  "dynamo-error | - | dynamo-runtime dynamo-kv-router dynamo-router-protocol dynamo-kv-events dynamo-kv-hashing dynamo-multimodal dynamo-tokens tokio anyhow thiserror tracing async-trait derive_builder prometheus axum hyper reqwest zmq"
  "dynamo-multimodal | - | dynamo-runtime dynamo-kv-router dynamo-router-protocol dynamo-kv-events dynamo-kv-hashing dynamo-kv-hints dynamo-kv-indexer dynamo-scheduler dynamo-tokens tokio anyhow tracing async-trait derive_builder thiserror prometheus axum hyper reqwest zmq"
  "dynamo-router-protocol | - | dynamo-runtime dynamo-kv-router dynamo-kv-hashing dynamo-multimodal dynamo-tokens tokio anyhow tracing async-trait derive_builder prometheus axum hyper reqwest zmq"
  "dynamo-kv-hashing | dynamo-multimodal dynamo-tokens | dynamo-runtime dynamo-kv-router dynamo-router-protocol tokio anyhow tracing async-trait prometheus axum hyper reqwest zmq"
  "dynamo-kv-events | dynamo-kv-hashing dynamo-multimodal dynamo-router-protocol | dynamo-runtime dynamo-kv-router tokio anyhow async-trait prometheus axum hyper reqwest zmq"
  "dynamo-kv-hints | dynamo-kv-hashing dynamo-kv-indexer dynamo-router-protocol | dynamo-runtime dynamo-kv-router dynamo-slot-tracker dynamo-session-affinity dynamo-worker-selection dynamo-admission dynamo-scheduler prometheus axum hyper reqwest zmq"
  "dynamo-kv-indexer | dynamo-kv-events dynamo-kv-hashing dynamo-router-protocol dynamo-tokens | dynamo-runtime dynamo-kv-router dynamo-kv-hints axum hyper reqwest zmq"
  "dynamo-slot-tracker | dynamo-router-protocol dynamo-tokens | dynamo-runtime dynamo-kv-router dynamo-kv-indexer dynamo-kv-events dynamo-kv-hashing dynamo-multimodal axum hyper reqwest zmq prometheus"
  "dynamo-session-affinity | dynamo-router-protocol | dynamo-runtime dynamo-kv-router dynamo-kv-indexer dynamo-slot-tracker dynamo-worker-selection dynamo-admission dynamo-kv-events dynamo-kv-hashing dynamo-multimodal axum hyper reqwest zmq prometheus"
  "dynamo-worker-selection | dynamo-router-protocol dynamo-session-affinity dynamo-slot-tracker | dynamo-runtime dynamo-kv-router dynamo-kv-indexer dynamo-kv-events dynamo-kv-hashing dynamo-multimodal axum hyper reqwest zmq prometheus"
  "dynamo-admission | dynamo-router-protocol dynamo-session-affinity dynamo-slot-tracker dynamo-tokens | dynamo-runtime dynamo-kv-router dynamo-kv-indexer dynamo-worker-selection dynamo-kv-events dynamo-kv-hashing dynamo-multimodal axum hyper reqwest zmq prometheus"
  "dynamo-scheduler | dynamo-admission dynamo-kv-events dynamo-kv-hashing dynamo-kv-hints dynamo-kv-indexer dynamo-multimodal dynamo-router-protocol dynamo-session-affinity dynamo-slot-tracker dynamo-tokens dynamo-worker-selection | dynamo-runtime dynamo-kv-router axum hyper reqwest zmq prometheus"
)

# Every crate the split produces (SPLIT_V2 3a-final). The composition check
# below compares the ROUTER_FAMILY crates found in dynamo-kv-router's
# no-features tree against COMPOSITION_ROUTER_DEPS.
ROUTER_FAMILY="dynamo-multimodal dynamo-error dynamo-kv-hashing dynamo-router-protocol dynamo-kv-events dynamo-kv-hints dynamo-kv-indexer dynamo-slot-tracker dynamo-session-affinity dynamo-worker-selection dynamo-admission dynamo-scheduler dynamo-metrics dynamo-kv-router"

# Plugin crates: a statically linked policy or classifier crate depends on
# dynamo-kv-router alone and imports through plugins::prelude. Each row is the
# EXACT set of dynamo-* direct normal dependencies (dev-dependencies may name
# the kernel for tests and benches). The example catalog aggregates its sibling
# example policies; the EPP example links the catalog into the gateway binary.
PLUGIN_TABLE=(
  "dynamo-router-policies | dynamo-kv-router"
  "dynamo-worker-selection-policy-catalog | dynamo-kv-router"
  "dynamo-custom-policy-example-soft-pin-repin | dynamo-kv-router"
  "dynamo-custom-policy-example-simple-filter-score-pick | dynamo-kv-router"
  "dynamo-custom-policy-example-disagg-filter-score-pick | dynamo-kv-router"
  "dynamo-custom-policy-example-simple-stacked-score-pick | dynamo-kv-router"
  "dynamo-custom-policy-example-catalog | dynamo-kv-router dynamo-custom-policy-example-soft-pin-repin dynamo-custom-policy-example-simple-filter-score-pick dynamo-custom-policy-example-disagg-filter-score-pick dynamo-custom-policy-example-simple-stacked-score-pick"
  "dynamo-custom-policy-example-epp | dynamo-kv-router dynamo-custom-policy-example-catalog dynamo-ext-proc"
)

# Features no normal (non-dev) dependency may enable on a table crate, checked
# on the workspace graph (`cargo tree --workspace -e normal,features`): the
# protocol crate's chosen-value AttemptId constructors live behind them.
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

# Executable behavior the protocol crate must not contain (SPLIT_V2 G9).
PROTOCOL_CRATE="dynamo-router-protocol"
PROTOCOL_BEHAVIOR_DIR="lib/router/protocol/src"
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

# check_dynamo_direct PKG ALLOWED LABEL: the dynamo-* direct normal
# dependencies of PKG equal ALLOWED (a space-separated set, '-' for none).
check_dynamo_direct() {
  local crate="$1" allowed="$2" label="$3"
  direct_deps "$crate" | grep -E '^dynamo-' | lines >"$tmp/dd_found.txt" || true
  words "$allowed" >"$tmp/dd_want.txt"
  if cmp -s "$tmp/dd_found.txt" "$tmp/dd_want.txt"; then
    ok "$crate: Dynamo deps == $(show <"$tmp/dd_want.txt") ($label)"
  else
    bad "$crate: Dynamo deps $(show <"$tmp/dd_found.txt") != allowed $(show <"$tmp/dd_want.txt") ($label)"
  fi
}

table_crates=""

# ---------------------------------------------------------------------------
# Per-crate checks
# ---------------------------------------------------------------------------
for row in "${TABLE[@]}"; do
  IFS='|' read -r crate allowed forbidden <<<"$row"
  crate="$(trim "$crate")"
  allowed="$(trim "$allowed")"
  forbidden="$(trim "$forbidden")"
  table_crates="$table_crates $crate"

  in_workspace "$crate" || die "$crate is not a workspace package (root Cargo.toml members)"

  # Column 2: direct Dynamo dependencies, exact set.
  check_dynamo_direct "$crate" "$allowed" "direct"

  # Column 3: forbidden crates anywhere in the --all-features closure.
  tree "$crate" "$crate" --all-features >"$tmp/tree_all.txt"
  awk '{ print $1 }' "$tmp/tree_all.txt" | lines >"$tmp/names_all.txt"
  comm -12 "$tmp/names_all.txt" <(words "$forbidden") >"$tmp/hits.txt"
  if [ ! -s "$tmp/hits.txt" ]; then
    ok "$crate: none of $(words "$forbidden" | show) in its --all-features tree ($(wc -l <"$tmp/names_all.txt") crates)"
  else
    bad "$crate: forbidden crates in its --all-features tree: $(show <"$tmp/hits.txt")"
  fi
done

# ---------------------------------------------------------------------------
# Protocol: no executable behavior in the source tree (SPLIT_V2 G9). A line
# carrying ALLOW_MARK is exempt and printed for the reviewer.
# ---------------------------------------------------------------------------
[ -d "$PROTOCOL_BEHAVIOR_DIR" ] || die "$PROTOCOL_CRATE: behavior directory $PROTOCOL_BEHAVIOR_DIR does not exist"
mapfile -t rs_files < <(find "$PROTOCOL_BEHAVIOR_DIR" -type f -name '*.rs' | sort)
[ "${#rs_files[@]}" -gt 0 ] || die "$PROTOCOL_CRATE: no .rs files under $PROTOCOL_BEHAVIOR_DIR"
rc=0
grep -nE -- "$BEHAVIOR_RE" "${rs_files[@]}" >"$tmp/grep.txt" || rc=$?
case $rc in 0|1) ;; *) die "grep failed ($rc) under $PROTOCOL_BEHAVIOR_DIR" ;; esac
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
  echo "note  $PROTOCOL_CRATE: lines exempted by '$ALLOW_MARK':"
  sed 's/^/        /' "$tmp/grep_allowed.txt"
fi
if [ ! -s "$tmp/grep_bad.txt" ]; then
  ok "$PROTOCOL_CRATE: no executable behavior under $PROTOCOL_BEHAVIOR_DIR (${#rs_files[@]} files, $(wc -l <"$tmp/grep_allowed.txt") exempt lines)"
else
  bad "$PROTOCOL_CRATE: executable behavior under $PROTOCOL_BEHAVIOR_DIR ($(wc -l <"$tmp/grep_bad.txt") lines):"
  sed 's/^/        /' "$tmp/grep_bad.txt"
fi

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

# ---------------------------------------------------------------------------
# Plugin crates: dynamo-kv-router is the one Dynamo crate they name.
# ---------------------------------------------------------------------------
for row in "${PLUGIN_TABLE[@]}"; do
  IFS='|' read -r crate allowed <<<"$row"
  crate="$(trim "$crate")"
  allowed="$(trim "$allowed")"
  in_workspace "$crate" || die "$crate is not a workspace package (root Cargo.toml members)"
  check_dynamo_direct "$crate" "$allowed" "plugin rule"
done

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
feature_alt="$(words "$ROUTER_TEST_FEATURES" | paste -sd'|' -)"
grep -E "^($crate_alt) feature \"($feature_alt)\"" "$tmp/ws.txt" | sort -u >"$tmp/ws_hits.txt" || true
if [ ! -s "$tmp/ws_hits.txt" ]; then
  ok "workspace: no normal dependency enables $(words "$ROUTER_TEST_FEATURES" | show) on a router crate"
else
  bad "workspace: test or bench features enabled by a normal dependency: $(show <"$tmp/ws_hits.txt")"
fi

# ---------------------------------------------------------------------------
# Source rules over every router crate (SPLIT_V2 G10): no #[doc(hidden)] (an
# item is documented or absent) and no cross-crate glob re-export. The
# composition crate is exempt for its one remaining glob: `test_utils` over
# the kernel's test helpers, compiled for tests and the `bench` feature only.
# ---------------------------------------------------------------------------
ROUTER_CRATE_DIRS="lib/multimodal lib/error lib/router/protocol lib/router/kv-indexer lib/router/slot-tracker lib/router/session-affinity lib/router/worker-selection lib/router/admission lib/router/scheduler lib/router/kv-router lib/kv-events lib/kv-hashing lib/router/kv-hints"
GLOB_EXEMPT_DIRS="lib/router/kv-router"
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
exempt_re="^($(words "$GLOB_EXEMPT_DIRS" | paste -sd'|' -))/"
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
      comm -12 "$tmp/inv.txt" "$tmp/table_crates.txt" >"$tmp/inv_hits.txt"
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
