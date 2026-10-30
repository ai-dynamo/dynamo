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
#   (no flag)  per-crate checks from TABLE below plus the workspace feature
#              check; pre-commit runs this.
#   --host     additionally resolve the host packages in HOSTS and fail if any
#              table crate reaches a heavy external crate in the host's own
#              graph (feature unification included); CI runs this.
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
  "dynamo-router-protocol | - | dynamo-runtime dynamo-kv-router dynamo-kv-hashing dynamo-tokens tokio anyhow tracing async-trait derive_builder prometheus axum hyper reqwest zmq | serde serde_json thiserror | lib/router/protocol/src"
  "dynamo-kv-events | dynamo-router-protocol dynamo-kv-hashing | dynamo-runtime dynamo-kv-router tokio anyhow async-trait prometheus axum hyper reqwest zmq | dynamo-kv-hashing dynamo-router-protocol rmp-serde rustc-hash serde serde_json thiserror tracing | -"
  "dynamo-kv-hashing | dynamo-router-protocol | dynamo-runtime dynamo-kv-router tokio anyhow tracing async-trait prometheus axum hyper reqwest zmq | derive_builder dynamo-router-protocol dynamo-tokens serde serde_json thiserror xxhash-rust | -"
)

# Every crate the split produces (SPLIT_V2 3a-final). Column 2 counts any of
# these found in a tree as a router dependency, so a crate that reaches for the
# monolith or a sibling fails before that sibling has a row of its own.
ROUTER_FAMILY="dynamo-error dynamo-kv-hashing dynamo-router-protocol dynamo-kv-events dynamo-zmq dynamo-kv-indexer dynamo-slot-tracker dynamo-session-affinity dynamo-worker-selection dynamo-admission dynamo-scheduler dynamo-metrics dynamo-kv-router"

# Features no normal (non-dev) dependency may enable on a table crate, checked
# on the workspace graph (`cargo tree --workspace -e normal,features`).
ROUTER_TEST_FEATURES="testing bench profile"

# --host: packages whose resolved graph must not route a table crate to one of
# HOST_FORBIDDEN. Hosts missing from the workspace are skipped with a notice.
HOSTS="dynamo-llm dynamo-ext-proc dynamo-mocker"
HOST_FORBIDDEN="prometheus axum hyper reqwest zmq"

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
# --host: no table crate reaches a heavy external crate in a host's own graph
# ---------------------------------------------------------------------------
if [ "$host_mode" -eq 1 ]; then
  for host in $HOSTS; do
    if ! in_workspace "$host"; then
      echo "skip  $host: not a workspace package"
      continue
    fi
    tree "$host" "$host" >"$tmp/host_tree.txt"
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
        tree "$host" "$x" -i "$spec" >"$tmp/inv_tree.txt"
        awk '{ print $1 }' "$tmp/inv_tree.txt" | lines >"$tmp/inv.txt"
        comm -12 "$tmp/inv.txt" "$tmp/table_crates.txt" >"$tmp/inv_hits.txt"
        if [ ! -s "$tmp/inv_hits.txt" ]; then
          ok "$host: no router crate reaches $spec"
        else
          bad "$host: router crates reach $spec: $(show <"$tmp/inv_hits.txt")"
        fi
      done
    done
  done
fi

echo "router layering: $checks checks, $failures failures"
if [ "$failures" -eq 0 ]; then exit 0; fi
exit 1
