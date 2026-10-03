#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Ship the campaign commit, the deploy scripts and a frozen plan to the GPU cluster, and prepare the
# source tree there (run on the workstation; touches only the cluster's login node, no allocation).
#
# Usage: stage_source.sh PLAN_DIR [COMMIT]
#   COMMIT defaults to the worktree HEAD. The commit must already contain this deploy directory.
#
# 1. A thin git bundle COMMIT ^BASE, where BASE (LR_BUNDLE_BASE, default 373bc9f9 on main) is
#    already in the cluster-side clone LR_BASE_REPO, is verified locally and copied to
#    LR_LIVE_ROOT/bundles/.
# 2. On the login node: a local clone of LR_BASE_REPO, fetch of the bundle, detached
#    checkout of COMMIT, then the live-only engine shims from patches/ applied as an uncommitted
#    diff. git is not in the vLLM image, so every git step happens here.
# 3. The plan directory is copied to LR_LIVE_ROOT/plans/<plan_id>/ and the deploy directory as of
#    COMMIT is the one jobs run (LR_SRC/benchmarks/learned_routing/live/deploy).
# 4. The node-side site settings (lr_node_site_env) are written to site.env in that deploy
#    directory, where common.sh reads them on the allocated nodes.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/common.sh"
lr_need LR_SSH_ALIAS LR_LIVE_ROOT LR_BASE_REPO
[[ $# -ge 1 ]] || die "usage: stage_source.sh PLAN_DIR [COMMIT]"
plan_dir="$(cd "$1" && pwd)"
wt="${LR_WT:-$(cd "$here/../../../.." && pwd)}"
commit="$(git -C "$wt" rev-parse "${2:-HEAD}")"
base="${LR_BUNDLE_BASE:-373bc9f92944507f8a82dd602dc391c0f83d61f1}"
base_repo="$LR_BASE_REPO"
ssh_alias="$LR_SSH_ALIAS"
tag="${commit:0:12}"
plan_id="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["plan_id"])' "$plan_dir/PLAN.json")"
plan_commit="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"]["commit"])' "$plan_dir/PLAN.json")"
for path in lib components Cargo.lock; do
  [[ "$(git -C "$wt" rev-parse "$commit:$path")" == "$(git -C "$wt" rev-parse "$plan_commit:$path")" ]] \
    || die "$path at $tag differs from the plan's source commit ${plan_commit:0:12}"
done
git -C "$wt" cat-file -e "$commit:benchmarks/learned_routing/live/deploy/job_node.sh" \
  || die "commit $tag does not contain the deploy scripts; commit them first"
git -C "$wt" merge-base --is-ancestor "$base" "$commit" || die "$base is not an ancestor of $tag"

# A bundle needs a ref; create it in a fresh object-sharing clone so the worktree's refs stay
# untouched. The scratch directory is printed for the cleanup ledger (nothing is deleted here).
scratch="$(mktemp -d "${LR_LOCAL_SCRATCH:-${TMPDIR:-/tmp}}/lr-live-stage-$tag.XXXXXX")"
bundle="$scratch/lr-$tag.bundle"
ref="refs/heads/lr-live-$tag"
git clone --quiet --no-checkout --shared "$wt" "$scratch/clone"
git -C "$scratch/clone" update-ref "$ref" "$commit"
git -C "$scratch/clone" bundle create --quiet "$bundle" "$ref" "^$base"
log "local scratch (list it for cleanup): $scratch"
git -C "$wt" bundle verify "$bundle" > /dev/null
bundle_sha="$(sha256sum "$bundle" | awk '{print $1}')"
log "bundle $bundle ($(du -h "$bundle" | cut -f1)) sha256 $bundle_sha"

ssh -o BatchMode=yes "$ssh_alias" "mkdir -p '$LR_LIVE_ROOT/bundles' '$LR_LIVE_ROOT/src' '$LR_LIVE_ROOT/plans'"
scp -q "$bundle" "$ssh_alias:$LR_LIVE_ROOT/bundles/"
# A fresh incoming name per run: scp -r into an existing directory would nest the copy.
incoming="$LR_LIVE_ROOT/plans/.incoming-$plan_id-$(date -u +%Y%m%dT%H%M%SZ)-$$"
scp -q -r "$plan_dir" "$ssh_alias:$incoming"

ssh -o BatchMode=yes "$ssh_alias" bash -s -- "$LR_LIVE_ROOT" "$tag" "$commit" "$bundle_sha" \
  "$base_repo" "$ref" "$plan_id" "$incoming" <<'REMOTE'
set -euo pipefail
root=$1 tag=$2 commit=$3 bundle_sha=$4 base_repo=$5 ref=$6 plan_id=$7 incoming=$8
bundle="$root/bundles/lr-$tag.bundle"
src="$root/src/$tag"
printf '%s  %s\n' "$bundle_sha" "$bundle" | sha256sum -c - > /dev/null
manifest="$src.source-manifest.json"
# LFS test media are not needed to build or serve, and the base repository has no LFS objects.
export GIT_LFS_SKIP_SMUDGE=1
if [[ ! -s "$manifest" ]]; then
  # Resumable: a partial clone or checkout from an interrupted run is completed, not removed.
  if [[ ! -d "$src/.git" ]]; then
    test "$(git -C "$base_repo" rev-parse --is-shallow-repository)" = false
    git clone --quiet --no-checkout "$base_repo" "$src"
  fi
  git -C "$src" bundle verify "$bundle" > /dev/null
  git -C "$src" fetch --quiet "$bundle" "$ref:refs/remotes/live/$tag"
  git -C "$src" -c advice.detachedHead=false checkout --quiet --force --detach "$commit"
  for patch in "$src"/benchmarks/learned_routing/live/deploy/patches/*.patch; do
    git -C "$src" apply --check "$patch"
    git -C "$src" apply "$patch"
  done
fi
test "$(git -C "$src" rev-parse HEAD)" = "$commit"
[[ -s "$manifest" ]] || python3 - "$src" "$commit" "$bundle" "$bundle_sha" > "$manifest.partial" <<'PY'
import hashlib, json, subprocess, sys
src, commit, bundle, bundle_sha = sys.argv[1:5]
git = lambda *a: subprocess.run(["git", "-C", src, *a], check=True, capture_output=True).stdout
patches = sorted(subprocess.run(["bash", "-c", f"ls {src}/benchmarks/learned_routing/live/deploy/patches/*.patch"],
                                capture_output=True, text=True).stdout.split())
print(json.dumps({
    "commit": commit,
    "bundle": bundle,
    "bundle_sha256": bundle_sha,
    "trees": {p: git("rev-parse", f"HEAD:{p}").decode().strip() for p in ("lib", "components", "Cargo.lock")},
    "patches": {p.rsplit("/", 1)[1]: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in patches},
    "diff_files": git("diff", "--name-only").decode().split(),
    "diff_sha256": hashlib.sha256(git("diff", "--binary")).hexdigest(),
}, indent=1, sort_keys=True))
PY
[[ -s "$manifest" ]] || mv "$manifest.partial" "$manifest"
plan_dst="$root/plans/$plan_id"
if [[ -d "$plan_dst" ]]; then
  # Plans are write-once: an identical re-upload is kept aside as a duplicate, never merged.
  diff -r -q "$incoming" "$plan_dst" > /dev/null
  mv -T "$incoming" "$incoming.dup"
else
  mv -T "$incoming" "$plan_dst"
fi
cat "$manifest"
echo "plan_dir=$plan_dst"
echo "deploy_dir=$src/benchmarks/learned_routing/live/deploy"
REMOTE
lr_node_site_env | ssh -o BatchMode=yes "$ssh_alias" \
  "cat > '$LR_LIVE_ROOT/src/$tag/benchmarks/learned_routing/live/deploy/site.env'"
echo "LR_COMMIT=$commit"
