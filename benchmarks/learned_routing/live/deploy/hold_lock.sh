#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Cooperative GPU hold lock on the workstation, shared with any other sessions that hold nodes on
# the same GPU cluster.
#
#   hold_lock.sh status
#   hold_lock.sh acquire OWNER START HOURS [--take-expired]
#   hold_lock.sh release START
#
# Protocol: a GPU hold exists only while LR_HOLD_LOCK_DIR/ACTIVE exists (default
# ~/.lr-gpu-hold; set it in site.env to share a lock directory with other tools). ACTIVE is a tab-separated record (owner, start, created_utc,
# hard_expiry_utc) created exclusively; the hard expiry is at most 3 h after creation. Release
# renames it to released-<START>-<UTC stamp>; nothing in the directory is ever deleted. An ACTIVE
# past its hard expiry is renamed to expired-<its start>-<UTC stamp> only with --take-expired.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
dir="${LR_HOLD_LOCK_DIR:-$HOME/.lr-gpu-hold}"
active="$dir/ACTIVE"
stamp() { date -u +%Y%m%dT%H%M%SZ; }
field() { awk -F'\t' -v k="$1" '$1 == k {print $2}' "$active"; }
die() {
  echo "error: $*" >&2
  exit 1
}

case "${1:-}" in
  status)
    if [[ -e "$active" ]]; then
      cat "$active"
    else
      echo "free"
    fi
    ;;
  acquire)
    [[ $# -ge 4 ]] || die "usage: hold_lock.sh acquire OWNER START HOURS [--take-expired]"
    owner=$2 start=$3 hours=$4 take_expired=${5:-}
    [[ "$start" =~ ^[A-Za-z0-9._-]+$ ]] || die "START must be [A-Za-z0-9._-]+"
    awk -v h="$hours" 'BEGIN { exit !(h > 0 && h <= 3) }' || die "HOURS must be in (0, 3]"
    mkdir -p "$dir"
    if [[ -e "$active" ]]; then
      expiry="$(field hard_expiry_utc)"
      if [[ -n "$expiry" && "$(date -u -d "$expiry" +%s)" -lt "$(date -u +%s)" && "$take_expired" == --take-expired ]]; then
        mv "$active" "$dir/expired-$(field start)-$(stamp)"
      else
        cat "$active" >&2
        die "GPU hold lock is held (hard expiry $expiry)"
      fi
    fi
    created="$(date -u +%FT%TZ)"
    expiry="$(date -u -d "+$(awk -v h="$hours" 'BEGIN { printf "%d", h * 3600 }') seconds" +%FT%TZ)"
    (
      set -o noclobber
      printf 'owner\t%s\nstart\t%s\ncreated_utc\t%s\nhard_expiry_utc\t%s\n' \
        "$owner" "$start" "$created" "$expiry" > "$active"
    ) 2> /dev/null || die "lost the race for $active"
    cat "$active"
    ;;
  release)
    [[ $# -eq 2 ]] || die "usage: hold_lock.sh release START"
    [[ -e "$active" ]] || die "no ACTIVE lock"
    [[ "$(field start)" == "$2" ]] || die "ACTIVE belongs to start '$(field start)', not '$2'"
    mv "$active" "$dir/released-$2-$(stamp)"
    echo "released $2"
    ;;
  *)
    die "usage: hold_lock.sh status | acquire OWNER START HOURS [--take-expired] | release START"
    ;;
esac
