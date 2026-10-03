#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Two-node variant only: stage a pinned, checksummed etcd release on the cluster's shared filesystem
# (workstation side; downloads locally, copies to LR_LIVE_ROOT/tools/). File discovery uses inotify on one node, so
# workers on a second node need etcd discovery.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/common.sh"
lr_need LR_SSH_ALIAS LR_LIVE_ROOT
version="${LR_ETCD_VERSION:-v3.5.21}"
sha256="${LR_ETCD_SHA256:-adddda4b06718e68671ffabff2f8cee48488ba61ad82900e639d108f2148501c}"
name="etcd-$version-linux-amd64"
scratch="$(mktemp -d "${TMPDIR:-/tmp}/lr-etcd.XXXXXX")"
curl -sfL --max-time 300 -o "$scratch/$name.tar.gz" \
  "https://github.com/etcd-io/etcd/releases/download/$version/$name.tar.gz"
printf '%s  %s\n' "$sha256" "$scratch/$name.tar.gz" | sha256sum -c -
ssh -o BatchMode=yes "$LR_SSH_ALIAS" "mkdir -p '$LR_LIVE_ROOT/tools'"
scp -q "$scratch/$name.tar.gz" "$LR_SSH_ALIAS:$LR_LIVE_ROOT/tools/"
ssh -o BatchMode=yes "$LR_SSH_ALIAS" "cd '$LR_LIVE_ROOT/tools' \
  && printf '%s  %s\n' '$sha256' '$name.tar.gz' | sha256sum -c - && tar -xzf '$name.tar.gz' \
  && '$LR_LIVE_ROOT/tools/$name/etcd' --version | head -1"
echo "local scratch (list it for cleanup): $scratch"
