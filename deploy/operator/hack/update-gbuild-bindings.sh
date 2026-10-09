#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Usage: hack/update-gbuild-bindings.sh /path/to/nvidia-lpu/capnp
# Copy upstream-generated bindings at this exact revision, then rewrite Go imports.
# Replace the pin with the final merged revision of nvidia-lpu/capnp#113 before merging.
revision=6eea97de00c925263d755259ea7376fdba7216e5
source_repo="${1:?Usage: update-gbuild-bindings.sh /path/to/nvidia-lpu/capnp}"
operator_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="$operator_root/internal/dynamo/lpx/manifest"
staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT

for family in common deployment; do
    git -C "$source_repo" show "$revision:go/gbuild_$family/v1/gbuild_$family.capnp.go" > "$staging/$family.go"
    sed -i \
        's|github.com/nvidia-lpu/capnp/v12/go/gbuild_common/v1|github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/common/v1|g' \
        "$staging/$family.go"
    gofmt -w "$staging/$family.go"
done

for family in common deployment; do
    cp "$staging/$family.go" "$output_root/$family/v1/gbuild_$family.capnp.go"
done
