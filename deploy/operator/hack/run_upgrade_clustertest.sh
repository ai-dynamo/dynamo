#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Never inherit a user's kubeconfig or reuse a pre-existing cluster for history preparation.
run_dir=$(mktemp -d "${TMPDIR:-/tmp}/dynamo-upgrade.XXXXXXXXXX")
cluster_name=$(basename "$run_dir" | tr '[:upper:].' '[:lower:]-')
export KUBECONFIG="$run_dir/kubeconfig"
export DYNAMO_CLUSTERTEST_CONTEXT="kind-$cluster_name"
kind_bin="${KIND_BIN:-$(pwd)/bin/kind}"
ownership_file="$run_dir/owned"
cleanup() {
    status=$?
    trap - EXIT
    if [[ -f "$ownership_file" ]]; then
        "$kind_bin" delete cluster --name "$cluster_name" || status=1
    fi
    rm -rf "$run_dir"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
# The shared setup records ownership only after refusing any pre-existing cluster.
make setup-clustertest-kind CLUSTERTEST_REUSE=false \
    CLUSTERTEST_KIND_NAME="$cluster_name" CLUSTERTEST_KUBECONFIG="$KUBECONFIG" \
    CLUSTERTEST_OWNERSHIP_FILE="$ownership_file"
make setup-clustertest-crds-in-cluster CLUSTERTEST_KUBECONFIG="$KUBECONFIG"
make "$1" CLUSTERTEST_KUBECONFIG="$KUBECONFIG"
