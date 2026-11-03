#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Builds the campaign's binaries in parallel on the keeper's nodes, from the login node:
# the Global Frontend from $CAMPAIGN/src/main on node A and from $CAMPAIGN/src/pr on node B,
# and the mock pool on node C. Fails if the two frontends are the same binary.
set -euo pipefail
source "$(dirname "$0")/env.sh"

job=$(cat "$CAMPAIGN/control/job")
mapfile -t nodes < "$CAMPAIGN/control/nodes.txt"
build() {
    local host=$1
    shift
    srun --jobid="$job" --overlap -N1 -n1 -w "$host" --cpus-per-task=144 --mem=0 --cpu-bind=none \
        -J gfs-build bash "$BENCH/slurm/build.sh" "$@"
}
build "${nodes[0]}" frontend main "$CAMPAIGN/src/main" &
main=$!
build "${nodes[1]}" frontend pr "$CAMPAIGN/src/pr" &
pr=$!
build "${nodes[2]}" mock &
mock=$!
wait $main
wait $pr
wait $mock
if [[ $(cut -d' ' -f1 "$CAMPAIGN/builds/main/sha256") == $(cut -d' ' -f1 "$CAMPAIGN/builds/pr/sha256") ]]; then
    echo "main and pr built the same binary" >&2
    exit 1
fi
echo "builds: main $(cat "$CAMPAIGN/builds/main/source-sha"), pr $(cat "$CAMPAIGN/builds/pr/source-sha")"
