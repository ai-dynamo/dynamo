#!/usr/bin/env bash
# Apply claims + DGD through the vCluster API. Requires IMAGE=
# (full runtime image URI from Build on Demand).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
NS=${NS:-schwinns-vcluster}
IMAGE=${IMAGE:?set IMAGE to the vllm-runtime URI}
VARIANT=${VARIANT:-failover}  # failover | gms-only | no-gms
DGD_FILE="$ROOT/deploy/dsv4-pro-gms-v0/dgd-${VARIANT}.yaml"
[ -f "$DGD_FILE" ] || { echo "missing $DGD_FILE"; exit 1; }

tmp=$(mktemp)
sed "s|__IMAGE__|$IMAGE|g" "$DGD_FILE" > "$tmp"

vc() { vcluster connect schwinns-vcluster -n schwinns-vcluster -- "$@"; }

vc kubectl -n "$NS" apply -f "$ROOT/deploy/dsv4-pro-gms-v0/claims.yaml"
vc kubectl -n "$NS" apply -f "$tmp"
rm -f "$tmp"
echo "applied $VARIANT with $IMAGE"
