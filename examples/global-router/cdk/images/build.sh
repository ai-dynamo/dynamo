#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: bash images/build.sh DYNAMO_CHECKOUT [IMAGE_PREFIX] [TAG]" >&2
  exit 2
fi
gr_source=$(cd "$1" && pwd)
gr_prefix=${2:-global-router-dev}
gr_revision=$(git -C "$gr_source" rev-parse HEAD)
gr_tag=${3:-${gr_revision:0:12}}
gr_dockerfile=$(cd "$(dirname "$0")" && pwd)/Dockerfile
gr_assets=$(dirname "$gr_dockerfile")/assets
mkdir -p "$gr_assets"
if [[ ! -f "$gr_assets/swagger-ui-v5.33.1.zip" ]]; then
  curl --fail --location --retry 2 \
    -o "$gr_assets/swagger-ui-v5.33.1.zip" \
    https://github.com/swagger-api/swagger-ui/archive/refs/tags/v5.33.1.zip
fi
python3 - "$gr_assets/swagger-ui-v5.33.1.zip" <<'PY'
import hashlib, pathlib, sys
assert hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest() == "ce6c41f5ec2310abccbaa1a18bc3aa3c5f7d401d5b0e7c4f9231ae222b956b12", "Swagger UI archive checksum mismatch"
PY
git -C "$gr_source" diff --exit-code --quiet
git -C "$gr_source" diff --cached --exit-code --quiet
for gr_component in global-discovery-service global-frontend pool-relay; do
  docker buildx build --platform linux/amd64 --load \
    --build-context "build-assets=$gr_assets" \
    --target "$gr_component" --build-arg SOURCE_REVISION="$gr_revision" \
    -f "$gr_dockerfile" -t "$gr_prefix/$gr_component:$gr_tag" "$gr_source"
done
docker run --rm --platform linux/amd64 "$gr_prefix/global-discovery-service:$gr_tag" --help
docker run --rm --platform linux/amd64 "$gr_prefix/global-frontend:$gr_tag" --help
docker run --rm --platform linux/amd64 --entrypoint python3 "$gr_prefix/pool-relay:$gr_tag" \
  -c 'from dynamo.llm import PoolRelay; import dynamo.pool_relay.__main__; print("PoolRelay imports OK")'
