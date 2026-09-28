#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Exercise the packaged binary and its HTTP listener without a Relay deployment.
set -euo pipefail

image="${1:-global-router-poc:local}"
container="global-router-poc-smoke-$$"
fixture_dir="$(mktemp -d)"
cleanup() {
    docker rm -f "$container" >/dev/null 2>&1 || true
    rm -rf "$fixture_dir"
}
trap cleanup EXIT

cat > "$fixture_dir/config.json" <<'JSON'
{
  "listen": "0.0.0.0:8080",
  "freshness": {
    "catalog_max_age_ms": 45000,
    "readiness_max_age_ms": 45000,
    "capacity_max_age_ms": 30000,
    "load_max_age_ms": 30000,
    "kv_usage_max_age_ms": 30000,
    "kv_overlap_max_age_ms": 30000
  },
  "overlap_max_age_ms": 30000,
  "pools": [
    {
      "site_id": "ohio",
      "namespace": "dynamo",
      "dgd_name": "mocker-ohio-1",
      "region": "us-east-2",
      "runtime_namespace": "dynamo-mocker-ohio-1",
      "model": "mocker",
      "private_frontend_base_url": "http://127.0.0.1:18000",
      "relay_grpc_url": "http://127.0.0.1:19000",
      "stats_grpc_url": "http://127.0.0.1:19001",
      "subscriber_id": "container-smoke-ohio"
    },
    {
      "site_id": "west",
      "namespace": "dynamo",
      "dgd_name": "mocker-west-1",
      "region": "us-west-2",
      "runtime_namespace": "dynamo-mocker-west-1",
      "model": "mocker",
      "private_frontend_base_url": "http://127.0.0.1:18001",
      "relay_grpc_url": "http://127.0.0.1:19010",
      "stats_grpc_url": "http://127.0.0.1:19011",
      "subscriber_id": "container-smoke-west"
    }
  ]
}
JSON
chmod 0644 "$fixture_dir/config.json"

docker run -d --name "$container" -p 127.0.0.1::8080 \
    --mount "type=bind,source=$fixture_dir/config.json,target=/etc/global-router/config.json,readonly" \
    "$image" /etc/global-router/config.json >/dev/null
port="$(docker port "$container" 8080/tcp)"
base_url="http://127.0.0.1:${port##*:}"
started=false
for _ in {1..30}; do
    if curl --silent --max-time 2 --output /dev/null "$base_url/pools"; then
        started=true
        break
    fi
    sleep 0.2
done
if [[ "$started" != true ]]; then
    docker logs "$container" >&2
    echo "Router did not start" >&2
    exit 1
fi

check_status() {
    local expected="$1" method="$2" path="$3" body="${4:-}"
    local actual
    if [[ "$method" == POST ]]; then
        actual="$(curl --silent --show-error --max-time 5 --output /dev/null --write-out '%{http_code}' \
            --header 'content-type: application/json' --data "$body" "$base_url$path")"
    else
        actual="$(curl --silent --show-error --max-time 5 --output /dev/null --write-out '%{http_code}' "$base_url$path")"
    fi
    if [[ "$actual" != "$expected" ]]; then
        echo "$method $path returned $actual; expected $expected" >&2
        docker logs "$container" >&2
        exit 1
    fi
    echo "$method $path: $actual"
}

check_status 200 GET /pools
check_status 200 POST /overlap_scores '{"model":"mocker","token_ids":[1,2]}'
check_status 503 GET /readyz
check_status 503 POST /v1/completions '{"model":"mocker","prompt":"hello"}'

docker stop --timeout 5 "$container" >/dev/null
exit_code="$(docker inspect "$container" --format '{{.State.ExitCode}}')"
if [[ "$exit_code" != 0 ]]; then
    echo "Router exited with $exit_code after SIGTERM; expected 0" >&2
    exit 1
fi
echo 'SIGTERM: clean exit (0)'
