#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Exercise both gRPC listeners and graceful shutdown without a Dynamo cluster.
set -euo pipefail

image="${1:-kv-dc-relay-poc:local}"
container="kv-dc-relay-poc-smoke-$$"
fixture_dir="$(mktemp -d)"
cleanup() {
    docker rm -f "$container" >/dev/null 2>&1 || true
    rm -rf "$fixture_dir"
}
trap cleanup EXIT

cat > "$fixture_dir/config.json" <<'JSON'
{
  "dc_id": "relay-container-smoke",
  "runtime_namespace": "relay-container-smoke",
  "wan_listen": "0.0.0.0:50051",
  "stats_listen": "0.0.0.0:50052",
  "stats_allow_non_loopback": true
}
JSON
chmod 0644 "$fixture_dir/config.json"

docker run -d --name "$container" \
    -p 127.0.0.1::50051 -p 127.0.0.1::50052 \
    --env DYN_DISCOVERY_BACKEND=mem \
    --env DYN_REQUEST_PLANE=tcp \
    --env DYN_EVENT_PLANE=zmq \
    --mount "type=bind,source=$fixture_dir/config.json,target=/etc/kv-dc-relay/config.json,readonly" \
    "$image" /etc/kv-dc-relay/config.json >/dev/null
wan_binding="$(docker port "$container" 50051/tcp)"
stats_binding="$(docker port "$container" 50052/tcp)"
wan_port="${wan_binding##*:}"
stats_port="${stats_binding##*:}"

ready=false
for _ in {1..80}; do
    if python3 - "$wan_port" "$stats_port" <<'PY'
import socket
import sys
try:
    for port in map(int, sys.argv[1:]):
        with socket.create_connection(('127.0.0.1', port), timeout=0.2):
            pass
except OSError:
    sys.exit(1)
PY
    then
        ready=true
        break
    fi
    if [[ "$(docker inspect --format '{{.State.Running}}' "$container")" != true ]]; then
        break
    fi
    sleep 0.25
done
if [[ "$ready" != true ]]; then
    docker logs "$container" >&2
    echo "Relay listeners did not start" >&2
    exit 1
fi

# GetRelayInfo requires the relay v1 contract marker (fixed32 field 127).
python3 - "$fixture_dir/wan-request.bin" "$fixture_dir/stats-request.bin" <<'PY'
from pathlib import Path
import sys
payload = bytes.fromhex('fd 07 31 52 56 4b')
Path(sys.argv[1]).write_bytes(b'\x00' + len(payload).to_bytes(4, 'big') + payload)
Path(sys.argv[2]).write_bytes(b'\x00' * 5)  # google.protobuf.Empty
PY
curl --silent --show-error --http2-prior-knowledge --max-time 5 \
    --header 'content-type: application/grpc' --header 'te: trailers' \
    --data-binary "@$fixture_dir/wan-request.bin" \
    --dump-header "$fixture_dir/wan-headers" --output "$fixture_dir/wan-response.bin" \
    "http://127.0.0.1:$wan_port/dynamo.kvrelay.v1.KvEventRelay/GetRelayInfo"
python3 - "$fixture_dir/wan-response.bin" <<'PY'
from pathlib import Path
import sys
reply = Path(sys.argv[1]).read_bytes()
assert len(reply) >= 7 and reply[0] == 0, 'missing RelayInfo gRPC frame'
assert int.from_bytes(reply[1:5], 'big') == len(reply) - 5, 'invalid RelayInfo frame length'
assert reply[5:7] == b'\x08\x01', 'wrong Relay protocol version'
PY
rg -qi '^grpc-status: 0' "$fixture_dir/wan-headers"
echo 'GetRelayInfo: gRPC OK'

# WatchLoad is a stream. Its initial empty snapshot must arrive immediately.
set +e
curl --silent --http2-prior-knowledge --max-time 2 \
    --header 'content-type: application/grpc' --header 'te: trailers' \
    --data-binary "@$fixture_dir/stats-request.bin" \
    --dump-header "$fixture_dir/stats-headers" --output "$fixture_dir/stats-response.bin" \
    "http://127.0.0.1:$stats_port/dynamo.kvdc.relay.v1.KvDcRelay/WatchLoad"
curl_result=$?
set -e
if [[ "$curl_result" != 0 && "$curl_result" != 28 ]]; then
    docker logs "$container" >&2
    echo "WatchLoad request failed with curl exit $curl_result" >&2
    exit 1
fi
python3 - "$fixture_dir/stats-response.bin" <<'PY'
from pathlib import Path
import sys
reply = Path(sys.argv[1]).read_bytes()
assert len(reply) > 5 and reply[0] == 0, 'missing initial WatchLoad gRPC frame'
assert int.from_bytes(reply[1:5], 'big') <= len(reply) - 5, 'invalid WatchLoad frame length'
PY
rg -qi '^content-type: application/grpc' "$fixture_dir/stats-headers"
echo 'WatchLoad: initial gRPC snapshot received'

docker stop --timeout 5 "$container" >/dev/null
exit_code="$(docker inspect --format '{{.State.ExitCode}}' "$container")"
if [[ "$exit_code" != 0 ]]; then
    docker logs "$container" >&2
    echo "Relay exited with $exit_code after SIGTERM" >&2
    exit 1
fi
echo 'SIGTERM: clean exit 0'
