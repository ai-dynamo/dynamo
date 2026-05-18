#!/bin/bash
# Assert that a registered instance's G2 block layout matches the requested mode.
#
# This is the reproducer for the env-var stripping bug: vLLM's EngineCore
# subprocess does not inherit the parent's environment, so KVBM_BLOCK_LAYOUT
# was silently dropped before the JSON-injection fix landed. Without JSON
# injection the connector always started in Operational mode, even when the
# caller set KVBM_BLOCK_LAYOUT=universal.
#
# Usage:
#   bash verify-block-layout.sh <instance_id> <expected_mode>
#
#   <instance_id>  - UUID returned by the hub's CD instances endpoint
#   <expected_mode> - "operational" or "universal"
#
# Env vars:
#   KVBM_HUB_CONTROL_PORT  (default: 8337)
#
# Exit codes:
#   0  — mode matches expected
#   1  — mode mismatch or describe unavailable (loud error printed)
set -eu

INSTANCE_ID=${1:?"usage: $0 <instance_id> <expected_mode>"}
EXPECTED=${2:?"usage: $0 <instance_id> <expected_mode>"}
CTRL_PORT=${KVBM_HUB_CONTROL_PORT:-8337}

case "$EXPECTED" in
  operational|universal) ;;
  *) echo "expected_mode must be 'operational' or 'universal', got: '$EXPECTED'" >&2; exit 1 ;;
esac

HUB_DESCRIBE_URL="http://127.0.0.1:${CTRL_PORT}/v1/instances/${INSTANCE_ID}/describe"

# Force-fetch a fresh describe from the leader (not the stale cache).
DESCRIBE=$(curl -fsS "${HUB_DESCRIBE_URL}?force=true" 2>/dev/null) || {
  echo "FAIL: could not reach hub describe endpoint: ${HUB_DESCRIBE_URL}" >&2
  exit 1
}

# The hub's GET /v1/instances/{id}/describe wraps the InstanceDescription
# in a control envelope:
#   { "description": <InstanceDescription>, "cached": ..., "age_secs": ..., "source": ... }
# (see lib/kvbm-hub/src/features/control_plane/manager.rs::get_describe).
# Unwrap the envelope first. If `description` is absent we accept the
# top-level object as a fallback — some intermediaries strip the envelope.
#
# Inside the InstanceDescription we look at:
# 1. `config.block_layout` — the KvbmConfig blob the connector pushed.
#    Authoritative source.
# 2. G2 worker layout `block_layout` string — fallback when config blob
#    hasn't been injected yet (pre-stamping snapshot).
ACTUAL=$(python3 -c "
import json, sys

envelope = json.loads(sys.argv[1])
d = envelope.get('description', envelope)

# Preferred: read from config blob (set by connector after init).
cfg = d.get('config')
if cfg and isinstance(cfg, dict) and 'block_layout' in cfg:
    print(cfg['block_layout'])
    sys.exit(0)

# Fallback: infer from any G2 worker layout block_layout string.
# In Universal mode the G2 layout is 'universal'; in Operational it is
# 'operational_nhd' or 'operational_hnd'.
for w in d.get('workers', []):
    for layout in w.get('layouts', []):
        if layout.get('tier') == 'g2':
            bl = layout.get('block_layout', '')
            if bl.startswith('universal'):
                print('universal')
            elif bl.startswith('operational'):
                print('operational')
            sys.exit(0)

# Neither config blob nor G2 layouts available yet (pre-stamping).
print('unknown')
" "$DESCRIBE")

if [ "$ACTUAL" = "$EXPECTED" ]; then
  echo "OK: instance $INSTANCE_ID block_layout=$ACTUAL (expected=$EXPECTED)"
  exit 0
else
  echo "FAIL: instance $INSTANCE_ID block_layout=$ACTUAL (expected=$EXPECTED)" >&2
  echo "      This likely means KVBM_BLOCK_LAYOUT was not injected via JSON into" >&2
  echo "      kv_connector_extra_config.default.block_layout — the env-var path" >&2
  echo "      is stripped by vLLM's EngineCore subprocess spawn." >&2
  exit 1
fi
