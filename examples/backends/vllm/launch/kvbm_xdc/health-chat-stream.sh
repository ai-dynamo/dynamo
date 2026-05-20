#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$SCRIPT_DIR/hardware-profiles.sh"
kvbm_xdc_apply_hardware_profile model-only

URL=${URL:?URL is required, for example http://127.0.0.1:8001}
: "${MODEL:?MODEL must be set by KVBM_HARDWARE_PROFILE or env override}"

curl -fsS "$URL/health" >/dev/null
curl -fsS "$URL/v1/models" >/dev/null
python - "$URL" "$MODEL" <<'PY'
import json
import sys
import urllib.request

url, model = sys.argv[1], sys.argv[2]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Return the single word ready."}],
    "max_tokens": 8,
    "temperature": 0,
    "stream": True,
}
req = urllib.request.Request(
    url.rstrip("/") + "/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"content-type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=120) as resp:
    first = resp.readline()
    if not first:
        raise SystemExit("no streaming response")
print("streaming chat OK")
PY
