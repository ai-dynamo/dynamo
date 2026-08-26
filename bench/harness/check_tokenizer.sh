#!/usr/bin/env bash
# Verify the tokenizer aiperf will use agrees with the server's.
#
# aiperf uses its tokenizer for two things that both silently corrupt results if
# it is wrong: generating synthetic prompts of a requested token length, and
# counting tokens client-side. A tokenizer that merely *loads* proves nothing --
# the wrong one loads perfectly well and reports plausible numbers that do not
# describe what the engine actually received.
#
# The check: encode a string locally, send the identical string to the server,
# and compare against the prompt_tokens the server reports from its own
# tokenizer. Agreement means aiperf's counts describe reality.
set -uo pipefail
MODEL=${MODEL:?set MODEL}
URL=${URL:?set URL}
TOKENIZER=${TOKENIZER:?set TOKENIZER}

# Ensure the transformers version here too, not just in aiperf_load.sh. Load
# teardown recreates the bench pod, which wipes every pip install, so any step
# that assumes a previously-installed version will fail the first time it runs
# after a teardown -- which is every run.
TRANSFORMERS_MIN=${TRANSFORMERS_MIN:-5.15.1}
tv=$(python3 -c "import transformers;print(transformers.__version__)" 2>/dev/null)
if [ "$tv" != "$TRANSFORMERS_MIN" ]; then
  pip install -q --disable-pip-version-check --upgrade "transformers==$TRANSFORMERS_MIN" >/dev/null 2>&1
  tv=$(python3 -c "import transformers;print(transformers.__version__)" 2>/dev/null)
fi
echo "  transformers: $tv"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python3 - "$MODEL" "$URL" "$TOKENIZER" <<'PY'
import json, sys, urllib.request
model, url, tok_id = sys.argv[1], sys.argv[2], sys.argv[3]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(tok_id)
print(f"  tokenizer: {type(tok).__name__} from {tok_id}")

cases = [
    "the quick brown fox jumps over the lazy dog",
    "Hello, world! 123",
    " ".join(["token"] * 200),
    "def f(x):\n    return x ** 2  # squared\n",
]
ok = True
for s in cases:
    local = len(tok.encode(s))
    body = json.dumps({"model": model, "prompt": s, "max_tokens": 1,
                       "temperature": 0, "stream": False}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=120))
        server = r.get("usage", {}).get("prompt_tokens")
    except Exception as e:
        print(f"  request failed: {type(e).__name__} {str(e)[:120]}")
        ok = False
        continue
    mark = "OK  " if local == server else "MISMATCH"
    if local != server:
        ok = False
    print(f"  {mark} local={local:<5} server={server:<5} {repr(s)[:46]}")

print("  RESULT: tokenizers agree" if ok else
      "  RESULT: *** TOKENIZERS DISAGREE — aiperf ISL/OSL would be wrong ***")
sys.exit(0 if ok else 1)
PY
