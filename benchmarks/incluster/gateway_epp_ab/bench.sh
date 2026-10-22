#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Runs inside the aiperf Job (aiperf-job.yaml) for one arm of the gateway A/B
# benchmark. run.sh mounts this script at /bench with three inputs:
#   params.env          the arm, the Gateway URL and the workload
#   targets.tsv         every /metrics endpoint: role, pod, container, URL
#   server-metrics.txt  the endpoints aiperf scrapes during each run
#
# It sends one smoke request through the Gateway, snapshots every /metrics
# endpoint, then runs aiperf once per round and load point, with a snapshot
# after each run. When it is done it writes /results/DONE and waits until
# run.sh has copied /results and written /results/COLLECTED.

set -euo pipefail

set -a
# shellcheck disable=SC1091  # mounted from the ConfigMap at run time
source /bench/params.env
set +a

out=/results/$ARM
mkdir -p "$out/prometheus"

finish() {
  local status=$?
  echo "$status" > "$out/exit-code"
  touch /results/DONE
  echo "bench.sh: finished with status $status; waiting for run.sh to copy /results"
  until [[ -e /results/COLLECTED ]]; do sleep 5; done
  exit "$status"
}
trap finish EXIT

# Fetch every endpoint in targets.tsv into one tar.gz, as <role>/<pod>__<container>.prom.
# An endpoint that fails is recorded as .err instead, with the error.
snapshot() {
  python3 - "$1" /bench/targets.tsv <<'EOF'
import concurrent.futures
import io
import sys
import tarfile
import time
import urllib.request

archive, targets = sys.argv[1], sys.argv[2]
with open(targets) as f:
    rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]


def fetch(row):
    role, pod, container, url = row
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return f"{role}/{pod}__{container}.prom", response.read(), True
    except Exception as e:  # noqa: BLE001 - record any failure and keep going
        return f"{role}/{pod}__{container}.err", f"{url}: {e}\n".encode(), False


ok = 0
with tarfile.open(archive, "w:gz") as tar, concurrent.futures.ThreadPoolExecutor(64) as pool:
    for name, data, fetched in pool.map(fetch, rows):
        ok += fetched
        info = tarfile.TarInfo(name)
        info.size = len(data)
        info.mtime = int(time.time())
        tar.addfile(info, io.BytesIO(data))
print(f"metrics snapshot {archive}: {ok}/{len(rows)} endpoints", flush=True)
EOF
}

# One request through the Gateway, retried until the route and the router are
# ready. A run that cannot reach its arm stops here instead of measuring errors.
python3 - <<'EOF'
import json
import os
import sys
import time
import urllib.error
import urllib.request

url = os.environ["GATEWAY_URL"] + "/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "X-Gateway-Model-Name": os.environ["MODEL"],
    "X-Dynamo-AB-Arm": os.environ["ARM_HEADER"],
}
body = json.dumps(
    {
        "model": os.environ["MODEL"],
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 8,
    }
).encode()
deadline = time.monotonic() + float(os.environ["SMOKE_TIMEOUT_S"])
while True:
    try:
        request = urllib.request.Request(url, data=body, headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            print(f"smoke: HTTP {response.status} {response.read()[:200]!r}", flush=True)
            break
    except urllib.error.HTTPError as e:
        error = f"HTTP {e.code}: {e.read()[:300]!r}"
    except OSError as e:
        error = repr(e)
    if time.monotonic() > deadline:
        sys.exit(f"smoke: no successful request through the Gateway: {error}")
    print(f"smoke: retrying after {error}", flush=True)
    time.sleep(5)
EOF

python3 -c 'import aiperf; print(aiperf.__version__)' > "$out/aiperf-version.txt" || true

IFS=, read -ra rates <<< "$REQUEST_RATES"
IFS=, read -ra concurrencies <<< "$CONCURRENCIES"
points=()
for rate in "${rates[@]}"; do points+=("rps$rate"); done
for concurrency in "${concurrencies[@]}"; do points+=("c$concurrency"); done
if ((${#points[@]} == 0)); then
  echo "bench.sh: set REQUEST_RATES or CONCURRENCIES" >&2
  exit 2
fi

mapfile -t server_metrics_urls < /bench/server-metrics.txt
server_metrics=()
if ((${#server_metrics_urls[@]} > 0)); then
  # shellcheck disable=SC2206  # the formats are a space-separated list
  server_metrics=(--server-metrics "${server_metrics_urls[@]}" --server-metrics-formats $SERVER_METRICS_FORMATS)
fi
prefix=()
if ((PREFIX_LENGTH > 0 && NUM_PREFIXES > 0)); then
  prefix=(--prefix-prompt-length "$PREFIX_LENGTH" --num-prefix-prompts "$NUM_PREFIXES")
fi

snapshot "$out/prometheus/000_baseline.tar.gz"
seq=0
for ((round = 0; round < ROUNDS; round++)); do
  for i in "${!points[@]}"; do
    point=${points[i]}
    # Both arms use the same seed for a round and point, so they send the same
    # prompts on the same schedule. Each run gets a new seed, so no run replays
    # prompts that an earlier run left in the prefix caches.
    seed=$((SEED + round * ${#points[@]} + i))
    if [[ $point == rps* ]]; then
      rate=${point#rps}
      requests=$(python3 -c "print(max(100, round($rate * $DURATION_S)))")
      warmup=$(python3 -c "print(max(10, round($rate * $WARMUP_S)))")
      load=(--request-rate "$rate" --arrival-pattern poisson)
    else
      concurrency=${point#c}
      requests=$((concurrency * 10 > 500 ? concurrency * 10 : 500))
      warmup=$((concurrency * 2))
      load=(--concurrency "$concurrency")
    fi
    run=${point}_r${round}
    dir=$out/$run
    mkdir -p "$dir"
    cmd=(
      aiperf profile
      --model "$MODEL"
      --tokenizer "$MODEL"
      --url "$GATEWAY_URL"
      --endpoint-type chat
      --streaming
      --header "X-Gateway-Model-Name:$MODEL"
      --header "X-Dynamo-AB-Arm:$ARM_HEADER"
      "${load[@]}"
      --request-count "$requests"
      --warmup-request-count "$warmup"
      # One dataset entry per request: aiperf reuses entries, and a reused
      # prompt would hit the prefix cache as a whole.
      --num-dataset-entries $((requests + warmup))
      --synthetic-input-tokens-mean $((ISL - (${#prefix[@]} > 0 ? PREFIX_LENGTH : 0)))
      --synthetic-input-tokens-stddev 0
      "${prefix[@]}"
      --output-tokens-mean "$OSL"
      --output-tokens-stddev 0
      --extra-inputs ignore_eos:true
      # The mocker emits random token ids, which do not survive re-tokenizing.
      --use-server-token-count
      --random-seed "$seed"
      --no-gpu-telemetry
      "${server_metrics[@]}"
      --artifact-dir "$dir"
      --ui none
    )
    printf '%q ' "${cmd[@]}" > "$dir/command.txt"
    echo >> "$dir/command.txt"
    echo "=== $ARM $run (seed $seed) started $(date -u +%FT%TZ)"
    status=0
    "${cmd[@]}" > "$dir/console.log" 2>&1 || status=$?
    tail -n 40 "$dir/console.log"
    printf '%s\t%s\n' "$run" "$status" >> "$out/runs.tsv"
    seq=$((seq + 1))
    snapshot "$out/prometheus/$(printf '%03d' "$seq")_$run.tar.gz"
    sleep "$COOLDOWN_S"
  done
done
