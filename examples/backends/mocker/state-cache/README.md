<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Validate State-Cache Checkpoint Reuse

**Experimental.** This example sends two serial requests through the NVIDIA Dynamo
HTTP frontend and Mocker to verify recurrent-state accounting and partial prefix
reuse. Each request has exactly 24300 input tokens and two output tokens. The
AISimulate engine supplies the scheduler and cache implementation at commit
`c59a00a3fb7d531eea6bea9887235e92b535ec38`.

Use a built and installed checkout of this branch. See the
[development build instructions](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/community/contributing/overview.md)
for prerequisites. Run the commands below from the repository root with that
Python environment active. No GPU, model weights, etcd, or NATS server is required.
Supply a local Kimi K3 tokenizer/config directory through `MODEL_PATH`; Mocker
reads tokenizer/config files without loading weights. The tokenizer must contain
token IDs through 4095 and include its chat template in the tokenizer metadata.

### Prepare Shared Settings

Set `MODEL_PATH` to your tokenizer/config directory. Choose an unused HTTP port.
Create a fresh discovery directory so this example cannot discover other workers:

```bash
export MODEL_PATH=/absolute/path/to/kimi-k3-tokenizer
export MODEL_NAME=state-cache-mocker
export HTTP_PORT=8000
export STATE_CACHE_DEMO_DIR="$(mktemp -d)"
export DYN_FILE_KV="$STATE_CACHE_DEMO_DIR/discovery"

python3 - <<'PY'
import os
import shlex
from pathlib import Path

settings = Path(os.environ["STATE_CACHE_DEMO_DIR"]) / "environment.sh"
names = ("MODEL_PATH", "MODEL_NAME", "HTTP_PORT", "STATE_CACHE_DEMO_DIR", "DYN_FILE_KV")
settings.write_text("".join(f"export {name}={shlex.quote(os.environ[name])}\n" for name in names))
print("Copy this line into each additional terminal:")
print(f"source {shlex.quote(str(settings))}")
PY
```

### Start the Frontend

In the first terminal, start the frontend with file discovery, TCP requests and
responses, and ZeroMQ events:

```bash
python3 -m dynamo.frontend \
  --http-port "$HTTP_PORT" \
  --discovery-backend file \
  --request-plane tcp \
  --response-plane tcp \
  --event-plane zmq \
  --router-mode round-robin
```

### Start the Mocker Worker

In a second terminal, activate the same Python environment and paste the printed
`source` command. Start one aggregated vLLM simulator:

```bash
python3 -m dynamo.mocker \
  --model-path "$MODEL_PATH" \
  --model-name "$MODEL_NAME" \
  --endpoint dyn://statesdemo.mocker.generate \
  --discovery-backend file \
  --request-plane tcp \
  --response-plane tcp \
  --event-plane zmq \
  --router-mode round-robin \
  --engine-type vllm \
  --disaggregation-mode agg \
  --num-workers 1 \
  --num-gpu-blocks-override 512 \
  --block-size 1536 \
  --prefix-match-unit 128 \
  --state-cache '{"bytes_per_request":24576}' \
  --kv-cache-bytes-per-token 16 \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 8192 \
  --no-enable-kv-events
```

`block_size=1536` is the physical KV allocation granularity;
`prefix_match_unit=128` is the prefix matching granularity. `state_cache` describes
one complete recurrent state, currently through `bytes_per_request`. The values
16 bytes per token and 24576 bytes per state are artificial **per-rank** sizing
inputs, not measured Kimi K3 memory usage. The fixed pool has 512 physical blocks,
shared by token KV and recurrent states. This example makes no latency or
throughput claim.

Partial prefix matching does not export KV events. `--no-enable-kv-events`
disables the KV publisher and worker-local KV index, while round-robin routing
keeps requests on the only worker. ZeroMQ remains the general event transport.

### Send the Requests

In a third terminal, activate the same Python environment and paste the printed
`source` command. Check that `/v1/models` lists `state-cache-mocker`, then run the
client:

```bash
curl --fail "http://127.0.0.1:$HTTP_PORT/v1/models"

python3 examples/backends/mocker/state-cache/client.py \
  --base-url "http://127.0.0.1:$HTTP_PORT" \
  --model "$MODEL_NAME" \
  --shared-prefix 24192
```

The client sends explicit token IDs to `/v1/completions`, with `max_tokens=2` and
`ignore_eos=true`. It checks each response's input count, output count, actual
returned token IDs, and `usage.prompt_tokens_details.cached_tokens`. A new nonce
produces a distinct run prefix followed by a repeated synthetic pattern, with
all IDs between 256 and 4095. The second prompt differs at exactly the requested
shared-prefix length. Use `--nonce` only to reproduce a pattern against a fresh
worker; repeating a nonce on a warm worker intentionally fails the cold-hit check.

For a shared prefix of 24192, the cold response reports zero cached tokens and the
warm response reports 24192. The client reports four output tokens and
`recomputed_prefill_tokens=24408`. This value is **inferred from HTTP usage** as
the sum of prompt tokens minus cached tokens; it is not a measurement of committed
forward-pass work.

Run the complete retention matrix against the same worker:

```bash
for shared in 24192 24191 23700 23040 21504 15360 7680 0; do
  python3 examples/backends/mocker/state-cache/client.py \
    --base-url "http://127.0.0.1:$HTTP_PORT" \
    --model "$MODEL_NAME" \
    --shared-prefix "$shared"
done
```

| Shared prefix | Warm request restores | Inferred prefill tokens across both requests | Output tokens |
|---:|---:|---:|---:|
| 24192 | 24192 | 24408 | 4 |
| 24191 | 23040 | 25560 | 4 |
| 23700 | 23040 | 25560 | 4 |
| 23040 | 23040 | 25560 | 4 |
| 21504 | 0 | 48600 | 4 |
| 15360 | 0 | 48600 | 4 |
| 7680 | 0 | 48600 | 4 |
| 0 | 0 | 48600 | 4 |

Cold prefill ends at `7680 / 15360 / 23040 / 24192 / 24300`. The final 108 tokens
after 24192 are still computed before generation starts. After completion, the
retained reusable states are the last full-block checkpoint at 23040 and the
partial-prefix checkpoint at 24192. Earlier chunk ends at 7680 and 15360 are no
longer retained. The physical block boundary at 21504 is not a retained state.
This is chunk-boundary retention, not a configurable periodic checkpoint interval.

For a control run without state cache, stop the worker and restart its command
without `--prefix-match-unit`, `--state-cache`, and `--kv-cache-bytes-per-token`.
Keep the other flags, then add `--legacy` to the client command. A 24192-token
shared prefix then restores 23040 tokens and gives an inferred total of 25560.
The `--legacy` client flag changes the expectation; it does not reconfigure the
server.

### Check Actual Committed Work

The Rust regression submits real requests through Dynamo's public live engine
API and sums every completed forward pass, independently of HTTP usage:

```bash
cargo test -p dynamo-mocker --test state_cache_live --locked -- --nocapture
```

It verifies the prefill step endpoints, retained-prefix behavior, committed
prefill totals, and complete output. The supported mode is manual sizing,
aggregated vLLM, and the G1 state cache. Speculative decoding, KV-event export,
and Belady replay are rejected with explicit `prefix_match_unit`. Native DCP
cache-group layouts, extra checkpoints within a prefill chunk, configurable
retention intervals, shared-prefix junction retention, and asynchronous
overlapped forwards are not modeled by this example.

Stop the worker and frontend with Ctrl-C when finished.
