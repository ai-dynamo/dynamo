<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# WEKA lifecycle correctness replay

This experiment selects two real root requests and one completed 11-turn subagent from `semianalysisai/cc-traces-weka-with-subagents-052726-256k`. It preserves input lengths, `hash_ids`, session IDs, and parent/child structure while reducing outputs and delays. Every reconstructed prompt fits within Qwen3-0.6B's 40,960-token context. It is a lifecycle and KV-control correctness fixture, not a performance sample.

## Build the fixture

```bash
python experiments/continuum-kv-hints/weka-correctness/build_fixture.py \
  --source /tmp/weka-traces.jsonl \
  --output-dir /tmp/continuum-weka-correctness/fixture
```

## Verify AIPerf lifecycle headers

Start the capture endpoint:

```bash
python experiments/continuum-kv-hints/weka-correctness/capture_server.py \
  --output /tmp/continuum-weka-correctness/captured-requests.jsonl
```

Run AIPerf from the `karenc/continuum-lifecycle-annotations` branch:

```bash
QWEN_TOKENIZER_DIR=/path/to/Qwen3-0.6B/snapshot

AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true \
  aiperf profile \
  --scenario inferencex-agentx-mvp \
  --model Qwen/Qwen3-0.6B \
  --url http://127.0.0.1:8189 \
  --endpoint /v1/chat/completions \
  --endpoint-type chat \
  --streaming \
  --custom-dataset-type weka_trace \
  --input-file /tmp/continuum-weka-correctness/fixture/trace.json \
  --num-dataset-entries 1 \
  --concurrency 1 \
  --request-count 13 \
  --benchmark-duration 60 \
  --stats-interval 30 \
  --random-seed 42 \
  --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 \
  --trajectory-start-max-ratio 0.75 \
  --use-server-token-count \
  --no-gpu-telemetry \
  --tokenizer "$QWEN_TOKENIZER_DIR" \
  --tokenizer-trust-remote-code \
  --slice-duration 1.0 \
  --unsafe-override \
  --output-artifact-dir /tmp/continuum-weka-correctness/aiperf
```

The command keeps the fixed InferenceX protocol and replay flags. It intentionally scales concurrency, duration, and dataset count down for a one-trajectory correctness run. The local AIPerf branch does not expose the SemiAnalysis fork's `--warmup-requests-per-lane` flag. The fixture timestamps are already compressed, so this run does not use either `--ignore-trace-delays` or `--trace-idle-gap-cap-seconds`.

Validate the captured lifecycle contract:

```bash
python experiments/continuum-kv-hints/weka-correctness/check_headers.py \
  /tmp/continuum-weka-correctness/captured-requests.jsonl
```

The next stage replaces the capture endpoint with the networked Dynamo/vLLM stack and checks SessionPrefixIndexer lineage plus deferred `kv.retain` and `kv.evict` actions.

## Run through Dynamo and vLLM

`run_networked.sh` launches the experiment branches, enables G1 KV events and SessionPrefixIndexer, and runs the same fixture through the public Dynamo endpoint. The fixed-TTL policy emits deferred retention for continuing sessions and deferred eviction for final sessions. Its log records the number of lineage blocks resolved from SessionPrefixIndexer for every hint.

```bash
bash experiments/continuum-kv-hints/weka-correctness/run_networked.sh
```

Results are written under `/tmp/continuum-weka-correctness/networked` by default.

To run the same replay against the pinned container instead of source worktrees:

```bash
CONTINUUM_IMAGE=nvcr.io/nvidian/dynamo-dev/karenc:dynamo-kv-hints-6d7cf575cb-vllm-4091050295 \
  bash experiments/continuum-kv-hints/weka-correctness/run_container.sh
```

Container results are written under `/tmp/continuum-weka-correctness/container-networked` by default.
