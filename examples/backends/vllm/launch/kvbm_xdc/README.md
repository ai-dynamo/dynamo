# KVBM Cross-Datacenter Harness

This directory contains topology-oriented run harnesses for KVBM
cross-datacenter experiments. Experiment letters are metadata, not script
names, and hardware assumptions live in profile/env variables rather than in
workflow names.

## Fixed Experiment Contract

| Field | Value |
|---|---|
| Model | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` |
| Framework | Dynamo vLLM: `python -m dynamo.vllm` |
| User endpoint | `dynamo.frontend` OpenAI `/v1/chat/completions` |
| Endpoint type | `chat` |
| Streaming | `true` |
| Recorded GPU class | H100 or A100 only |

Do not use `python -m vllm.entrypoints.openai.api_server` for the primary
Experiment E path. That is a raw vLLM diagnostic or legacy hub-CD smoke, not
the Dynamo vLLM framework under test.

## Hardware Profiles

Hardware, model, sizing, and placement defaults live in
`hardware-profiles.sh`. Workflow scripts source that file and fail early if the
selected profile or caller-provided env does not define the values they need.

| Profile | Purpose | Model | Placement |
|---|---|---|---|
| `h100-a100` | Recorded Experiment E/F runs | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | decode GPU 0, prefill GPU 1 |
| `spark-gb10` | Local smoke/debug only | `Qwen/Qwen3-0.6B` | decode and prefill share GPU 0 |
| `custom` | Any other cluster allocation | caller-set | caller-set |

Use `KVBM_HARDWARE_PROFILE=h100-a100` or a fully specified `custom` profile for
results that feed Experiment E. `spark-gb10` keeps the workflow debuggable on
Spark, but those outputs are not comparable experiment data.

For a non-Spark cluster, keep the workflow script the same and provide the
profile explicitly:

```bash
KVBM_HARDWARE_PROFILE=custom \
MODEL=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
GPU_CLASS=H100 \
GPU_MEMORY_UTILIZATION=0.70 \
MAX_MODEL_LEN=2048 \
MAX_NUM_SEQS=8 \
CPU_CACHE_GB=16 \
DECODE_CUDA_VISIBLE_DEVICES=0 \
PREFILL_CUDA_VISIBLE_DEVICES=1 \
ARTIFACT_DIR=/tmp/kvbm-xdc/same-node \
bash run-dynamo-kvbm-chat-smoke.sh
```

## Experiment Map

| Experiment | Primary question | Topology | Script |
|---|---|---|---|
| E baseline | What is TTFT on the same-node KVBM/NIXL path for this model? | `dynamo.frontend` + decode `dynamo.vllm` using `NixlConnector` + prefill `dynamo.vllm` using `PdConnector(DynamoConnector,NixlConnector)` | `run-dynamo-kvbm-chat-smoke.sh` with decode and prefill on the same H100/A100 node |
| E cross-DC | What changes when the same model spans datacenters? | Same as baseline, with shared NATS/etcd and explicit routable NIXL side-channel hosts | `launch-dynamo-frontend.sh`, `launch-kvbm-vllm.sh`, then `run-dynamo-kvbm-chat-smoke.sh` in `NODE_ROLE=client` mode |
| F | Does the hub-controlled P2P G2 transfer path work and what trace does it produce? | `kvbm_hub` P2P transfer primitives between two KVBM instances | `.claude/skills/p2p-smoke/p2p-smoke.sh` |

Raw hub conditional-disagg scripts are diagnostics, not this Dynamo-native
harness. Keep them separate so Experiment E stays on `dynamo.frontend` and
`python -m dynamo.vllm`.

## Usage

Same-node H100/A100 baseline:

```bash
ARTIFACT_DIR=/tmp/kvbm-xdc/same-node \
KVBM_HARDWARE_PROFILE=h100-a100 \
NODE_ROLE=all \
bash run-dynamo-kvbm-chat-smoke.sh
```

Cross-datacenter runs use the same workflow split across roles. Set
`ETCD_ENDPOINTS`, `NATS_SERVER`, and each worker's routable
`VLLM_NIXL_SIDE_CHANNEL_HOST` through `DECODE_SIDE_CHANNEL_HOST` or
`PREFILL_SIDE_CHANNEL_HOST`.

```bash
# Frontend / runtime-infra node
ARTIFACT_DIR=/tmp/kvbm-xdc/frontend NODE_ROLE=frontend bash run-dynamo-kvbm-chat-smoke.sh

# Decode node
ARTIFACT_DIR=/tmp/kvbm-xdc/decode NODE_ROLE=decode DECODE_SIDE_CHANNEL_HOST=192.0.2.21 bash run-dynamo-kvbm-chat-smoke.sh

# Prefill node
ARTIFACT_DIR=/tmp/kvbm-xdc/prefill NODE_ROLE=prefill PREFILL_SIDE_CHANNEL_HOST=192.0.2.31 bash run-dynamo-kvbm-chat-smoke.sh

# Client / measurement node
ARTIFACT_DIR=/tmp/kvbm-xdc/client NODE_ROLE=client FRONTEND_HOST=192.0.2.10 bash run-dynamo-kvbm-chat-smoke.sh
```

Locked AIPerf pass against the chat endpoint:

```bash
URL=http://192.0.2.10:8000 \
ARTIFACT_DIR=/tmp/kvbm-xdc/aiperf \
KVBM_HARDWARE_PROFILE=h100-a100 \
bash run-aiperf-locked.sh
```

## Runtime Invariants

- Cross-node runs need shared runtime infra reachable from all nodes:
  `ETCD_ENDPOINTS=192.0.2.10:2379`, `NATS_SERVER=nats://192.0.2.10:4222`,
  `DYN_DISCOVERY_BACKEND=etcd`, `DYN_REQUEST_PLANE=tcp`, and
  `DYN_EVENT_PLANE=nats`.
- Set `VLLM_NIXL_SIDE_CHANNEL_HOST` explicitly on each worker to that node's
  routable address. Do not rely on autodetection for cross-datacenter runs.
- Prefill uses `PdConnector` wrapping KVBM and NIXL. Worker startup validates
  the canonical KVBM module exports and the NIXL registry entry before launching
  `python -m dynamo.vllm`.
- Use `kvbm.v2.vllm.connector` for the `PdConnector` module path. The legacy
  `kvbm.vllm_integration.connector` path remains a compatibility shim for
  existing examples and older configs.
- Capture `chat-ttft.json`, logs, metadata, and `trace.html` for every run.
- Keep machine-specific placement in `KVBM_HARDWARE_PROFILE` or explicit env
  overrides, not in workflow script names.
- If startup takes the compile-heavy vLLM path, use:
  `VLLM_RUNNER=generate`, `VLLM_USE_AOT_COMPILE=0`,
  `VLLM_USE_STANDALONE_COMPILE=0`, and
  `VLLM_ENABLE_V1_MULTIPROCESSING=0`.
