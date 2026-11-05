<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Image encoder disaggregation

These experimental examples require matching TensorRT-LLM OpenEngine encoder-handoff and
Dynamo sidecar builds. Older runtimes without the advertised encoder capabilities
cannot run these topologies. Image handoffs support TP/PP/CP=1; video and audio
encoder handoffs are not supported.

| Topology | Encoder output consumer | KV transfer |
| --- | --- | --- |
| E+PD | Combined prefill/decode worker | No inter-worker KV transfer |
| E+P+D | Prefill worker, followed by a separate decode worker | Native P-to-D NIXL transfer |

The encoder fetches and processes media, then exports embeddings and model-specific
inputs. The consuming worker has its vision encoder disabled. OpenEngine gRPC carries
references and control metadata; NIXL transfers encoder outputs directly between the
engine processes. In E+P+D, P consumes the encoder reference once and returns the KV
session with decode-preparation metadata. D receives that KV session and does not
replay the encoder reference.

Encoder references are limited to 64 KiB. E+P+D also requires portable decode
state, limited separately to 2 MiB; the engine rejects a context request if that
state cannot be produced within the limit. The state grows with the expanded
prompt length.

## Single-host launch

Use the TensorRT-LLM environment for `TRTLLM_PYTHON` and the Dynamo environment for
the frontend and sidecars. Set `MODEL_REVISION` to the checkpoint's immutable commit:

```bash
export MODEL=Qwen/Qwen3-VL-2B-Instruct
export MODEL_REVISION=89644892e4d85e24eaac8bacfd4f463576704203
export TRTLLM_PYTHON=/path/to/trtllm/bin/python3
export DYN_DISCOVERY_BACKEND=file
```

For E+PD on two GPUs:

```bash
ENCODER_GPU=0 PD_GPU=1 bash lib/sidecar/trtllm/launch/disagg_multimodal_e_pd.sh
```

For E+P+D on three GPUs:

```bash
ENCODER_GPU=0 PREFILL_GPU=1 DECODE_GPU=2 \
  bash lib/sidecar/trtllm/launch/disagg_multimodal_e_p_d.sh
```

The E+P+D launcher enables the native KV transceiver on P and D and starts the
frontend with `--no-router-conditional-disagg`. D requires the P-produced KV
handoff; skipping P would leave only an already-consumed encoder reference.
Use `--help` for port and device overrides. Managed launches use numbered
`DYN_ENGINE_GRPC_PORT1/2/3` and `DYN_SYSTEM_PORT1/2/3` for E, P/PD, and D.
Ordinary text requests bypass the encoder.

## Cross-host placement

The scripts above launch every process locally. For separate hosts, launch each
engine and sidecar on its assigned host with shared Dynamo discovery and a
consistent served model name. Give the engines the same immutable model revision.
For E+P+D, start the frontend with `--no-router-conditional-disagg` on every host
that runs a frontend.

- E: `mm_encoder_only: true`, a sidecar with `--disaggregation-mode encode`, and
  `TRTLLM_OPENENGINE_ENCODER_ENDPOINT` set to its reachable control address.
- PD: `disable_mm_encoder: true` and an aggregated sidecar with `--route-to-encoder`.
- Separate P: `disable_mm_encoder: true`, native NIXL KV transfer, and a sidecar
  with `--disaggregation-mode prefill --route-to-encoder`.
- Separate D: `disable_mm_encoder: true`, the same native KV transfer backend,
  and a sidecar with `--disaggregation-mode decode`.

Set `TRTLLM_OPENENGINE_ENCODER_TRANSFER=nixl` and
`TRTLLM_OPENENGINE_ENCODER_MODEL_REVISION` to the immutable revision on E and its
consumers. Encoder control and native transfer endpoints must be reachable across
hosts. A shared model-cache filesystem is optional; every engine must have the
matching checkpoint locally.
