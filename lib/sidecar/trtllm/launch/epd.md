<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Image encoder disaggregation

These experimental examples require matching TensorRT-LLM OpenEngine encoder-handoff and
Dynamo sidecar builds. Older runtimes without the advertised encoder capabilities
cannot run these topologies. Image handoffs support TP/PP/CP=1; video and audio
encoder handoffs are not supported.

Dynamo supports native NIXL encoder handoff only. Producers must advertise NIXL
as their single encoder transfer backend and typed abort cleanup support.
Consumers must accept NIXL and may advertise additional backends. Shared-safetensors handoff
remains an upstream helper for local experiments and is outside this Dynamo
integration. Use matching proposal builds with encoder-result release support;
the baseline OpenEngine package pins do not supply these capabilities.

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
the frontend and sidecars. Set `MODEL_REVISION` to the checkpoint's full immutable
40-character commit SHA. The launcher resolves that snapshot once through
`huggingface_hub` in the TensorRT-LLM environment, then passes its local path to
every engine and sidecar. This keeps weights, tokenizer, configuration and chat
template on one revision. The public served model name remains `MODEL`.

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

The E+P+D launcher enables the native KV transceiver on P and D. The router
keeps image encoder handoffs on the mandatory prefill path because D requires
P-produced KV. Conditional prefill bypass remains available for text requests.
Use `--help` for port and device overrides. Managed launches use numbered
`DYN_ENGINE_GRPC_PORT1/2/3` and `DYN_SYSTEM_PORT1/2/3` for E, P/PD, and D.
Ordinary text requests bypass the encoder.

When `CUDA_VISIBLE_DEVICES` is inherited, every role defaults to its first entry,
including GPU UUIDs and MIG device IDs. This keeps a managed single-GPU launch
inside its allocation. Use `ENCODER_GPU`, `PD_GPU`, `PREFILL_GPU` and `DECODE_GPU`
to assign separate devices from an allocation; these are physical CUDA device
identifiers, not indices into the inherited list. Without an inherited allocation,
E and PD default to device 0; separate P and D default to 1 and 2.

`ENCODER_BIND_HOST` selects the encoder engine's interface. Its local sidecar
connects to the same host by default; wildcard binds `0.0.0.0` and `::` map to
loopback destinations `127.0.0.1` and `::1`. Set `ENCODER_CONNECT_HOST` when the
sidecar needs a different destination. `ENCODER_ADVERTISE_HOST` is the control
address used by consumers and defaults to that connect host; set a reachable
address explicitly when consumers run on another host.

## Cross-host placement

The scripts above launch every process locally. For separate hosts, launch each
engine and sidecar on its assigned host with shared Dynamo discovery and a
consistent served model name. Resolve the same immutable model revision on each
host and pass the resulting local path to both engine and sidecar. Set the
engine's `--served_model_name` to the common served name so host-specific cache
paths do not change model discovery.
For E+P+D, the committed prefill capability keeps encoder handoffs on P while
allowing text requests to use conditional prefill bypass.

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

## Upstream-nightly CI

The shared TRT-LLM media workflow runs `image-epd` (E+PD) and
`image-epd-disaggregated` (E+P+D) against the latest published official nightly.
Each job runs its engine roles on one GPU, sequentially after an aggregated
baseline. Two opposite-color images must produce the expected captions and match
the baseline's output tokens and expanded usage. The tests also check encoder
capabilities, single-use references, and KV-only decode continuation.

These tests exercise the Rust sidecar adapters and native engines directly;
Dynamo HTTP routing and cross-host transfers require separate validation.
Missing runtime capabilities fail the job rather than skipping it. The EPD
prerequisites must be published in the upstream nightly before these jobs can pass.
