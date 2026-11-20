<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Image encoder disaggregation

These experimental examples require matching TensorRT-LLM OpenEngine and
Dynamo sidecar builds implementing unary Encode, encoded-media references and
explicit release. The engine schema release must match `../proto/REVISION`; older
runtimes cannot run these topologies. Image handoffs support TP/PP/CP=1; video and audio
encoder handoffs are not supported.

These topologies do not support Dynamo image-aware KV routing. Native prefix-cache
reuse can still occur, but it does not establish image-block overlap for Dynamo
worker selection. Integrating encoder-handoff identity with routing is separate work.

Dynamo forwards the framework's opaque reference bytes unchanged. The framework
serializes and validates any format or version metadata within those bytes.
TRT-LLM owns their validation, tensor transfer and release; this implementation
uses native NIXL across hosts. The initial deployment requires E and disables
the consumer's vision encoder. An unavailable encoder or transfer fails the
request; a retry needs a fresh encoding result.

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

Unused encoder references are released after observed cancellation, detected response
publication failure, or rejection of a received encoder result. A completed Encode
leaves the reference available to its consumer. Delivery has no frontend acknowledgement;
if a result is accepted by the transport but never reaches the frontend, the producer
reclaims it through engine-side expiry and collection. Release is idempotent and
prevents a new claim; it does not invalidate an active transfer. A claimed
reference is never reusable, including after transfer failure.

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
keeps image encoder handoffs on the prefill path because D requires P-produced KV,
even when P can encode inline. Conditional prefill bypass remains available for text requests.
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
host and pass the resulting local path to both engine and sidecar. Replicas within
one role's worker set must use the same container-visible absolute path: Dynamo
includes the sidecar's source path in the discovery identity (`mdcsum`). Mount
host-specific cache directories at a common container path. Different roles can
use different paths. Set the engine's `--served_model_name` to the common public
name; this alias does not normalize discovery identities.
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

The upstream TRT-LLM nightly workflow selects the standard serve cases
`trtllm_image_e_pd` and `trtllm_image_e_p_d`. Each runs an aggregated baseline,
then the configured EPD topology on one GPU. Two distinct images must produce
correct colors and match the baseline's expanded token usage. Per-worker request
counters require exactly one dispatch through each stage; split P/D responses
must identify different workers. Tests use pinned model snapshots and disable
encoder and KV reuse for this comparison.

These cases exercise the HTTP frontend, routing, sidecars and native engines.
A missing or mismatched contract fails the job rather than skipping it. The EPD
prerequisites must be published in the upstream nightly before these jobs can
pass. Source overlays used for pre-publication and cross-host validation are
separate evidence, not a stock-nightly pass. Native protocol ownership and
cancellation tests remain CPU integration coverage.

The launchers use the shared process manager for teardown. No EPD-specific
process-tracking implementation is required.
