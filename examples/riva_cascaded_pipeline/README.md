<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# RIVA cascaded voice pipeline (ASR → LLM → TTS)

A proof-of-concept that expresses a cascaded voice agent as Dynamo workers instead of a pipecat pipeline. RIVA NIM speech connectors are wrapped as Dynamo workers (ASR, TTS), an LLM runs as a standard Dynamo vLLM backend, and a realtime orchestrator worker chains them together behind the Dynamo frontend's OpenAI Realtime WebSocket API.

```
realtime WS client ──> dynamo.frontend (/v1/realtime, typed realtime PushRouter)
                          │  (ModelType.Realtime, bidirectional)
                          ▼
                   orchestrator worker  ── serve_bidirectional_endpoint
                     │            │            │
              ASR client    LLM client    TTS client
                     │            │            │
               ASR worker    vLLM worker   TTS worker
            (RIVA recognize) (text-in-out) (RIVA synthesize)
```

A turn: the client streams microphone audio as `input_audio_buffer.append` events; an `input_audio_buffer.commit` marks end-of-turn. The orchestrator then runs ASR (audio → transcript) → LLM (chat, text → text) → TTS (text → audio) and streams the reply back as `response.output_audio.delta` / `response.done` events.

## Layout

```
riva_nim/        Portable building-block package (relative imports only).
  config.py        RIVA connection config + CLI flags.
  riva_client.py   build_auth / build_asr_service / build_tts_service.
  asr_worker.py    Audio → transcript (RIVA offline_recognize).
  tts_worker.py    Text → audio (RIVA synthesize).
  orchestrator.py  ModelType.Realtime engine chaining ASR → LLM → TTS.
container/       Dockerfile + build.sh that layer nvidia-riva-client on a Dynamo image.
tests/           Unit tests (run inside the layered image).
```

`riva_nim/` is self-contained and named `riva_nim` (not `riva`) so it does not shadow the `nvidia-riva-client` package, which imports as top-level `riva`. It can be relocated to `components/src/dynamo/riva/` without import changes.

## Prerequisites

- A Dynamo vLLM image (e.g. `dynamo:latest-vllm-runtime`) — carries the Dynamo Python bindings.
- A reachable RIVA server providing ASR and TTS — either a local [RIVA Skills](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/) / NIM deployment, or the hosted NVCF endpoint (see [NVCF route](#nvcf-route)).
- Dynamo's runtime dependencies for local discovery: `etcd` and `nats` (the workers default to the `etcd` discovery backend and `tcp` request plane).

## 1. Build the image

`nvidia-riva-client` is not in stock Dynamo images, so build one that layers it on:

```bash
cd examples/riva_cascaded_pipeline
BASE_IMAGE=dynamo:latest-vllm-runtime TAG=dynamo-riva:latest container/build.sh
```

## 2. Start the RIVA NIMs

The ASR and TTS workers each talk to a RIVA NIM over gRPC. Start one NIM per service — they are separate containers, so give each its own gRPC port. The images come from NGC, so log in and set your key first:

```bash
docker login nvcr.io          # username: $oauthtoken, password: <NGC API key>
export NGC_API_KEY=<your-ngc-api-key>
```

ASR NIM — Parakeet CTC 1.1B (en-US), the ASR NIM the [nemotron-voice-agent blueprint](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent) uses (check the [NGC catalog](https://catalog.ngc.nvidia.com/) for newer tags):

```bash
docker run -d --name riva-asr --gpus '"device=0"' --shm-size=8g \
  -e NGC_API_KEY -v ~/.cache/nim:/opt/nim/.cache \
  -e NIM_GRPC_API_PORT=50051 -p 50051:50051 \
  nvcr.io/nim/nvidia/parakeet-1-1b-ctc-en-us:1.4.0
```

TTS NIM — Magpie TTS Multilingual (the source of the default `Magpie-Multilingual.EN-US.Aria` voice), on a separate gRPC port so both NIMs run on one host:

```bash
docker run -d --name riva-tts --gpus '"device=0"' --shm-size=8g \
  -e NGC_API_KEY -v ~/.cache/nim:/opt/nim/.cache \
  -e NIM_GRPC_API_PORT=50052 -p 50052:50052 \
  nvcr.io/nim/nvidia/magpie-tts-multilingual:1.6.0
```

The blueprint runs both speech NIMs on one GPU and the LLM on another; adjust `--gpus` to your hardware.

Each worker's `--riva-server` must point at the matching NIM's gRPC port (ASR → `localhost:50051`, TTS → `localhost:50052` above). A single RIVA Skills server that hosts both ASR and TTS on one port works too — then point both workers at that one address. The URL is always configurable; the NIM need not be on `localhost`.

## 3. Start runtime dependencies

Start `etcd` and `nats` (the standard Dynamo local dependencies) — the workers default to the `etcd` discovery backend and `tcp` request plane.

## 4. Launch the workers

Run each inside the `dynamo-riva:latest` image, sharing the host network with `etcd`/`nats`/the NIMs. The image bakes in the `riva_nim` package (on `PYTHONPATH`), so `python -m riva_nim.<worker>` runs without mounting the source. `launch_workers.sh` starts all of them — ASR, TTS, the vLLM LLM worker, and the orchestrator (`LLM_MODEL=nvidia/Llama-3.3-Nemotron-Super-49B-v1.5 ./launch_workers.sh`); only the frontend is launched separately. It sources the repo's `examples/common/launch_utils.sh`, so run it from a checkout / mounted worktree (the individual commands below work in the bare image too).

ASR worker (points at the ASR NIM):

```bash
python -m riva_nim.asr_worker --riva-server localhost:50051 --endpoint dynamo.riva-asr.generate
```

TTS worker (points at the TTS NIM):

```bash
python -m riva_nim.tts_worker --riva-server localhost:50052 --voice Magpie-Multilingual.EN-US.Aria --endpoint dynamo.riva-tts.generate
```

Both speech workers accept `--timeout-s` (default 30) — a client-side gRPC deadline that cancels the RIVA call if the backend hangs, so an unresponsive NIM can't tie up worker threads.

LLM worker (text-in-text-out). `--use-vllm-tokenizer` makes the worker register as `ModelInput.Text` and accept chat `messages` directly; `chat` is in the default `--endpoint-types`. The worker serves `dynamo.backend.generate` by default. The blueprint pairs the speech NIMs with an NVIDIA Nemotron chat model; serve the corresponding Hugging Face weights with vLLM — e.g. `nvidia/Llama-3.3-Nemotron-Super-49B-v1.5` (large, multi-GPU). Swap in a smaller chat model such as `Qwen/Qwen3-0.6B` to try the pipeline on a single small GPU:

```bash
python -m dynamo.vllm --model nvidia/Llama-3.3-Nemotron-Super-49B-v1.5 --use-vllm-tokenizer
```

Orchestrator (the realtime model the frontend exposes). Point its client flags at the three endpoints above; set `--llm-model` to the model name the vLLM worker serves:

```bash
python -m riva_nim.orchestrator \
  --model-name riva-voice-agent \
  --asr-endpoint dynamo.riva-asr.generate \
  --llm-endpoint dynamo.backend.generate \
  --tts-endpoint dynamo.riva-tts.generate \
  --llm-model nvidia/Llama-3.3-Nemotron-Super-49B-v1.5 \
  --endpoint dynamo.riva-orchestrator.generate
```

Frontend:

```bash
python -m dynamo.frontend
```

## 5. Talk to it

Connect a WebSocket to the frontend's `/v1/realtime` endpoint and drive a turn with OpenAI-Realtime events:

- `session.update` — configure the session (echoed back as `session.updated`).
- `input_audio_buffer.append` — stream base64 LINEAR_PCM audio chunks (repeat).
- `input_audio_buffer.commit` — end the turn; the agent replies with `response.created` → `response.output_audio.delta` (base64 audio) → `response.output_audio.done` → `response.done`.

## Endpoint wiring

| Worker       | Default endpoint                  | Orchestrator flag |
|--------------|-----------------------------------|-------------------|
| ASR          | `dynamo.riva-asr.generate`        | `--asr-endpoint`  |
| TTS          | `dynamo.riva-tts.generate`        | `--tts-endpoint`  |
| LLM (vLLM)   | `dynamo.backend.generate`         | `--llm-endpoint`  |
| Orchestrator | `dynamo.riva-orchestrator.generate` | `--endpoint`    |

The orchestrator's `--*-endpoint` flags must match each worker's served endpoint. If your vLLM worker uses a non-default namespace, update `--llm-endpoint` accordingly.

## NVCF route

The same workers can target the hosted NVCF endpoint instead of a local server — no code change, just connection flags on the ASR and TTS workers:

```bash
python -m riva_nim.asr_worker \
  --riva-server grpc.nvcf.nvidia.com:443 \
  --riva-use-ssl \
  --riva-function-id <ASR_FUNCTION_ID> \
  --riva-api-key <NVCF_API_KEY>
```

`--riva-use-ssl` selects a TLS channel; `--riva-function-id` and `--riva-api-key` are sent as the `function-id` and `authorization: Bearer …` gRPC metadata RIVA NVCF expects. TTS takes the same flags with its own function id.

## RIVA connection flags (shared)

| Flag                  | Default            | Meaning |
|-----------------------|--------------------|---------|
| `--riva-server`       | `localhost:50051`  | `host:port` of the RIVA gRPC server. |
| `--riva-use-ssl`      | off                | Use a TLS channel (required for NVCF). |
| `--riva-api-key`      | unset              | NVCF API key → `authorization: Bearer` metadata. |
| `--riva-function-id`  | unset              | NVCF function id → `function-id` metadata. |
| `--riva-ssl-root-cert`| unset              | Path to a TLS root certificate. |

## Tests

The unit tests run inside the layered image (Dynamo bindings + RIVA client present); they mock RIVA and the downstream clients, so no server is needed:

```bash
docker run --rm -v "$PWD":/work -w /work dynamo-riva:latest \
  bash -c 'pip install -q pytest pytest-asyncio && pytest tests/ -p no:cacheprovider'
```

## Limitations (proof-of-concept scope)

- **Offline speech, not streaming.** ASR uses `offline_recognize` and TTS `synthesize` (one request → one result) per turn. RIVA streaming and interim transcripts are out of scope.
- **Single-turn LLM.** Each turn sends only the current transcript; no conversation history is carried across turns.
- **Cancellation is observed between stages.** A turn cancelled mid-chain stops before the next stage, but an already-in-flight ASR/LLM/TTS call still completes. Barge-in / interruption is not implemented.
- **Audio format assumptions.** ASR expects 16 kHz LINEAR_PCM, TTS emits 22.05 kHz LINEAR_PCM. No resampling is done between the realtime transport and RIVA; match the client's audio format to the workers' configured rates.
- **NVCF path is documented, not validated here.** The local path is what the tests and manual flow exercise.
