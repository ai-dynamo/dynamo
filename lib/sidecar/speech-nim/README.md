<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo Speech NIM sidecar

Connect [NVIDIA Speech NIM](https://docs.nvidia.com/nim/speech/latest/) to
Dynamo's speech endpoints. Run one sidecar beside each Speech NIM container,
using a loopback gRPC endpoint on the same host or in the same pod.

| Mode | Dynamo endpoint | Audio |
| --- | --- | --- |
| `asr` | `/v1/realtime` transcription | 24 kHz mono PCM16 input |
| `tts` | `/v1/audio/speech` | Streaming PCM16 output; sample rate set with `--sample-rate-hz` |

## Run

The shared Dynamo sidecar image selects the speech engine and then its mode:

```bash
export DYNAMO_SIDECAR_IMAGE="nvcr.io/nvidia/ai-dynamo/dynamo-sidecar:<version>"
docker run --rm --network host "${DYNAMO_SIDECAR_IMAGE}" \
  speech-nim asr --grpc-endpoint 127.0.0.1:50052
docker run --rm --network host "${DYNAMO_SIDECAR_IMAGE}" \
  speech-nim tts --grpc-endpoint 127.0.0.1:50051
```

Run these commands in separate terminals after starting the NIMs and Dynamo's
runtime services. Choose an image tag containing Speech NIM support. To build
from this checkout instead:

```bash
docker build -f lib/sidecar/Dockerfile -t dynamo-sidecar:dev .
```

For a local executable:

```bash
cargo build --release --locked -p dynamo-speech-nim-sidecar
./target/release/dynamo-speech-nim-sidecar asr --help
./target/release/dynamo-speech-nim-sidecar tts --help
```

The [Nemotron speech example](../../../examples/nemotron_speech_cascaded_pipeline/README.md)
provides the complete deployment, frontend, and speech validation commands.

## Supported configuration

The sidecar connects to local plaintext gRPC endpoints. It registers a worker
after the NIM reports a loaded model. ASR uses explicit audio commit and clear;
TTS supports PCM output at normal speed. Use `--help` for model, language, voice,
capacity, transport, and startup settings.

CPU-only protocol, registration, and executable tests run with:

```bash
cargo test --locked -p dynamo-speech-nim-sidecar --all-targets
```
