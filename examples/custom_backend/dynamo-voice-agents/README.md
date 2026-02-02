<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Streaming ASR Voice Agents

Real-time speech-to-text transcription using NVIDIA Dynamo distributed runtime with the Parakeet RNNT model. Supports both file-based transcription and **real-time microphone streaming**.

## Architecture

### File-Based Transcription
```text
┌─────────────┐      ┌─────────────────┐      ┌──────────────────┐
│  ASR Client │ ───▶ │  Audio Chunker  │ ───▶ │  ASR Inference   │
│             │      │  (streaming_asr/│      │  (streaming_asr/ │
│             │ ◀─── │   chunker)      │ ◀─── │   inference)     │
└─────────────┘      └─────────────────┘      └──────────────────┘
     audio file           load & chunk            RNNT model
                          forward to              inference
                          inference               (GPU)
```

### Real-Time Microphone Streaming
```text
┌─────────────────┐                      ┌─────────────────────────┐
│   Mic Client    │  ══════════════════▶ │  Streaming ASR Service  │
│  (local machine)│     audio chunks     │    (GPU server)         │
│                 │  ◀══════════════════ │  streaming_asr/realtime/│
└─────────────────┘   transcriptions     │    transcribe_stream    │
    microphone                           └─────────────────────────┘
    capture                                    Parakeet RNNT
                                               real-time inference
```

## Components

| Component | File | Description |
|-----------|------|-------------|
| **ASR Inference** | `asr_inference.py` | Loads NVIDIA Parakeet RNNT model for file-based chunked inference |
| **Audio Chunker** | `audio_chunker.py` | Loads audio files, computes chunk boundaries, forwards to inference |
| **Streaming ASR Service** | `streaming_asr_service.py` | Real-time streaming service for microphone input |
| **File Client** | `asr_client.py` | Test client for file-based transcription |
| **Mic Client** | `mic_client.py` | Real-time microphone streaming client |
| **Protocol** | `protocol.py` | Pydantic models for inter-worker communication |
| **Tracing** | `tracing.py` | OpenTelemetry tracing utilities |

## Quick Start

### Option 1: Real-Time Microphone Streaming (Remote Client)

This is the recommended setup for real-time transcription from your local microphone to a remote GPU server.

#### Step 1: Start the Server (GPU Machine)

**Using Docker (Recommended):**
```bash
# Build the image
docker build -t dynamo-voice-agents .

# Run with remote access enabled (auto-detects server IP)
docker run --gpus '"device=0"' \
    -p 2379:2379 \
    -e ASR_CUDA_DEVICE=0 \
    -e DYN_STORE_KV=etcd \
    -e ENABLE_REMOTE_ACCESS=true \
    dynamo-voice-agents

# Or specify the server IP explicitly
docker run --gpus '"device=0"' \
    -p 2379:2379 \
    -e ASR_CUDA_DEVICE=0 \
    -e DYN_STORE_KV=etcd \
    -e DYN_TCP_RPC_HOST=<SERVER_IP> \
    dynamo-voice-agents
```

**Using Slurm (NVIDIA DGX/OCI):**
```bash
# Launch interactive job with container
srun -A <account> -p interactive_singlenode -G 4 --time 04:00:00 \
    --container-mounts /path/to/cache:/root/.cache \
    --container-image gitlab-master.nvidia.com/fciannella/dynamo-voice-agents/dynamo-voice-agents:latest \
    --pty bash

# Inside the container, start services with remote access
export DYN_STORE_KV=etcd
export DYN_TCP_RPC_HOST=$(hostname -I | awk '{print $1}')  # Auto-detect IP
bash start_services.sh &

# Note the server IP address for the client
hostname -I | awk '{print $1}'
```

**Verify Services Started:**
```
============================================
All services started successfully!
  - etcd PID: xxxxx
  - Inference Worker PID: xxxxx
  - Chunker Worker PID: xxxxx
  - Streaming Service PID: xxxxx
============================================

Endpoints available:
  File-based transcription:
    - streaming_asr/inference/process
    - streaming_asr/chunker/transcribe
  Real-time microphone streaming:
    - streaming_asr/realtime/transcribe_stream
```

#### Step 2: Run the Client (Local Machine with Microphone)

**Install Dependencies:**
```bash
pip install ai-dynamo sounddevice uvloop
```

**Run Microphone Client:**
```bash
cd examples/custom_backend/dynamo-voice-agents/src

# Set environment to connect to remote server
export DYN_STORE_KV=etcd
export ETCD_ENDPOINTS=http://<SERVER_IP>:2379
export DYN_REQUEST_PLANE=tcp

# Run the microphone client
python3 mic_client.py
```

**Expected Output:**
```
============================================================
🎙️  Real-time Streaming ASR Client
============================================================
⏳ Waiting for ASR service...
✅ Connected to ASR service!

📢 Available audio devices:
  → [15] default (inputs: 32)

🎤 Using default input device: [15] default
📊 Sample rate: 16000 Hz, Chunk: 500ms

🔴 Recording... Press Ctrl+C to stop.

🎤 Starting streaming session: abc123...
============================================================
📝 hello world how are you today
```

Press `Ctrl+C` to stop recording.

---

### Option 2: File-Based Transcription (Local)

For transcribing audio files without microphone streaming.

**Using Docker:**
```bash
# Build the image
docker build -t dynamo-voice-agents .

# Run with GPU 0
docker run --gpus '"device=0"' \
    -e ASR_CUDA_DEVICE=0 \
    -v $(pwd)/data:/app/data:ro \
    dynamo-voice-agents

# Or use docker compose
docker compose up -d

# View logs
docker compose logs -f
```

### Option 3: Local Development (No Docker)

**Prerequisites:**
- NVIDIA GPU with CUDA support
- Python 3.10+
- NeMo Toolkit 2.4 installed (`pip install nemo_toolkit[asr]==2.4.0`)
- Dynamo runtime (`pip install ai-dynamo`)
- Audio processing libraries: libsndfile, sox, ffmpeg

**Terminal 1 - Start Inference Worker:**
```bash
cd examples/custom_backend/dynamo-voice-agents/src
export ASR_CUDA_DEVICE=0  # GPU device to use
python3 asr_inference.py
```

**Terminal 2 - Start Audio Chunker:**
```bash
cd examples/custom_backend/dynamo-voice-agents/src
python3 audio_chunker.py
```

**Terminal 3 - Run Client:**
```bash
cd examples/custom_backend/dynamo-voice-agents/src
python3 asr_client.py /path/to/your/audio.wav
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ASR_CUDA_DEVICE` | `0` | CUDA device index for inference |
| `ASR_MODEL_NAME` | `nvidia/parakeet-rnnt-1.1b` | ASR model to use |
| `WORKER_LOG_LEVEL` | `INFO` | Logging level |
| `DYN_STORE_KV` | `file` | Dynamo KV store backend (`file` or `etcd`) |
| `DYN_REQUEST_PLANE` | `tcp` | Dynamo request plane (`tcp` or `nats`) |
| `DYN_TCP_RPC_HOST` | `0.0.0.0` | Server IP for remote access (auto-detected if `ENABLE_REMOTE_ACCESS=true`) |
| `ENABLE_REMOTE_ACCESS` | `false` | Enable remote client access (auto-configures etcd and IP) |
| `ETCD_ENDPOINTS` | `http://localhost:2379` | etcd endpoint(s) for client connection |
| `OTEL_EXPORT_ENABLED` | `false` | Enable OpenTelemetry tracing |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | `http://localhost:4317` | OTLP endpoint |

### Docker Run Examples

```bash
# Use specific GPU
docker run --gpus '"device=1"' \
    -e ASR_CUDA_DEVICE=0 \
    dynamo-voice-agents

# Enable tracing with Jaeger
docker run --gpus all \
    -e OTEL_EXPORT_ENABLED=true \
    -e OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://host.docker.internal:4317 \
    dynamo-voice-agents

# Mount audio data directory
docker run --gpus all \
    -v /path/to/audio/files:/app/data:ro \
    dynamo-voice-agents
```

### Docker Compose with Tracing

```bash
# Start with Jaeger tracing
docker compose --profile tracing up -d

# Access Jaeger UI at http://localhost:16686
```

## Testing

### With the Included Client

```bash
# Inside the container
docker exec -it dynamo-asr python /app/src/asr_client.py /app/data/sample.wav

# Or from host with mounted data
docker run --gpus all \
    -v $(pwd)/my-audio.wav:/app/data/test.wav:ro \
    dynamo-voice-agents \
    python /app/src/asr_client.py /app/data/test.wav
```

### Sample Output

```
Audio file: /app/data/sample.wav
Waiting for workers...
Connected.

[PARTIAL] Hello
[PARTIAL] Hello world
[PARTIAL] Hello world how are
[FINAL]   Hello world how are you today
```

## Streaming Configuration

The chunking configuration optimizes for a balance of quality and latency:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Chunk size | 2.0s | Audio processed per chunk |
| Left context | 10.0s | Previous audio context |
| Right context | 2.0s | Lookahead audio context |
| Sample rate | 16kHz | Audio sample rate |

## Model Information

**Default Model:** `nvidia/parakeet-rnnt-1.1b`

- Architecture: RNNT (Recurrent Neural Network Transducer)
- Parameters: 1.1 billion
- Languages: English
- Streaming: Native support with stateful decoding

The model is automatically downloaded from Hugging Face on first run. Use the `asr-model-cache` volume to persist the download.

## Troubleshooting

### GPU Not Found

```bash
# Verify NVIDIA driver and container toolkit
nvidia-smi
docker run --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```

### Model Download Fails

```bash
# Check network and HuggingFace access
# Set HF_TOKEN if using gated models
docker run -e HF_TOKEN=your_token ...
```

### Workers Can't Connect

```bash
# Ensure both workers use same DYN_STORE_KV
# Check the state directory is writable
ls -la /app/state/
```

## Architecture Details

See [docs/dynamo-asr-webrtc-architecture.md](docs/dynamo-asr-webrtc-architecture.md) for the complete architecture proposal including WebRTC integration plans.



## Running on OCI


```
docker build -f Dockerfile -t gitlab-master.nvidia.com/fciannella/dynamo-voice-agents/dynamo-voice-agents:0.0.1 -t gitlab-master.nvidia.com/fciannella/dynamo-voice-agents/dynamo-voice-agents:latest .
```

```
docker push gitlab-master.nvidia.com/fciannella/dynamo-voice-agents/dynamo-voice-agents:latest

docker push gitlab-master.nvidia.com/fciannella/dynamo-voice-agents/dynamo-voice-agents:0.0.1
```


```
srun -A llmservice_nemo_mlops -p interactive_singlenode -G 4 --time 04:00:00 --container-mounts /lustre/fsw/portfolios/llmservice/users/fciannella/cache:/root/.cache,/lustre/fsw/portfolios/llmservice/users/fciannella/src:/root/src --container-image gitlab-master.nvidia.com/fciannella/dynamo-voice-agents/dynamo-voice-agents:0.0.1 --pty bash

```