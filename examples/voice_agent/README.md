# Voice Agent

This is the start of an end-to-end voice agent example. For now it contains a
minimal local realtime deployment, a Python Pipecat backend, a mock Dynamo ASR
worker, and a Python WebSocket client.

This branch includes the realtime protocol and `ModelManager` wiring from the
open realtime PRs. The public WebSocket endpoint exchanges OpenAI Realtime
events on `/v1/realtime`: the client receives `session.created`, sends
`session.update` with `session.model`, then streams input with
`input_audio_buffer.append`.

The example keeps the normal Dynamo shape: a frontend process owns the
HTTP/WebSocket server, and a backend process connects to the frontend and does
the backend work. The Pipecat backend then calls a separate Dynamo ASR endpoint
over the normal Dynamo request plane. The bridge between the realtime frontend
and the Pipecat backend is still example-local because remote bidirectional
realtime dispatch is separate follow-up work.

The backend bridge uses newline-delimited JSON Realtime events, so the Python
backend does not need Rust realtime type bindings yet.

## Layout

```
examples/voice_agent/
├── frontend/   # Rust realtime frontend (HTTP/WebSocket server + backend bridge)
├── backend/    # Python Pipecat realtime backend + mock Dynamo ASR worker
└── client/     # Python WebSocket client
```

## Install Python dependencies

Install the client and Pipecat backend dependencies into your Python
environment:

```bash
PYTHON=/mnt/scratch/nealv/venvs/dynamo_realtime/bin/python
uv pip install --python "$PYTHON" -r examples/voice_agent/client/requirements.txt
uv pip install --python "$PYTHON" -r examples/voice_agent/backend/requirements.txt
```

The Pipecat backend and mock ASR worker also require `dynamo` and
`dynamo-runtime` in that same environment.

## Run the mock Dynamo ASR worker

From the repo root, start the mock ASR endpoint in one terminal:

```bash
PYTHON=/mnt/scratch/nealv/venvs/dynamo_realtime/bin/python
DYN_DISCOVERY_BACKEND=file DYN_REQUEST_PLANE=tcp DYN_EVENT_PLANE=zmq \
  "$PYTHON" examples/voice_agent/backend/mock_asr_worker.py
```

The worker serves `voice_agent.asr.transcribe`. It accepts an `AsrRequest`
containing base64-encoded WAV or PCM audio and streams `AsrTranscript` chunks
back to the caller. For now this is a deterministic mock; it proves the
Pipecat-to-Dynamo request path without requiring a real ASR model worker.

## Run the realtime frontend

In another terminal, start the realtime frontend:

```bash
CARGO_TARGET_DIR=/mnt/scratch/nealv/cargo/targets \
cargo run --manifest-path examples/voice_agent/frontend/Cargo.toml \
  --bin voice-agent-realtime-frontend -- --port 8080 --backend-port 8081
```

The frontend registers realtime model `echo`, listens on `127.0.0.1:8080`, and
exposes:

- `GET /health`
- `WS /v1/realtime`

It also listens for the backend bridge connection on `127.0.0.1:8081`.

## Run the Pipecat realtime backend

In another terminal, run the Pipecat backend:

```bash
PYTHON=/mnt/scratch/nealv/venvs/dynamo_realtime/bin/python
DYN_DISCOVERY_BACKEND=file DYN_REQUEST_PLANE=tcp DYN_EVENT_PLANE=zmq \
  "$PYTHON" examples/voice_agent/backend/pipecat_backend.py \
  --connect 127.0.0.1:8081
```

This keeps the Dynamo/OpenAI Realtime shape at the frontend. The backend maps
`input_audio_buffer.append` into a Pipecat `InputAudioRawFrame`, wraps the
single append as one VAD-delimited utterance for this PoC, sends it through a
custom `DynamoASRService`, and returns the transcript as
`conversation.item.input_audio_transcription.completed` plus streamed text
response events.

## Run the Python client

Then send one request:

```bash
PYTHON=/mnt/scratch/nealv/venvs/dynamo_realtime/bin/python
"$PYTHON" examples/voice_agent/client/client.py --audio-file /path/to/input-16k-mono.wav
```

The client sends a 16-bit mono PCM WAV file as an
`input_audio_buffer.append` event and prints streamed text deltas as they
arrive. The backend defaults to 16 kHz input and does not resample.

For larger files, split the WAV into multiple realtime append events so each
WebSocket frame stays below the default frame limit:

```bash
PYTHON=/mnt/scratch/nealv/venvs/dynamo_realtime/bin/python
"$PYTHON" examples/voice_agent/client/client.py \
  --audio-file /path/to/long-input-16k-mono.wav \
  --chunk-seconds 120 \
  --timeout 300
```

## Current Limitations

- The ASR worker is a mock Dynamo worker, not a real model worker yet.
- The frontend/backend bridge is local-only JSON and not the final Dynamo
  realtime request plane.
- The Pipecat ASR backend currently treats one `input_audio_buffer.append` as a
  complete utterance. Use client-side chunking for large files in this PoC.
- A true Python Dynamo realtime worker will need bindings and runtime support
  for bidirectional Realtime event streams once remote dispatch lands.
- There is no microphone capture, playback, resampling, multi-append buffering,
  or real VAD yet.
