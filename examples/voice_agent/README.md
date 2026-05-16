# Voice Agent

This is the start of an end-to-end voice agent example. For now it contains a
minimal local realtime deployment and a Python WebSocket client.

This branch includes the realtime protocol and `ModelManager` wiring from the
open realtime PRs. The public WebSocket endpoint exchanges OpenAI Realtime
events on `/v1/realtime`: the client receives `session.created`, sends
`session.update` with `session.model`, then streams input with
`input_audio_buffer.append`.

The example keeps the normal Dynamo shape: a frontend process owns the
HTTP/WebSocket server, and a backend process connects to the frontend and does
the echo work. The bridge between them is still example-local because remote
bidirectional request-plane dispatch is separate follow-up work.

## Run the realtime frontend

From the repo root:

```bash
CARGO_TARGET_DIR=/mnt/scratch/nealv/cargo/targets \
cargo run --manifest-path examples/voice_agent/realtime/Cargo.toml \
  --bin voice-agent-realtime-frontend -- --port 8080 --backend-port 8081
```

The frontend registers realtime model `echo`, listens on `127.0.0.1:8080`, and
exposes:

- `GET /health`
- `WS /v1/realtime`

It also listens for the backend bridge connection on `127.0.0.1:8081`.

## Run the realtime backend

In a second terminal:

```bash
CARGO_TARGET_DIR=/mnt/scratch/nealv/cargo/targets \
cargo run --manifest-path examples/voice_agent/realtime/Cargo.toml \
  --bin voice-agent-realtime-backend -- --connect 127.0.0.1:8081
```

The backend echoes appended audio bytes by streaming
`response.output_audio.delta` events and then `response.done`.

## Run the Python client

Install the only client dependency:

```bash
PYTHON=/mnt/scratch/nealv/venvs/dynamo_realtime/bin/python
uv pip install --python "$PYTHON" -r examples/voice_agent/requirements.txt
```

Then send one request:

```bash
"$PYTHON" examples/voice_agent/client.py --text "hello realtime"
```

The client base64-encodes `--text` into an `input_audio_buffer.append` event,
collects echoed `response.output_audio.delta` events, decodes them, and prints
the original text.

## Current Limitations

- This is an audio-event echo scaffold, not a full voice agent yet.
- The frontend/backend bridge is local-only and not the final Dynamo realtime
  request plane.
- Audio is treated as opaque base64 payload for now; there is no capture,
  playback, VAD, transcription, or model inference.
