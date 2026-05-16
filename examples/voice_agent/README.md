# Voice Agent

This is the start of an end-to-end voice agent example. For now it contains a
minimal local realtime deployment and a Python WebSocket client.

The current `main` branch realtime endpoint is still experimental. It accepts
chat-completion-style JSON frames on `/v1/realtime` and returns annotated
chat-completion stream chunks. The example below keeps the normal Dynamo shape:
a frontend process owns the HTTP/WebSocket server, and a backend process
connects to the frontend and performs the echo work.

The backend bridge here is intentionally example-local. It emulates the
frontend/backend split while realtime backend routing is still being wired into
the normal Dynamo request plane.

## Run the realtime frontend

From the repo root:

```bash
cargo run --manifest-path examples/voice_agent/realtime/Cargo.toml \
  --bin voice-agent-realtime-frontend -- --port 8080 --backend-port 8081
```

If local disk space is tight, set `CARGO_TARGET_DIR` to a mounted scratch target
directory before running Cargo.

The server listens on `127.0.0.1:8080` by default and exposes:

- `GET /health`
- `WS /v1/realtime`

It also listens for the backend on `127.0.0.1:8081`.

## Run the realtime backend

In a second terminal:

```bash
cargo run --manifest-path examples/voice_agent/realtime/Cargo.toml \
  --bin voice-agent-realtime-backend -- --connect 127.0.0.1:8081
```

The backend connects to the frontend bridge and echoes the latest user text
back one character at a time.

## Run the Python client

Install the only client dependency:

```bash
PYTHON=/path/to/python
uv pip install --python "$PYTHON" -r examples/voice_agent/requirements.txt
```

Then send one text request:

```bash
"$PYTHON" examples/voice_agent/client.py --text "hello realtime"
```

The client sends:

```json
{
  "model": "echo",
  "messages": [{ "role": "user", "content": "hello realtime" }]
}
```

and prints streamed `delta.content` text until it sees
`finish_reason: "stop"`.

## Current limitations

- This is a text-only echo scaffold, not a voice agent yet.
- The frame format matches the current merged `/v1/realtime` implementation,
  not the dedicated OpenAI Realtime event types being added in follow-up PRs.
- The backend bridge is local-only and not the final Dynamo realtime request
  plane.
