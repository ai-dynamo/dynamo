# Voice Agent

This is the start of an end-to-end voice agent example. For now it contains a
minimal local realtime deployment, a Python backend, and a Python WebSocket
client.

This branch includes the realtime protocol and `ModelManager` wiring from the
open realtime PRs. The public WebSocket endpoint exchanges OpenAI Realtime
events on `/v1/realtime`: the client receives `session.created`, sends
`session.update` with `session.model`, then streams input with
`input_audio_buffer.append`.

The example keeps the normal Dynamo shape: a frontend process owns the
HTTP/WebSocket server, and a backend process connects to the frontend and does
the backend work. The bridge between them is still example-local because remote
bidirectional request-plane dispatch is separate follow-up work.

The backend bridge uses newline-delimited JSON Realtime events, so the Python
backend does not need Rust realtime type bindings yet.

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

## Run the Python realtime backend

In a second terminal:

```bash
PYTHON=/mnt/scratch/nealv/venvs/dynamo_realtime/bin/python
"$PYTHON" examples/voice_agent/backend.py --connect 127.0.0.1:8081
```

The backend echoes appended audio bytes by streaming
`response.output_audio.delta` events and then `response.done`.

## Run the Pipecat NVIDIA ASR backend

As an alternate backend, this example can transcribe input audio with Pipecat's
NVIDIA ASR service. For the local checkout at `/home/nealv/dynamo/pipecat`,
install Pipecat and the minimal dependencies into the requested venv:

```bash
PYTHON=/mnt/scratch/nealv/venvs/dynamo_realtime/bin/python
uv pip install --python "$PYTHON" --no-deps -e /home/nealv/dynamo/pipecat
uv pip install --python "$PYTHON" \
  loguru pydantic pydantic-core annotated-types typing-inspection wait_for2 \
  docstring_parser aiofiles aiohttp numpy pyloudnorm resampy soxr protobuf \
  Markdown Pillow openai nltk grpcio grpcio-tools nvidia-riva-client
```

Then run the Pipecat backend instead of `backend.py`:

```bash
PYTHON=/mnt/scratch/nealv/venvs/dynamo_realtime/bin/python
NVIDIA_API_KEY=... "$PYTHON" examples/voice_agent/pipecat_backend.py \
  --connect 127.0.0.1:8081
```

This keeps the Dynamo/OpenAI Realtime shape at the frontend. The backend maps
`input_audio_buffer.append` into a Pipecat `InputAudioRawFrame`, wraps the
single append as one VAD-delimited utterance for this PoC, sends it through
`NvidiaSegmentedSTTService`, and returns the transcript as
`conversation.item.input_audio_transcription.completed` plus a text response.
By default it targets NVIDIA's Parakeet CTC 0.6B ASR endpoint.

For a locally deployed Riva/NIM ASR service, point the backend at the insecure
gRPC port and disable cloud metadata. For example, if the NIM is bound on
`0.0.0.0:50051`, connect to it from the backend as `127.0.0.1:50051`:

```bash
PYTHON=/mnt/scratch/nealv/venvs/dynamo_realtime/bin/python
"$PYTHON" examples/voice_agent/pipecat_backend.py \
  --connect 127.0.0.1:8081 \
  --nvidia-server 127.0.0.1:50051 \
  --nvidia-no-ssl \
  --nvidia-function-id "" \
  --nvidia-model-name parakeet-0.6b-en-US-asr-offline
```

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

To exercise the Pipecat ASR backend, send a 16-bit mono PCM WAV file. The
backend defaults to 16 kHz input and does not resample:

```bash
"$PYTHON" examples/voice_agent/client.py --audio-file /path/to/input-16k-mono.wav
```

For larger files, split the WAV into multiple realtime append events so each
WebSocket frame stays below the default frame limit:

```bash
"$PYTHON" examples/voice_agent/client.py \
  --audio-file examples/voice_agent/MLKDream_64kb.wav \
  --chunk-seconds 120 \
  --timeout 300
```

## Current Limitations

- This is an audio-event echo scaffold, not a full voice agent yet.
- The frontend/backend bridge is local-only JSON and not the final Dynamo
  realtime request plane.
- The Pipecat ASR backend currently treats one `input_audio_buffer.append` as a
  complete utterance. Use client-side chunking for large files in this PoC.
- A true Python Dynamo realtime worker will need bindings and runtime support
  for bidirectional Realtime event streams once remote dispatch lands.
- There is no microphone capture, playback, resampling, multi-append buffering,
  or real VAD yet.
