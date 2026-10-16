<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Realtime WS migration benchmark

This local prototype harness measures the existing Dynamo audio paths, the same
public APIs backed by a Rust sidecar, and direct upstream endpoints. It runs
against already-running servers and does not create deployments.

Install `requirements.txt` into an isolated environment. Python 3.11 or later is
required. The harness saves dependency versions in each run's `config.json`.

```bash
python benchmarks/realtime_ws/benchmark.py \
  --profile dynamo-asr --url ws://127.0.0.1:8000/v1/realtime \
  --model Qwen/Qwen3-ASR-0.6B --audio /path/to/input.wav \
  --warmups 2 --requests 10 --concurrency 1 --pace 1 --chunk-ms 100 \
  --label baseline-asr --output-dir /path/to/new-results-directory
```

Use the same `dynamo-asr`, `dynamo-omni`, or `tts-http` profile for both the
baseline and candidate Dynamo frontend. Change the URL and label, and choose a
new output directory for each run.

| Profile | Full endpoint path | Input / termination |
|---|---|---|
| `dynamo-asr` | `/v1/realtime` | 24 kHz PCM16, nested transcription session; bare commit; transcription completed |
| `native-asr` | `/v1/realtime` | 16 kHz PCM16, top-level model; initial nonfinal commit starts generation, final commit closes input |
| `dynamo-omni` | `/v1/realtime` | Legacy 16 kHz PCM16, realtime session; bare commit; response done |
| `native-omni` | `/v1/realtime?model=...` | 24 kHz PCM16; commit followed by response.create; response done |
| `tts-http` | `/v1/audio/speech` | Text, `response_format=pcm`, `stream=true`; HTTP EOF |
| `native-tts-ws` | `/v1/audio/speech/stream` | session.config, input.text, input.done; binary PCM; session.done |

TTS defaults to voice `vivian`, language `English`, PCM16 at 24 kHz.
`native-tts-ws` uses `stream_audio=true`, `split_granularity=none`, and reads
the output rate/channels from `audio.start` (channels defaults to one for
v0.31.0rc1). TTS HTTP has no rate metadata; use `--output-rate` to match the
deployed model. ASR language defaults to `en`. Omni defaults to audio output;
`--modalities text` requests text-only output from backends that support it.
The prototype Omni sidecar currently forces audio output.

Use `--request-extra '{"seed":42,"max_new_tokens":256}'` to add TTS request
options. The native WS endpoint accepts these exact fields. HTTP frontend support
for these fields must be verified separately; the harness does not claim that
different APIs share all generation controls. `--session-extra` merges additional
fields into the realtime session, e.g. `'{"max_output_tokens":256}'`.
Native ASR has its own model-only configuration and ignores session-extra.

Each request opens a fresh connection. Warmups are sequential and excluded from
measured distributions and throughput. Measured requests use closed-loop bounded
concurrency, with no retries. An existing output directory is rejected to prevent
mixing runs. Failed requests and failed warmups cause a nonzero exit status.
ASR and text-only Omni require a non-whitespace effective transcript; a blank
final ASR transcript is a failure even if partial deltas arrived earlier.
`response.done` with status `incomplete` and reason `max_output_tokens` is recorded
as `truncated=true`, with a distinct `truncated` error category. It does not count
as successful full completion or enter successful-request latency distributions.
Its partial audio, terminal status/details, and per-request timing remain available
in `requests.jsonl`; `summary.json` counts truncations separately from other errors.

`--pace 1` sends audio at realtime speed, with the final commit at the end of its
duration. `--pace 0` sends as fast as possible; `--pace 2` sends at twice realtime
speed. Audio chunks are encoded before timing. Resampling is performed once on
the whole input file using scipy's polyphase filter, outside timed requests.
Source WAV and prepared PCM hashes, input duration, sample rates, chunk size,
pace, and protocol settings are retained for comparison. Baseline and candidate
must receive the same prepared PCM hash. Direct upstream runs need different
sample rates and are a diagnostic control, not an identical-wire comparison.
Use 100 ms input chunks for the primary sidecar comparison: the prototype
resampler batches that much input before emitting upstream audio. Smaller
chunks would additionally measure its batching delay. The sidecar requantizes
resampled audio to PCM16 for upstream WS, whereas the existing Python worker
can pass float waveforms internally; inspect output quality for this difference.

Outputs:

- `config.json`: exact workload, input hashes, environment/dependency versions.
- `requests.jsonl`: every request including warmups, transcripts, failures,
  timing, output duration, RMS and nonzero-sample counts, output hash, WAV path.
- `events.jsonl`: ordered-per-request received events and input markers with
  relative timestamps; base64 audio is redacted, binary chunks record byte counts.
- `summary.json`: successful-request distributions, error counts, wall-clock
  throughput, and counts of requests producing output before final input.
- Per-request WAVs for listening and independent transcription/quality checks.

First text means the first non-whitespace delta, or the complete ASR transcript
if the endpoint supplies no deltas. This raw metric includes model control
prefixes such as Qwen ASR's `language English<asr_text>`; it does not necessarily
measure the first spoken content. First audio means received PCM samples, never
HTTP headers or WAV headers. WAV returned despite a PCM request is a failure.
`*_from_first_input_s` starts immediately before the first input send;
`*_after_commit_s` starts immediately before the final input commit send (or TTS
HTTP POST). Negative values indicate output before commit. `setup_s` is local
request start to first input send, including WS connection and config sends, but
not waiting for an application-level config acknowledgement. HTTP `connect_s`
is null because HTTP headers are not a connection-ready timestamp.
`generation_rtf` is completion-after-commit divided by output audio duration.
Request completion is measured separately from connection teardown. File writes
are deferred until all measured requests finish, so they do not block in-flight
requests or inflate measured throughput. Payloads and events are retained in
memory until then; size the run to fit client memory.

Use paired runs on the same GPU/model revisions and launch settings. Record the
server-side sampling configuration separately: a client field alone does not prove
that both paths apply it. Small-N p95/p99 are descriptive, not reliable tail SLO
estimates. Compare transcripts, audio duration and retained audio before treating
a lower latency as an improvement. Transport chunk counts depend on framing and
are not model token counts. This harness does not establish full feature parity.

CPU-only protocol tests need `pytest`, `pytest-asyncio`, and `pytest-timeout`:

```bash
python -m pytest -c pyproject.toml --confcutdir=benchmarks/realtime_ws \
  -o filterwarnings=error -q benchmarks/realtime_ws/tests
```

The explicit root config supplies Dynamo markers; the warnings override keeps
this isolated environment independent of unrelated engine-package warnings.
Tests cover incremental ASR timing, native TTS completion/rate, HTTP headers vs
audio, failures/premature close/empty output, event redaction and resampling.
