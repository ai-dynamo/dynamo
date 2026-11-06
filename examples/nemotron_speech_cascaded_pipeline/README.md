<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Nemotron Speech Cascaded Pipeline

This directory holds OpenAI-compatible realtime ASR and streaming TTS adapters
for NVIDIA Speech NIMs serving Nemotron models, a Kubernetes manifest that
deploys them with a Nemotron LLM served by vLLM, and a speech smoke test.

For architecture, deployment, measurement, and cleanup, see
[Nemotron Speech Cascaded Pipeline](../../docs/fern/pages/use-cases/speech/nemotron-speech-pipeline.md).

## Contents

| Path | Description |
|---|---|
| `container/` | `Dockerfile`, `build.sh`, and `install.sh` for the CPU-only speech adapter image built on the Dynamo frontend image |
| `deploy/agg.yaml` | DynamoGraphDeployment with the frontend, vLLM worker, and ASR and TTS adapter workers with Speech NIM sidecars |
| `launch_workers.sh` | Launches the frontend, ASR and TTS adapter workers, and vLLM worker locally against reachable Speech NIM gRPC endpoints |
| `nemotron_speech/` | Adapter package: `asr/` realtime transcription worker, `tts/` streaming speech worker, Speech NIM connection config, and Riva helpers |
| `smoke_speech_loop.py` | Speech smoke test and ASR-to-LLM handoff measurement client |
| `requirements.txt` | Pinned `nvidia-riva-client` dependency for the adapter image |
| `tests/`, `pytest.ini` | Unit tests for connection configuration and the smoke client |

## Tests

Running the adapter workers or unit tests outside the container requires Python
3.11 or newer.

The example's unit tests cover connection configuration, endpoint resolution,
and the external ASR-to-LLM handoff without installing the Riva client.
Model-registration regressions are covered
by `lib/bindings/python/tests/test_runtime_data_discovery.py` in the regular
Dynamo binding suite. The Riva-dependent adapter, worker, and connection modules
have no automated test coverage. Run the deployed smoke test manually for
functional validation.

```bash
python3 -m pip install aiohttp pytest pytest-asyncio
PYTHONPATH=components/src:lib/bindings/python/src \
  python3 -m pytest -xvv examples/nemotron_speech_cascaded_pipeline/tests
bash -n examples/nemotron_speech_cascaded_pipeline/launch_workers.sh
bash -n examples/nemotron_speech_cascaded_pipeline/container/build.sh
bash -n examples/nemotron_speech_cascaded_pipeline/container/install.sh
pre-commit run check-yaml --files examples/nemotron_speech_cascaded_pipeline/deploy/agg.yaml
```

For functional validation, run `smoke_speech_loop.py` and confirm that the
transcript matches the synthesized sentence closely enough to recognize the
intended text.
