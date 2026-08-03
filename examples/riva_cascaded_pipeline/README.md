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

# Riva speech adapters for a cascaded voice agent

This example exposes Riva-compatible speech NIMs through Dynamo's standard
OpenAI-compatible APIs. A voice application such as Pipecat remains responsible
for the ASR -> LLM -> TTS cascade, conversation state, turn taking, tools, and
barge-in.

```text
browser -> Pipecat -> Dynamo frontend -> /v1/realtime        -> ASR adapter -> ASR NIM
                                    |-> /v1/chat/completions -> vLLM worker
                                    `-> /v1/audio/speech     -> TTS adapter -> TTS NIM
```

The deployment matches the Generic Assistant recipe in the Nemotron Voice
Agent Blueprint so the same UI and workload can compare direct NIM access with
Dynamo routing:

| Stage | Model |
| --- | --- |
| ASR | Nemotron ASR Streaming 1.2.0, English streaming profile |
| LLM | NVIDIA Nemotron 3 Nano 30B A3B FP8 |
| TTS | Magpie TTS Multilingual 1.8.0 |

Two DGD profiles keep model selection fixed while making resource intent clear:

| Profile | LLM | Magpie batch | Total GPUs | Purpose |
| --- | --- | ---: | ---: | --- |
| `latency` | TP=1 | 8 | 3 | Functional checks and single-session latency |
| `throughput` | TP=2 | 64 | 4 | Comparison with the Blueprint performance profile |

The throughput profile requires an 80 GB-class GPU for Magpie batch size 64.
The latency profile uses the same models and API contracts but is not a
throughput comparison target.

## Build

The Riva Python client is not included in the vLLM runtime. Build one image for
the frontend, vLLM worker, and lightweight speech adapters:

```bash
cd examples/riva_cascaded_pipeline
BASE_IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:<tag> \
  TAG=<registry>/dynamo-riva:<tag> ./container/build.sh
```

Use an immutable base image digest for measured runs.

## Run locally

Start the ASR and TTS NIMs before launching Dynamo. The host ports below avoid
collisions between their gRPC and health endpoints:

```bash
docker run -d --name nemotron-asr --gpus '"device=0"' --shm-size=16g \
  -e NGC_API_KEY -e NIM_GRPC_API_PORT=50152 -e NIM_HTTP_API_PORT=9012 \
  -e NIM_TAGS_SELECTOR='type=en-US,mode=str' \
  -p 50152:50152 -p 9012:9012 -v ~/.cache/nim/asr:/opt/nim/.cache \
  nvcr.io/nim/nvidia/nemotron-asr-streaming:1.2.0

docker run -d --name magpie-tts --gpus '"device=1"' --shm-size=16g \
  -e NGC_API_KEY -e NIM_GRPC_API_PORT=50151 -e NIM_HTTP_API_PORT=9011 \
  -e NIM_TAGS_SELECTOR='name=magpie-tts-multilingual,batch_size=8' \
  -p 50151:50151 -p 9011:9011 -v ~/.cache/nim/tts:/opt/nim/.cache \
  nvcr.io/nim/nvidia/magpie-tts-multilingual:1.8.0
```

Inside the built Dynamo image, start the normal local discovery dependencies,
then launch all public endpoints:

```bash
ASR_RIVA_SERVER=localhost:50152 \
TTS_RIVA_SERVER=localhost:50151 \
LLM_GPU_DEVICES=2,3 \
LLM_TP_SIZE=2 \
  ./launch_workers.sh
```

This layout uses GPUs 0 and 1 for the NIM containers and GPUs 2 and 3 for vLLM.
For a three-GPU latency setup, use `LLM_GPU_DEVICES=2 LLM_TP_SIZE=1`. Each speech
worker waits for its Riva gRPC channel before registering the model, so the
frontend cannot route traffic to an unready NIM.

The resulting contracts are:

- `ws://localhost:8000/v1/realtime`, transcription sessions using 24 kHz PCM
- `http://localhost:8000/v1/chat/completions`
- `http://localhost:8000/v1/audio/speech`, streaming 24 kHz PCM

The TTS adapter requires Dynamo's streaming `/v1/audio/speech` support from
[PR #12100](https://github.com/ai-dynamo/dynamo/pull/12100). Until that change
lands on `main`, build this example from a revision containing the PR; the
non-streaming frontend returns only the first upstream audio chunk.

The realtime ASR adapter intentionally disables server VAD. Pipecat's local VAD
and Smart Turn processors commit input audio, preserving identical conversation
behavior across both backend profiles.

Run a compact speech-path check after all three models are warm:

```bash
python3 smoke_speech_loop.py
```

The check streams TTS output into the realtime ASR session and reports TTS TTFB,
ASR first-transcript latency, PCM RMS, and the final transcript. It fails on HTTP
or WebSocket error, silent audio, or an empty transcript.

## Deploy with DGD

The deployment uses the same adapter image for the frontend, vLLM, and speech
workers. ASR and TTS NIMs run as sidecars in their respective worker pods.

Before applying it:

1. Push the built Dynamo image to a registry available to the cluster.
2. Set the image pull secret names for the Dynamo image and NGC NIM images.
3. Create `ngc-api` with an `NGC_API_KEY` key and `hf-token-secret` with the
   Hugging Face token expected by the vLLM runtime.
4. Provide an RWX PVC named `model-cache`, or change the claim name.

Render the image reference without editing the tracked manifest:

```bash
export PROFILE=latency  # or throughput
export DYNAMO_IMAGE=<registry>/dynamo-riva:<tag>
kubectl kustomize "deploy/${PROFILE}" \
  | sed "s|dynamo-riva:latest|${DYNAMO_IMAGE}|g" \
  | kubectl apply -f -
kubectl get dgd,pods
```

Forward the generated frontend service after all model containers are ready:

```bash
kubectl get services
kubectl port-forward service/<frontend-service> 8000:8000
```

If the Prometheus Operator is installed, apply
`deploy/observability/podmonitor.yaml` to scrape Dynamo component metrics every
15 seconds.

## Use the Blueprint UI

Run the Nemotron Voice Agent Generic Assistant with its Dynamo service profile:

This command requires the companion Dynamo profile in the Blueprint repository.

```bash
docker compose --profile generic-assistant/dynamo up -d
```

Open `http://localhost:7860`. This is the same application, browser UI,
Pipecat pipeline, prompt, VAD, and turn processor used by the direct NIM
profile; only the service endpoints differ.

For a controlled latency comparison, use the Blueprint scaling benchmark for
both `generic-assistant/workstation-perf` and `generic-assistant/dynamo-perf`.
Warm all models, keep the GPU type and workload fixed, and compare ASR TTFB,
LLM TTFT, TTS TTFB, server E2E, client E2E, and throughput.

The recorded NIM image digests are:

- LLM: `sha256:ef711febf0e5b9884f9c37b4868b48d65e1899dc13055541bb746fcb09bac7e0`
- ASR: `sha256:0f01867023d93402fefab2859bdc363cf6f002e37083e5c0ca5d632df30e1850`
- TTS: `sha256:f71667404b3b72a80e24e9a39bf7fc36ac85b11289143cfced9702786ae31f6e`

## Tests

The unit tests mock Riva while exercising the public Dynamo event and audio
contracts:

```bash
python3 -m pip install -r examples/riva_cascaded_pipeline/requirements.txt \
  pytest pytest-asyncio
PYTHONPATH=components/src:lib/bindings/python/src \
  python3 -m pytest -xvv examples/riva_cascaded_pipeline/tests
bash -n examples/riva_cascaded_pipeline/{launch_workers.sh,container/build.sh}
kubectl kustomize examples/riva_cascaded_pipeline/deploy/latency >/dev/null
kubectl kustomize examples/riva_cascaded_pipeline/deploy/throughput >/dev/null
```

For functional validation, run `smoke_speech_loop.py` and confirm that the
transcript matches the synthesized sentence closely enough to recognize the
intended text.
