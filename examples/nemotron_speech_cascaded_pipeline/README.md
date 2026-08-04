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

# Nemotron Speech adapters for a cascaded voice agent

This example exposes NVIDIA Speech NIM microservices serving Nemotron models
through Dynamo's standard OpenAI-compatible APIs. A voice application such as Pipecat remains responsible
for the ASR -> LLM -> TTS cascade, conversation state, turn taking, tools, and
barge-in.

NVIDIA documents these containers as
[NVIDIA Speech NIM microservices](https://docs.nvidia.com/nim/speech/latest/index.html),
which serve Nemotron Speech models. The adapter intentionally uses `nvidia-riva-client`, which the current Speech
NIM documentation specifies for gRPC access.

```mermaid
flowchart LR
    Browser["Browser UI"]

    subgraph PipecatContainer["Container: Generic Assistant"]
        direction TB
        Pipecat["Same Pipecat pipeline"]
        STTClient["OpenAI Realtime STT<br/>WebSocket"]
        LLMClient["NvidiaLLMService<br/>OpenAI HTTP"]
        TTSClient["OpenAI-compatible TTS<br/>Streaming HTTP"]
    end

    subgraph DGD["DynamoGraphDeployment: nemotron-speech-cascaded"]
        direction LR

        subgraph FrontendPod["Kubernetes Pod: Frontend"]
            Frontend["Container: main<br/>Published Dynamo Frontend<br/>OpenAI APIs and routing"]
        end

        subgraph ASRPod["Kubernetes Pod: SpeechAsrWorker"]
            ASRWorker["Container: main<br/>Custom Dynamo ASR adapter<br/>CPU only"]
            ASR["Container: asr-nim<br/>Same Nemotron ASR NIM<br/>1 GPU"]
        end

        subgraph LLMPod["Kubernetes Pod: VllmWorker"]
            LLMWorker["Container: main<br/>Published Dynamo vLLM worker<br/>FP8, TP=1, 1 GPU"]
        end

        subgraph TTSPod["Kubernetes Pod: SpeechTtsWorker"]
            TTSWorker["Container: main<br/>Custom Dynamo TTS adapter<br/>CPU only"]
            TTS["Container: tts-nim<br/>Same Magpie TTS NIM<br/>1 GPU"]
        end
    end

    Browser <--> Pipecat

    Pipecat --> STTClient
    STTClient -->|"/v1/realtime"| Frontend
    Frontend --> ASRWorker
    ASRWorker <-->|"Speech NIM streaming gRPC"| ASR

    Pipecat --> LLMClient
    LLMClient -->|"/v1/chat/completions"| Frontend
    Frontend --> LLMWorker

    Pipecat --> TTSClient
    TTSClient -->|"/v1/audio/speech"| Frontend
    Frontend --> TTSWorker
    TTSWorker <-->|"Speech NIM online gRPC"| TTS

    classDef client fill:#eef6ff,stroke:#2563eb,color:#111827
    classDef dynamo fill:#fff7e6,stroke:#b45309,color:#111827
    classDef nim fill:#edf9f0,stroke:#15803d,color:#111827
    class Browser,Pipecat,STTClient,LLMClient,TTSClient client
    class Frontend,ASRWorker,LLMWorker,TTSWorker dynamo
    class ASR,TTS nim

    style PipecatContainer fill:#f8fafc,stroke:#2563eb,stroke-width:2px
    style DGD fill:#f8fafc,stroke:#1d4ed8,stroke-width:4px
    style FrontendPod fill:#ffffff,stroke:#64748b,stroke-width:2px,stroke-dasharray:5 5
    style ASRPod fill:#ffffff,stroke:#64748b,stroke-width:2px,stroke-dasharray:5 5
    style LLMPod fill:#ffffff,stroke:#64748b,stroke-width:2px,stroke-dasharray:5 5
    style TTSPod fill:#ffffff,stroke:#64748b,stroke-width:2px,stroke-dasharray:5 5
```

The thick blue border is the DGD, dashed borders are Kubernetes pods, and each
labeled inner box is a container. The Generic Assistant container runs outside
the DGD.

The deployment matches the Generic Assistant recipe in the Nemotron Voice
Agent Blueprint so the same UI and workload can compare direct NIM access with
Dynamo routing:

| Stage | Model |
| --- | --- |
| ASR | Nemotron ASR Streaming 1.2.0, English streaming profile |
| LLM | NVIDIA Nemotron 3 Nano 30B A3B FP8 |
| TTS | Magpie TTS Multilingual 1.8.0 |

## Deploy on Kubernetes

The manifest creates a Dynamo frontend, a vLLM worker, and separate ASR and TTS
worker pods. Each speech worker runs a Speech NIM as a sidecar. The deployment
uses three GPUs in total, one for each model.

### Prerequisites

- A Kubernetes cluster with at least three NVIDIA GPUs and the
  [Dynamo Kubernetes Platform](../../docs/fern/kubernetes/quickstart.mdx)
  installed.
- Docker and `envsubst`, plus access to a registry that the cluster can pull
  from.
- An NGC API key with access to the ASR and TTS NIM images.
- A Hugging Face token with access to the Nemotron LLM.
- A ReadWriteMany storage class for the shared model cache.

Run all commands from the Dynamo repository root. Set the deployment values
once:

```bash
export NAMESPACE=voice-agent
export DYNAMO_RUNTIME_VERSION=<compatible-published-version>
export DYNAMO_FRONTEND_IMAGE="nvcr.io/nvidia/ai-dynamo/dynamo-frontend:${DYNAMO_RUNTIME_VERSION}"
export DYNAMO_VLLM_IMAGE="nvcr.io/nvidia/ai-dynamo/vllm-runtime:${DYNAMO_RUNTIME_VERSION}"
export CUSTOM_IMAGE_REGISTRY=<registry-host>
export CUSTOM_IMAGE_REPOSITORY=<project>
export CUSTOM_SPEECH_ADAPTER_IMAGE="${CUSTOM_IMAGE_REGISTRY}/${CUSTOM_IMAGE_REPOSITORY}/dynamo-nemotron-speech-adapter:${DYNAMO_RUNTIME_VERSION}"
export CUSTOM_IMAGE_REGISTRY_USER=<username>
export CUSTOM_IMAGE_REGISTRY_PASSWORD=<password>
export NGC_API_KEY=<ngc-api-key>
export HF_TOKEN=<hugging-face-token>
export RWX_STORAGE_CLASS=<rwx-storage-class>
```

Use the same Dynamo runtime version for the two published images and the custom
adapter image. Its semantic tag lets the Dynamo operator derive compatibility
directly from each component's main image. The custom image can use any OCI
registry; the published Dynamo images are pulled directly from NVCR.

### 1. Build and push the adapter image

The speech worker's main container is a small CPU-only adapter. It derives from
the published Dynamo frontend image, which provides the Dynamo runtime without
vLLM, and adds the Riva client and this example's adapter code:

```bash
./examples/nemotron_speech_cascaded_pipeline/container/build.sh
docker push "${CUSTOM_SPEECH_ADAPTER_IMAGE}"
```

The selected published version must contain the Dynamo realtime and streaming
speech support required by this example. Use branch-built images only when
validating changes that have not reached a published release.

### 2. Create the namespace and credentials

```bash
kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml \
  | kubectl apply -f -

kubectl create secret docker-registry custom-adapter-image-pull-secret \
  --namespace "${NAMESPACE}" \
  --docker-server "${CUSTOM_IMAGE_REGISTRY}" \
  --docker-username "${CUSTOM_IMAGE_REGISTRY_USER}" \
  --docker-password "${CUSTOM_IMAGE_REGISTRY_PASSWORD}" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret docker-registry ngc-secret \
  --namespace "${NAMESPACE}" \
  --docker-server nvcr.io \
  --docker-username '$oauthtoken' \
  --docker-password "${NGC_API_KEY}" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic ngc-api \
  --namespace "${NAMESPACE}" \
  --from-literal=NGC_API_KEY="${NGC_API_KEY}" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic hf-token-secret \
  --namespace "${NAMESPACE}" \
  --from-literal=HF_TOKEN="${HF_TOKEN}" \
  --dry-run=client -o yaml | kubectl apply -f -
```

The custom registry credentials are only needed for a private adapter image.
For a public image, omit that secret and remove
`custom-adapter-image-pull-secret` from the speech worker pod templates.

### 3. Create the shared model cache

The three model pods may run on different nodes, so the cache must support
`ReadWriteMany`:

```bash
kubectl apply --namespace "${NAMESPACE}" -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: ${RWX_STORAGE_CLASS}
  resources:
    requests:
      storage: 200Gi
EOF
```

### 4. Deploy the graph

Render the image environment variables while applying the tracked manifest:

```bash
envsubst '${DYNAMO_FRONTEND_IMAGE} ${DYNAMO_VLLM_IMAGE} ${CUSTOM_SPEECH_ADAPTER_IMAGE}' \
  < examples/nemotron_speech_cascaded_pipeline/deploy/agg.yaml \
  | kubectl apply --namespace "${NAMESPACE}" -f -
```

Watch the model pods start. Initial NIM and LLM downloads can take several
minutes:

```bash
kubectl get dgd nemotron-speech-cascaded --namespace "${NAMESPACE}"
kubectl get pods --namespace "${NAMESPACE}" \
  --selector nvidia.com/dynamo-graph-deployment-name=nemotron-speech-cascaded --watch
```

After the pods appear, wait for every container to become ready:

```bash
kubectl wait --namespace "${NAMESPACE}" --for=condition=Ready pod \
  --selector nvidia.com/dynamo-graph-deployment-name=nemotron-speech-cascaded \
  --timeout=45m
```

If a pod does not become ready, inspect its events and container logs:

```bash
kubectl describe pod --namespace "${NAMESPACE}" \
  --selector nvidia.com/dynamo-graph-deployment-name=nemotron-speech-cascaded
kubectl logs --namespace "${NAMESPACE}" \
  --selector nvidia.com/dynamo-component=SpeechAsrWorker \
  --container asr-nim --tail=100
```

### 5. Connect and validate

Forward the generated frontend service:

```bash
kubectl port-forward --namespace "${NAMESPACE}" --address 0.0.0.0 \
  service/nemotron-speech-cascaded-frontend 8000:8000
```

Keep this command running for validation and the Blueprint demo. The broader
bind lets a local Docker container reach the listener; do not expose port 8000
through an external firewall because it does not provide authentication.

The deployment exposes:

- `ws://localhost:8000/v1/realtime`, transcription using 24 kHz PCM
- `http://localhost:8000/v1/chat/completions`
- `http://localhost:8000/v1/audio/speech`, streaming 24 kHz PCM

In another terminal, install the small client dependency and exercise TTS and
ASR together:

```bash
python3 -m pip install aiohttp
python3 examples/nemotron_speech_cascaded_pipeline/smoke_speech_loop.py
```

The check reports TTS TTFB, ASR first-transcript latency, PCM RMS, and the final
transcript. It fails on an API error, silent audio, or an empty transcript.

The TTS adapter requires a Dynamo runtime with streaming
`/v1/audio/speech` support. The realtime ASR adapter disables server VAD
because Pipecat's local VAD and Smart Turn processors commit the input audio.

## Run the Blueprint UI and Pipecat

The [Nemotron Voice Agent Blueprint](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent)
provides the browser UI and the Generic Assistant Pipecat pipeline. The
temporary [companion Dynamo profile](https://github.com/ptarasiewiczNV/nemotron-voice-agent/tree/e33d9bb86016239a35fdae2d1360dd4d2019257e)
adds OpenAI-compatible clients for this deployment. Access to that private fork
is required until the companion change is upstreamed.

There is no separate Pipecat command: the Blueprint Compose service runs both
Pipecat and the UI. Run the following steps on the machine that has Kubernetes
and Docker access.

### 1. Keep the Dynamo frontend reachable from Docker

Keep the port-forward from deployment step 5 running. The Compose container
resolves `host.docker.internal` to the Docker host, where that listener
exposes Dynamo on port 8000. If you stopped it, run the command from step 5
again before starting the Blueprint.

### 2. Start the Blueprint application

In a second terminal:

```bash
git clone git@github.com:ptarasiewiczNV/nemotron-voice-agent.git
cd nemotron-voice-agent
git checkout e33d9bb86016239a35fdae2d1360dd4d2019257e

cat > .env <<'EOF'
NVIDIA_API_KEY=not-used
TRANSPORT_SELECTION=websocket
PIPELINE_TLS=true
EOF

docker compose --profile generic-assistant/dynamo up --detach --build
docker compose --profile generic-assistant/dynamo ps
curl --fail --insecure https://localhost:7860/health
```

`TRANSPORT_SELECTION=websocket` makes the demo usable through a single SSH
tunnel. The Dynamo profile connects ASR, LLM, and TTS to
`host.docker.internal:8000`; it does not start duplicate model containers.

Follow the Pipecat logs while testing:

```bash
docker compose --profile generic-assistant/dynamo \
  logs --follow generic-assistant-dynamo
```

### 3. Open the UI

For a local Kubernetes/Docker host, open `https://localhost:7860`. For a remote
host, create the tunnel from your workstation:

```bash
ssh -N -L 7860:127.0.0.1:7860 <user>@<remote-host>
```

Then open `https://localhost:7860`, accept the development certificate, allow
microphone access, and start a conversation. This uses the same browser UI,
prompt, Pipecat pipeline, VAD, and turn processor as the direct NIM profile;
only the model service path differs.

Stop the application with:

```bash
docker compose --profile generic-assistant/dynamo down
```

Stop the Kubernetes port-forward with `Ctrl-C` in its terminal.

## Tests

The unit tests mock the Speech NIM services while exercising the public Dynamo
event and audio contracts:

```bash
python3 -m pip install -r examples/nemotron_speech_cascaded_pipeline/requirements.txt \
  pytest pytest-asyncio
PYTHONPATH=components/src:lib/bindings/python/src \
  python3 -m pytest -xvv examples/nemotron_speech_cascaded_pipeline/tests
bash -n examples/nemotron_speech_cascaded_pipeline/{launch_workers.sh,container/build.sh}
pre-commit run check-yaml --files examples/nemotron_speech_cascaded_pipeline/deploy/agg.yaml
```

For functional validation, run `smoke_speech_loop.py` and confirm that the
transcript matches the synthesized sentence closely enough to recognize the
intended text.
