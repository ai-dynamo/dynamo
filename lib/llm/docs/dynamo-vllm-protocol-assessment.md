<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo versus vLLM request-contract assessment

Use the [OpenAPI request-contract comparison guide](../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/openapi-request-contract-comparison.md) to compare
Dynamo with vLLM serve. This recipe supplies one recorded vLLM configuration,
not a backend-specific comparator or a claim of complete compatibility.

The guide explains schema-comparison scope and report interpretation; the
[tooling README](../../../scripts/protocol_compatibility/README.md) supplies CLI
installation, arguments, outputs, and exit codes. This recipe owns the concrete
vLLM inputs and optional deployment/recapture procedures below.

## Dynamo schema input

Generate the Dynamo side from the selected source checkout by default:

```sh
cargo run --locked -p dynamo-llm --no-default-features --bin generate-frontend-openapi
```

Pass `docs/frontends/openapi.json` to `assess --dynamo`; no Dynamo listener,
container, model or GPU is needed. Retain the revision, lockfile, command and
checksum as described in the framework-neutral guide. The pinned OpenAI
composition is still required. HTTP capture is an alternative when verifying
the contract exposed by a particular deployment.

## Pinned configuration

The reviewed [profile](../../../scripts/protocol_compatibility/frameworks/vllm-0.30.0.json)
records vLLM 0.30.0, its immutable image digest and source revision, and
Qwen/Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`. It also records
the historical HTTP capture checksum used by the regression fixture. A fresh
capture need not have that checksum: the checksum identifies evidence, not an
acceptance rule for all future vLLM captures.

This is a small text-generation schema probe, not a benchmark. Other models,
tasks or serving flags may expose different contracts and need their own capture.
The generic assessor accepts this profile's identity through `--framework-name`;
it does not load profiles or infer framework-specific schema exceptions.

## Deployment recapture (optional for comparison)

Prerequisites: Linux Docker with GPU support, capacity for the pinned model,
`jq`, the pinned comparison dependencies, and a Dynamo spec with its matching
OpenAI YAML. A published/checked-in spec is sufficient for comparison; this
recipe adds deployment evidence when reproducing a live vLLM capture. It is
caller-owned, not a requirement imposed by the generic acquisition tool.
Use task-specific container names and fresh evidence/output directories.

Set `MODEL_CACHE` to the model-specific Hugging Face cache containing `snapshots/`
and `blobs/`; mount it read-only. Do not mount credentials or an entire home
directory. Choose an available `GPU_DEVICE` and a unique `VLLM_CONTAINER`.

```sh
PROFILE=scripts/protocol_compatibility/frameworks/vllm-0.30.0.json
VLLM_IMAGE=$(jq -r '.image' "$PROFILE")
MODEL_REVISION=$(jq -r '.provenance.dependencies.model_revision' "$PROFILE")
docker run -d --name "$VLLM_CONTAINER" --gpus "device=$GPU_DEVICE" \
  --cpus 6 --memory 16g --shm-size 2g -p 127.0.0.1::8000 \
  -v "$MODEL_CACHE:/model-cache:ro" -e HF_HUB_OFFLINE=1 "$VLLM_IMAGE" \
  "/model-cache/snapshots/$MODEL_REVISION" \
  --served-model-name "$(jq -r '.serve.served_model_name' "$PROFILE")" \
  --max-model-len "$(jq -r '.serve.max_model_len' "$PROFILE")" \
  --gpu-memory-utilization "$(jq -r '.serve.gpu_memory_utilization' "$PROFILE")" \
  --enforce-eager --host 0.0.0.0 --port 8000
docker port "$VLLM_CONTAINER" 8000/tcp
```

Set `VLLM_PORT` to the reported host port. Wait for `/openapi.json` to return
HTTP 200; inspect logs if startup fails. Query `/version` as independent version
evidence and verify the image's revision labels against the profile. The recipe
records eager execution; review both the profile and command if changing it.

```sh
set -euo pipefail
mkdir deployment-evidence
curl --fail "http://127.0.0.1:$VLLM_PORT/version" > deployment-evidence/version.json
jq -e --arg expected "$(jq -r '.provenance.server_version' "$PROFILE")" \
  '.version == $expected' deployment-evidence/version.json

# Keep a whitelist of observed fields; never archive Config.Env or full inspect.
docker inspect "$VLLM_CONTAINER" | jq '.[0] | {
  container_id: .Id, image_id: .Image, requested_image: .Config.Image,
  running: .State.Running, entrypoint: .Config.Entrypoint, command: .Config.Cmd,
  ports: .NetworkSettings.Ports,
  resources: (.HostConfig | {NanoCpus, Memory, ShmSize, DeviceRequests, IpcMode})
}' > deployment-evidence/container.json
jq -e --arg image "$VLLM_IMAGE" --arg port "$VLLM_PORT" \
  '.running and .requested_image == $image and
   ([.ports[]?[]? | .HostPort] | index($port) != null)' \
  deployment-evidence/container.json
docker image inspect "$VLLM_IMAGE" | jq '.[0] | {
  image_id: .Id, repo_digests: .RepoDigests,
  source_revision: .Config.Labels["org.opencontainers.image.revision"]
}' > deployment-evidence/image.json
jq -n --slurpfile profile "$PROFILE" \
  --slurpfile container deployment-evidence/container.json \
  --slurpfile image deployment-evidence/image.json \
  --slurpfile version deployment-evidence/version.json \
  '{declared_profile: $profile[0], observed_container: $container[0],
    observed_image: $image[0], observed_version: $version[0]}' \
  > vllm-metadata.json
python -m scripts.protocol_compatibility acquire \
  --server "$(jq -r '.framework_name' "$PROFILE")" \
  --url "http://127.0.0.1:$VLLM_PORT/openapi.json" \
  --metadata vllm-metadata.json --output-dir captures/vllm
python -m scripts.protocol_compatibility assess \
  --dynamo docs/frontends/openapi.json --framework captures/vllm \
  --framework-name "$(jq -r '.framework_name' "$PROFILE")" \
  --openai openai-884aff95.yaml --oasdiff /absolute/path/to/oasdiff \
  --output-dir assessment-vllm
```

Before acquisition, review the recorded command/resources against the profile,
including model revision, served name, maximum length, eager mode and memory
fraction. Check any observed source label against the expected source revision;
an absent label is an evidence gap, not permission to substitute a tag as proof.
The commands above check version, running state, requested image and published
port; they do not attest a source build or automatically validate every flag.
Review command arguments for secrets before archiving them. The generic tool
preserves `vllm-metadata.json` as unverified caller annotations.

Review `report.md`, the complete request deltas, normalization audit and alias
value probes. Exit 1 means differences or gaps require review, not tool failure.
Retain both raw captures and acquisition metadata. Then stop and remove only
the task container, and verify GPU resources were released:

```sh
docker stop "$VLLM_CONTAINER"
docker rm "$VLLM_CONTAINER"
```

## Regression evidence and interpretation

The small generated fixture retains five field schemas verbatim from the
recorded HTTP capture: chat `user` and `chat_template_kwargs`, message
`reasoning`, and completion `user` and `suffix`. Their original JSON pointers,
image, source revision and full-capture SHA-256 accompany the fragments.
They are not complete request models. Use a full published spec or HTTP capture
for an actual contract assessment.

Tests exercise these vLLM fragments through the generic pipeline. They check
string/null normalization, explicit alias-name matching, and preservation of
value differences with the real pinned comparator. The Dynamo-shaped test
document is synthetic; the base workflow's Rust tests verify runtime aliases.
Neither successful fragment tests nor name matches establish backend parity.

Regenerate only from the reviewed full capture, and check freshness with:

```sh
python -m scripts.protocol_compatibility.tests.composition.vllm_fixture \
  --source captures/vllm/openapi.raw
python -m scripts.protocol_compatibility.tests.composition.vllm_fixture \
  --source captures/vllm/openapi.raw --check
```

The generator rejects a capture whose bytes do not match the profile's reference
checksum. Updating that reference requires review, not a checksum refresh to
silence a failing check. Ordinary CI runs the reduced regression tests without
requiring the archived full capture; `--check` is an explicit evidence check.

The recorded full assessment has two endpoint-level diff groups and seven known
coverage gaps. These are not two individual incompatibilities. The shared
schema/Serde checks cover 48 aligned cases and six disclosed gap witnesses.
Explicit alias metadata covers the two spellings above when present in Dynamo's
capture; their value schemas can still differ structurally or semantically.

## Updating vLLM

Select a new immutable image and record its actual source/version evidence.
Review the model and serving configuration, refresh the selected specs (or
recapture both servers for deployment-specific evidence), and run the
same generic assessment plus Rust/schema fidelity checks. Update the profile,
reviewed reference capture and generated fragments together when the regression
baseline changes. Keep raw evidence and record newly discovered differences.
Do not add schema rewrites simply to make the vLLM comparison pass.

This recipe does not schedule GPU CI, automatically adopt releases, or cover
responses, streaming, forwarding or inference behavior.

## Optional Dynamo deployment evidence

Ordinary comparison can use the existing generator's file output described in
the [comparison guide](../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/openapi-request-contract-comparison.md#schema-inputs-and-provenance). When verifying a deployment's
HTTP export, build and package the same helper from the exact source and lockfile:

```sh
cargo build --locked -p dynamo-llm --no-default-features --bin generate-frontend-openapi
mkdir image-context
cp target/debug/generate-frontend-openapi image-context/generate-frontend-openapi
strip image-context/generate-frontend-openapi
cp scripts/protocol_compatibility/frameworks/Dockerfile.dynamo image-context/Dockerfile
docker build --platform linux/amd64 --provenance=false -t "$DYNAMO_IMAGE_TAG" image-context
```

Retain compiler version, source revision/patch, Cargo lockfile and original/stripped
binary hashes. Publish the image through the caller's approved registry procedure,
verify its manifest digest, and set `DYNAMO_IMAGE` to the digest-qualified reference.
Packaging a binary is not proof of a reproducible source build.

```sh
docker run -d --name "$DYNAMO_CONTAINER" -p 127.0.0.1::8000 "$DYNAMO_IMAGE"
docker port "$DYNAMO_CONTAINER" 8000/tcp
```

Record the assigned `DYNAMO_PORT`, wait for readiness, and inspect the running
container using the same whitelist/image/port checks above. Verify the source
lockfile against the reviewed composition manifest's `dependencies` and preserve
the build evidence. If they differ, review the manifest/spec pair before using
composition. Caller annotations do not override the pinned OpenAI checksum or
schema-patch guards.

```sh
python -m scripts.protocol_compatibility acquire \
  --server dynamo --url "http://127.0.0.1:$DYNAMO_PORT/openapi.json" \
  --metadata dynamo-metadata.json --output-dir captures/dynamo
docker stop "$DYNAMO_CONTAINER"
docker rm "$DYNAMO_CONTAINER"
```

Create `dynamo-metadata.json` from reviewed build/deployment evidence before
acquisition, or omit `--metadata` when keeping that evidence separately.
The helper uses the actual HTTP router without workers or model weights; this
recapture uses no GPU and does not establish inference or runtime fidelity.
