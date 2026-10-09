<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cosmos3 Nano with vLLM-Omni

**Experimental.** Serve Cosmos3 Nano through NVIDIA Dynamo's image and video
APIs. One aggregated worker loads the native vLLM-Omni pipeline. Start it in
image mode for text-to-image, or video mode for text-to-video and image-to-video.
Stop the first worker before switching modes on a single GPU.

## Prerequisites

- A Dynamo environment containing this change, compatible Dynamo Python/runtime
  wheels, vLLM `0.30.0`, and vLLM-Omni `0.30.0rc1`.
- One GPU with enough memory for the model and generation. Qualification used
  one GH200 with 95.6 GiB GPU-visible memory; this is not a minimum
  memory claim. Smaller devices and offload configurations need separate tests.
- `requests`, Pillow, PyAV, and an imageio/FFmpeg installation with the
  `libvpx-vp9` encoder for MP4 output.
- Access to the [Cosmos3 Nano checkpoint](https://huggingface.co/nvidia/Cosmos3-Nano).

The GH200 development environment installs vLLM-Omni manually. Standard Dynamo
container builds skip vLLM-Omni on Arm64. This example does not imply that an
unmodified released image includes this setup.

Run the commands from the Dynamo repository root inside the prepared environment.
Download the pinned checkpoint and select local discovery and transport:

```bash
export DYN_COSMOS_MODEL_PATH="$(hf download nvidia/Cosmos3-Nano \
  --revision e59a53c25979a090fa8706c9acc0c254a6e89b92)"
export DYN_NAMESPACE=cosmos3-demo
export DYN_DISCOVERY_BACKEND=file
export DYN_REQUEST_PLANE=tcp
export DYN_EVENT_PLANE=zmq
export DYN_FILE_KV="$(mktemp -d)"
```

The launcher uses eager execution. It preserves native guardrail settings by
default. The development commands below explicitly pass `--no-guardrails`;
they do not load or run the optional Cosmos guardrail models. To use guardrails,
install the native Cosmos guardrail dependencies and checkpoints, and omit that
flag. Guardrails-enabled execution requires separate qualification.

## Generate an Image

Start the image worker:

```bash
DYN_COSMOS_MODALITY=image \
  bash examples/backends/vllm/launch/agg_omni_cosmos3.sh --no-guardrails
```

Wait for the worker to register `nvidia/Cosmos3-Nano` at `/v1/models`. In another
terminal in the same environment, send a request:

```bash
python examples/backends/vllm/omni/cosmos3/generate.py \
  --mode image \
  --prompt "A red car beside a mountain at sunset" \
  --output /tmp/cosmos-image.png
```

The client requests one 1024×1024 image with 50 steps, guidance scale 7, and seed
42. It verifies the PNG format and dimensions before saving the response.

## Generate a Video

Stop the image launcher. Start the video worker with the same environment:

```bash
DYN_COSMOS_MODALITY=video \
  bash examples/backends/vllm/launch/agg_omni_cosmos3.sh --no-guardrails
```

After the worker registers, request a short clip:

```bash
python examples/backends/vllm/omni/cosmos3/generate.py \
  --mode video \
  --prompt "A car driving along a coastal road. The camera moves smoothly forward." \
  --output /tmp/cosmos-video.mp4
```

The client requests 33 frames at 1280×720, 24 FPS, 35 steps, guidance scale 6,
and seed 42. It decodes every frame and checks the dimensions, frame count, and
frame rate before saving the MP4. These settings produce a 1.375-second clip;
they are a short-clip example, not Cosmos's native 189-frame video default.

## Animate a Reference Image

Keep the video worker running. Supply a client-local image:

```bash
python examples/backends/vllm/omni/cosmos3/generate.py \
  --mode video \
  --prompt "A car driving along a coastal road. The camera moves smoothly forward." \
  --input-reference "$DYN_COSMOS_MODEL_PATH/assets/example_i2v_input.jpg" \
  --output /tmp/cosmos-reference-video.mp4
```

The client encodes the image as a PNG data URI in `input_reference`. Dynamo
loads it and passes the PIL image in `multi_modal_data.image`; vLLM-Omni
performs reference preprocessing, latent conditioning, and generation.

## Request Controls and Limits

The client uses `/v1/images/generations` for images and `/v1/videos` for both
video workflows. It supplies `seed`, `num_inference_steps`, `guidance_scale`,
and `negative_prompt` under `nvext`; video requests additionally supply `fps`
and `num_frames`. Width and height use the top-level `size` field.

Set frame count and FPS explicitly when comparing native and Dynamo video
generation. Cosmos rounds video frame counts upward to `4k + 1` for temporal
compression. Requesting 33 frames avoids rounding. The launcher sets the
fallback playback rate to 24 FPS, matching Cosmos; it does not change the
native pipeline or its scheduler.

The client uses `b64_json`, so the returned media does not require a separate
HTTP file server. URL responses use the configured media storage; a `file://`
URL refers to the worker's filesystem. Override the frontend port with
`DYN_HTTP_PORT` and pass the corresponding `--url` to the client. Set
`DYN_SYSTEM_PORT` to choose the worker's health/metrics port.

The initial scope is Nano on one aggregated worker. Cosmos3 Super, distilled
checkpoints, guardrails-enabled runs, sound, video-to-video, actions, and
disaggregated serving are separate qualification targets.

## Validated Configuration

The pinned checkpoint and engine versions above were tested with matched
Dynamo Python/runtime wheels (`1.6.0.dev20261006`) and the source changes in
this example. The runs used eager execution, guardrails disabled, one GPU,
and one output per request.

| Workflow | Validation |
| --- | --- |
| Text-to-image | Five requests, including a repeated seed and guidance/step variations, matched native vLLM-Omni pixels exactly |
| Text-to-video | The 33-frame 720p response decoded at 24 FPS and matched native output after the same VP9 encoding |
| Image-to-video | Two contrasting references with the same prompt and seed produced reference-conditioned videos; both matched native output after the same encoding |

Video comparisons account for the delivery codec: an MP4 decoded after lossy
VP9 compression need not equal the raw native frames. The example client also
succeeded after deliberate HTTP disconnects in both image and video modes.
These checks establish this configuration, not performance or support for
every Cosmos checkpoint and generation setting.
