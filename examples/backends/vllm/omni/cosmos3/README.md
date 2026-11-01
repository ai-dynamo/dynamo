<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cosmos3 with vLLM-Omni

**Experimental.** Serve Cosmos3 Nano through NVIDIA Dynamo's image and video
APIs. One aggregated worker loads the native vLLM-Omni pipeline. Start it in
image mode for text-to-image, or video mode for text-to-video and image-to-video.
Stop the first worker before switching modes on a single GPU.

## Prerequisites

- A Dynamo environment containing this change, compatible Dynamo Python/runtime
  wheels, vLLM `0.31.0`, and vLLM-Omni `0.31.0rc1`.
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
bash examples/backends/vllm/launch/agg_omni_cosmos3_image.sh --no-guardrails
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
bash examples/backends/vllm/launch/agg_omni_cosmos3_video.sh --no-guardrails
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
The `agg_omni_cosmos3_i2v.sh` launcher starts the same video worker; the request's
`input_reference` selects image-to-video generation.

## Select a Model

The launchers and client accept `--model`, defaulting to `nvidia/Cosmos3-Nano`.
Pass the same identifier to both. `DYN_COSMOS_MODEL_PATH` optionally selects a
local checkpoint directory while the model identifier remains the public served
name. Unset that variable when switching to a model downloaded from the Hub, or
update it to the matching checkpoint.

| Checkpoint | Dynamo qualification |
| --- | --- |
| [Cosmos3 Nano](https://huggingface.co/nvidia/Cosmos3-Nano) | The image and short-video configuration documented here |
| [Cosmos3 Super](https://huggingface.co/nvidia/Cosmos3-Super) | Model selection is available; GPU execution has not been qualified in this example |

Super requires a separate memory and parallelism configuration. Consult its
model card before allocating hardware; selecting the model alone does not make
it fit on the Nano test GPU. Extra launcher options, such as
`--enable-layerwise-offload`, `--cfg-parallel-size`, and `--use-hsdp`, are passed
to the Omni worker. Their effectiveness must be tested for the selected model
and hardware.

The three workflow launchers share `agg_omni_cosmos3.sh`. The shared launcher
also accepts `DYN_COSMOS_MODALITY=image` or `video` for direct use.

## Send a JSON Request

Sample requests with structured Cosmos prompts live in
[`launch/cosmos3`](../../launch/cosmos3). They default to Nano and request
inline base64 media. The video samples use 33 frames at 24 FPS; the prompts
describe longer scenes, but these requests generate only short clips.

| File | Worker mode | Endpoint |
| --- | --- | --- |
| `t2i.json` | `image` | `/v1/images/generations` |
| `t2v.json` | `video` | `/v1/videos` |
| `i2v.json` | `video` | `/v1/videos`, with an HTTPS `input_reference` |

With the image worker running, send the image sample:

```bash
curl --fail-with-body -sS "http://localhost:${DYN_HTTP_PORT:-8000}/v1/images/generations" \
  -H 'Content-Type: application/json' \
  --data-binary @examples/backends/vllm/launch/cosmos3/t2i.json \
  | jq -r '.data[0].b64_json' | base64 -d > /tmp/cosmos-sample.png
```

For video, use the video worker, `/v1/videos`, and `t2v.json` or `i2v.json`;
save the decoded payload with an `.mp4` extension. Change the JSON `model`
field if the worker serves a different model. The image-to-video sample fetches
its reference over HTTPS; the Python client above supports client-local images.

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

The pinned checkpoint and engine versions above were tested in a prepared
Arm64 development environment. It used the Dynamo Python source and `1.6.0`
wheel built from this change, the prebuilt `ai-dynamo-runtime` wheel
`1.6.0.dev20261006`, and matching FlashInfer Python/cubin/JIT packages at
`0.7.0.post1` (CUDA 13.0 JIT cache). The runs used eager execution, guardrails
disabled, one GPU, and one output per request. This qualification does not
cover a clean build of the full Dynamo container.

| Workflow | Validation |
| --- | --- |
| Text-to-image | Five requests, including a repeated seed and guidance/step variations, matched native vLLM-Omni pixels exactly |
| Text-to-video | The 33-frame 720p response decoded at 24 FPS and matched native output byte-for-byte after the same VP9 encoding |
| Image-to-video | Two contrasting references with the same prompt and seed produced reference-conditioned videos; both matched native output byte-for-byte after the same encoding |

The three JSON samples above were also compared with native vLLM-Omni in this
configuration. `t2i.json` matched native pixels exactly; `t2v.json` and
`i2v.json` matched native output byte-for-byte after identical VP9 encoding.
The image-to-video sample used its HTTPS reference, with the same downloaded
image supplied to the native baseline.

Video comparisons account for the delivery codec: an MP4 decoded after lossy
VP9 compression need not equal the raw native frames. The example client also
succeeded after deliberate HTTP disconnects in both image and video modes.
Invalid dimensions, FPS, and image references returned HTTP 400/415, and
subsequent valid requests succeeded. Disconnect recovery does not establish
immediate cancellation of GPU work.
These checks establish this configuration, not performance or support for
every Cosmos checkpoint and generation setting.
